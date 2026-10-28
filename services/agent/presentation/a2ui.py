"""Compose the A2UI v0.9 risk canvas and citation metadata for an answer.

This module is the single presentation adapter for the agent's HTTP contract.
The canvas and the citation metadata are composed HERE — in the agent, where
the answer, the guardrails, and the retrieved evidence meet — never downstream
in the BFF or the browser. The caller (Django) passes the envelope through
unchanged; the browser renders it.

Three rules this module exists to enforce:

1. **Tools return JSON, the agent composes UI.** `prediction.py` deliberately
   never emits A2UI — a tool that returns UI is welded to one presentation
   layer and stops working from Claude Desktop or CI. The translation happens
   here, on the agent side, where a presentation choice belongs.

2. **Never render without a text fallback (R8).** Every return carries
   `fallback_text`. If the CDN is down, the renderer throws, or the spec
   version drifts, the caller still has something true to show. A demo that
   renders nothing is worse than one that renders plainly.

3. **The model's citation numbers are advisory.** They are renumbered
   deterministically (first appearance order) and the SourceCard resolves its
   passage by section intent, not by the model's numbering — see
   `services/agent/citations.py`.

The payload shape is v0.9, which is *not* backward compatible with the v0.8
examples that dominate search results. In v0.9 `component` is a string and
properties sit inline. Feeding v0.8 shapes to a v0.9 renderer raises nothing —
it logs "Component implementation not found for type: [object Object]" and
draws an empty box. The tests in tests/agent/test_a2ui.py pin the v0.9 shape
so that regression is caught here rather than in a browser.

The canvas uses the custom components registered in the site's clinical bundle
(static/js/a2ui_clinical_components.js): RiskInfluencers, LiteratureList,
SourceCard. The surface's `catalogId` must equal the combined catalog id the
front-end registers, so one surface can use basic AND custom components
together.
"""

from services.agent.postprocess.citations import (
    citation_remap,
    first_citation,
    intent_sections,
    renumber_citations,
)
from services.agent.presentation.domain import (
    GENERAL_INPATIENT,
    NON_ADMISSION,
    _usable_articles,
    is_matrix_eligible,
    resolve_phenotype,
)

# The citation records live in `note_presentation`. Imported by name here because
# this is where the rest of the package — and the tests — reach for them:
# `guardrail` reads `_usable_articles`, and the canvas and the composer call
# `resolve_sources`, `_narrative` and `_usable_articles` directly.
# The canvas lives in `risk_canvas`; the composer still calls into it, and the
# tests reach `_draws_matrix` here.
from services.agent.presentation.canvas import (  # noqa: F401 — re-exported
    _draws_matrix,
    _usable_estimate,
    compose_risk_canvas,
)

from services.agent.presentation import (  # noqa: F401 — re-exported
    A2UI_VERSION,
    AUDIENCE,
    CATALOG_ID,
    SURFACE_ID,
)


def admission_eligible(tool_calls: list[dict]) -> bool:
    """Whether the record this turn is about is an admission at all.

    Narrow on purpose: this is *not* `is_matrix_eligible`, which additionally requires a
    curated phenotype bucket before the comparative table may be drawn. Using that wider
    rule suppressed the risk widget for every admission whose condition had not been
    curated — most of the real corpus — which is a different question from whether an
    inpatient readmission estimate applies to the record.

    Defaults to True when no admission is named: a literature or free-text turn without a
    `hadm_id` is not making an inpatient claim, so there is nothing to suppress.
    """
    for call in tool_calls or []:
        hadm = (call.get("args") or {}).get("hadm_id")
        if isinstance(hadm, int) and not isinstance(hadm, bool):
            return hadm not in NON_ADMISSION
    return True


def predict_payload(tool_calls: list[dict]) -> dict | None:
    """The last `predict_readmission` response in a run, or None.

    The last call wins when several ran — it reflects the final state of the
    conversation. None means the agent answered without predicting (a meds or
    summarize question), which composes the no-estimate canvas.
    """
    for call in reversed(tool_calls or []):
        if call.get("name") == "predict_readmission":
            return call.get("response") or None
    return None


def rag_payload(tool_calls: list[dict]) -> dict | None:
    """The retrieval evidence for this run, as ONE passage array.

    The prompt numbers citations GLOBALLY across calls — "the first call's passages are
    ^[1], ^[2] …; the second call's passages continue after them" — so the array a
    marker indexes has to be that same concatenation. This returned only the FIRST
    retrieval call's payload, which meant a marker pointing into the second call
    resolved against an unrelated passage or clamped to the end of the first
    (Phase 01 audit, 2026-10-22; `/ehr` is the case, with its sectioned call plus the
    record-wide fallback).

    A single call is returned untouched, which is every command turn and most free-text
    ones — the merge only exists for the multi-call turn it was always wrong for.
    """
    rag_calls = [call for call in tool_calls or []
                 if call.get("name") in ("rag_search", "rag_search_sections",
                                         "discharge_medications")
                 and isinstance(call.get("response"), dict)]
    if not rag_calls:
        return None
    if len(rag_calls) == 1:
        return rag_calls[0]["response"]

    merged = dict(rag_calls[0]["response"])
    passages: list = []
    for call in rag_calls:
        passages.extend(call["response"].get("passages") or [])
    merged["passages"] = passages
    merged["returned"] = len(passages)
    return merged


def lab_trend_payload(tool_calls: list[dict]) -> dict | None:
    """The `get_lab_trends` response for this run, or None.

    This run, not this conversation. `compose_presentation` hands this turn's calls here
    while the citation layer and the scope read the whole conversation's evidence: a
    series the transcript already holds is quoted in prose and cited, and it is not drawn
    again. The card answers "show me the course", which is a question this turn either
    asked or did not.

    None for a failed call as well as for no call: the card draws a course, and a course
    with no values in it is not a shorter answer, it is a wrong one. The turn still says
    what happened — the tool's error reaches the model like any other result.
    """
    for call in tool_calls or []:
        if call.get("name") != "get_lab_trends":
            continue
        response = call.get("response") or {}
        if isinstance(response, dict) and not response.get("error"):
            return response
    return None


def literature_payload(tool_calls: list[dict]) -> dict | None:
    """The `search_literature` response for this run, or None.

    First match rather than last, unlike `predict_payload`: a run normally makes one
    search, and if a model made two the answer was written against the first.

    A failed search is returned as it stands. Its `error` key is what tells the
    composer and the caption that the tool did not answer, which is a different
    statement from "nothing was found" — and dropping it here would collapse the
    two into the same empty canvas.
    """
    for call in tool_calls or []:
        if call.get("name") == "search_literature":
            return call.get("response") or None
    return None


#: What a turn's answer rested on. The console labels the turn with this, so a reader
#: never mistakes a claim about the patient for a claim about the literature.
SCOPE_PATIENT = "patient_chart"
SCOPE_GENERAL = "general_evidence"


def turn_scope(tool_calls: list[dict]) -> str | None:
    """What this turn's evidence was, or None when it rested on nothing.

    A tool call that carried a `hadm_id` read the patient's record; a call without one
    did not. That is the whole test, and it is read off the arguments the tools were
    actually given rather than off a list of tool names — so a tool added later is
    classified by what it did rather than by being remembered here.

    None when no tool ran. A redirect ("that is /risk's answer to give") is a complete
    answer that rests on no evidence, and labelling it either way would claim a basis
    it does not have.
    """
    if not tool_calls:
        return None
    for call in tool_calls:
        args = call.get("args") or {}
        if args.get("hadm_id") is not None:
            return SCOPE_PATIENT
    return SCOPE_GENERAL


def _note_sections(tool_calls: list[dict]) -> dict[str, str]:
    """Every note section the evidence carries, section name -> its text.

    Read off the retrieval payloads rather than fetched: this is the chart the turn
    actually read, which is the only chart the composer is entitled to describe a patient
    from. A passage labelled with a section is that section; two passages labelled alike
    are joined, because a whole-note chunk and a sectioned one can both land on the same
    retrieval and the classifier should see the text once.
    """
    sections: dict[str, str] = {}
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        response = call.get("response")
        if not isinstance(response, dict):
            continue
        for passage in response.get("passages") or []:
            if not isinstance(passage, dict):
                continue
            name = str(passage.get("section") or "").strip()
            text = str(passage.get("text") or "").strip()
            if not name or not text:
                continue
            sections[name] = f"{sections[name]} {text}".strip() if name in sections else text
    return sections


def turn_phenotype(tool_calls: list[dict],
                   evidence: list[dict] | None = None) -> str:
    """The condition this turn's patient has, resolved from the chart it read.

    Read from the note sections the evidence carries, in the order a clinician reads a
    discharge summary (see `admission_phenotypes.resolve_phenotype`), and NOT from the
    admission's identifier. The reviewed map answered that question for the 84 rows it
    carried and answered nothing for any other admission — and "nothing" reached the
    surface as a table that was simply absent, which no reader can tell apart from a
    patient who cannot be compared at all. A patient admitted tomorrow has a note, so a
    patient admitted tomorrow resolves.

    `evidence` is the whole conversation's calls, because a `/pubmed` turn reads no chart
    of its own and a question about the patient's condition is still about this patient.
    What it must not be is the answer to a different question: a condition asserted from
    an identifier is exactly the hardcoded claim this replaces, so a turn with no chart
    anywhere in reach resolves nothing and draws no comparison. That is a real cost — a
    literature turn opened on a fresh thread shows records alone — and it is the honest
    one: without a chart there is no evidence for whose condition the cohorts are.

    An unresolved INPATIENT resolves to `GENERAL_INPATIENT` rather than to nothing, and
    the difference matters: the vocabulary failing to name a condition is a limit of the
    vocabulary, while a stay is still a stay and the all-cause hospital-wide rate is still
    the right thing to hold it against. A record that was never a stay resolves nothing,
    because a readmission rate does not describe it.
    """
    calls = [*(evidence or []), *(tool_calls or [])]
    hadm_id = next(
        (int((call.get("args") or {})["hadm_id"]) for call in calls
         if isinstance(call, dict)
         and isinstance((call.get("args") or {}).get("hadm_id"), int)
         and not isinstance((call.get("args") or {}).get("hadm_id"), bool)),
        None,
    )
    if hadm_id is not None and not is_matrix_eligible(hadm_id):
        return ""
    return resolve_phenotype(
        _note_sections(calls),
        fallback=GENERAL_INPATIENT if hadm_id is not None else "",
    )


def compose_presentation(question: str, answer: str,
                         tool_calls: list[dict],
                         command_payload: dict | None = None,
                         command: str | None = None,
                         context_grounded: bool = False,
                         evidence: list[dict] | None = None) -> dict:
    """The full presentation contract for an answered question.

    One call produces every field the browser consumes, so the answer's
    evidence semantics are decided in exactly one place (here) and the BFF is
    a pass-through:

      - answer: the guarded answer with citations renumbered to
        first-appearance order
      - sources: one resolved {cite, section, text, query} per citation in
        the answer — the browser looks a clicked footnote up by number
      - scope: what the answer rested on, or None when it rested on nothing —
        which includes a chart read that resolved no note citation, reported as
        `scope_suppressed` rather than asserted over prose it cannot evidence
      - a2ui: the composed risk-canvas envelope; its SourceCard is sources[0]

    citation_map and intent_sections are inputs to that resolution and are
    returned for tests and diagnostics; the browser does not need them.

    `command_payload` is a command's own structured result when it has one — `/meds`
    extracts rows the canvas draws as a grid. It is passed through rather than
    recomposed, because the rows were verified against the note text in the command
    and re-deriving them here would be a second implementation of that check.

    `context_grounded` is the caller saying this turn also rests on the patient record
    through a channel this layer cannot see: the pre-flight manifest that answered an
    administrative question, or the replayed turns' chart calls that a follow-up is
    quoting. It feeds `verified` and nothing else — evidence the composer never saw
    cannot change what it draws, but it does change whether the answer was checked
    against the record.

    Pure — no I/O, no clock, no randomness — so it is fully testable without
    a browser or a live endpoint.
    """
    renumbered = renumber_citations(answer)
    remap = citation_remap(answer)
    intent = intent_sections(question)
    # `evidence` is every call the conversation has shown the model — the replayed turns
    # first, then this one — because that is the array the answer's markers are numbered
    # against (`graph._render_tool_result` counts continuously across both). Resolving
    # against this turn's calls alone left a follow-up that quoted the chart with a
    # marker that pointed at nothing (TF-10, 2026-10-23).
    evidence_calls = list(evidence) if evidence is not None else tool_calls
    rag = rag_payload(evidence_calls)
    # The literature surface is THIS turn's search and nothing else.
    #
    # Read from the conversation, a `search_literature` call made three turns ago came
    # back on every turn after it: `resolve_sources` bound `^[1]` to a PubMed record the
    # reader was not being shown, the badge flipped to General Medical Evidence over an
    # answer about the discharge note, and `LiteratureList` re-rendered under medication
    # instructions (90000058, 2026-10-26). A search is an event, not a state: what this
    # turn answers from is what this turn looked up.
    literature = literature_payload(tool_calls)
    # The lab card reads THIS turn's calls, not the conversation's. `evidence_calls` is
    # the right array for what the answer rests on and for how its markers are numbered
    # (TF-10, 2026-10-23), but a chart is not a provenance statement — it is a drawing
    # of a course this turn read. Composed from the replay, a follow-up about anything
    # at all — the discharge diagnosis, a medication — redrew the sparkline panel of a
    # `/labs` turn from three turns ago, and each extra `/labs` turn in the thread added
    # another one (reported from the production UI, 2026-10-25). The comparative matrix
    # beside it is gated the same way and for the same reason (see `_draws_matrix`).
    labs = lab_trend_payload(tool_calls)
    # Resolved against the conversation's evidence, and against EVERY kind of it. A marker
    # addresses one array — the citable units the model was shown, numbered continuously
    # across the replayed turns and this one — and that array now holds laboratory
    # trajectories beside note passages. Resolving against the retrieval payload alone left
    # a lab citation pointing at a note passage, or at nothing.
    #
    # Named `record_sources` because these are the ones that ground the answer in THIS
    # patient's chart. What the turn publishes in `sources` is a wider list (see below);
    # what `verified` is decided from is this one, and the two must not be confused by a
    # later edit that adds a channel.
    record_sources = resolve_sources(renumbered, remap, rag, intent,
                                     evidence=evidence_calls)
    narrative = _narrative(renumbered, record_sources, literature)

    scope = turn_scope(evidence_calls)

    # A turn that retrieved records answers from them, so it is general medical evidence
    # even when a supplementary call also fetched this patient's own estimate (Phase 2,
    # 2026-10-16). The badge exists so a clinician never mistakes published literature for
    # something the chart supports; badging a comparator turn "Patient Chart Synthesis"
    # would claim the notes support a sentence drawn from JAMA. The one patient figure such
    # a turn states is attributed to the model in the table beside it, not to a passage.
    #
    # Without this, `/pubmed` read as a chart read that resolved no note citation — the
    # suppression rule immediately below — so every comparative turn dropped its scope pill
    # AND raised a visible safety swap over an answer nothing had been removed from. A
    # false correction is worse than none: it teaches a reader to skip the badge.
    if scope == SCOPE_PATIENT and _usable_articles(literature):
        scope = SCOPE_GENERAL

    # A chart read that resolved no note citation has nothing to substantiate the
    # claim. The scope says what the answer rests on, not that a chart tool ran, so
    # it is withheld here rather than asserted over prose it cannot evidence — the
    # same rule that gives a redirect no scope at all. Reported alongside it so the
    # BFF can carry the suppression into `guardrail_flags` instead of it ending
    # silently at this boundary.
    #
    # A lab card is the third channel, and it substantiates too (Phase 3, 2026-10-16). A
    # drill-down turn reads the patient's chart — `get_lab_trends` takes an admission, so
    # the scope is `patient_chart` and that is the right badge — but its evidence is a
    # trajectory rather than a passage, so `sources` is empty and the rule above would have
    # withheld the scope AND raised a visible safety swap over an answer nothing had been
    # removed from. The same false correction the comparator turn produced, from the same
    # cause: a channel added after the rule was written.
    # Read from the conversation's evidence, not from the drawing. These are two
    # different questions that used to share one variable: `labs` above answers "does
    # this turn draw a course", and this answers "was there a trajectory behind this
    # answer at all". A follow-up that quotes a replayed series is still a chart answer
    # — the scope pill says so — even though it no longer redraws the panel.
    lab_evidenced = bool((lab_trend_payload(evidence_calls) or {}).get("trends"))
    # A command's own structured result is a fourth channel (2026-10-18). `/risk` answers
    # from the model estimate and `/meds` from an extraction verified against the note,
    # so `sources` is empty on both by construction — not because a citation failed to
    # resolve — and the rule above withheld the scope and raised a visible safety swap
    # over an answer nothing had been removed from. Identical false positive to the
    # comparator and lab turns, from the identical cause: a channel added after the rule
    # was written.
    #
    # Deliberately limited to the commands whose payload IS their evidence. `/ehr` also
    # carries a payload (the estimate it compares against) but its prose is a note
    # synthesis, so an uncited `/ehr` answering is still the thing this flag exists to
    # report.
    payload_evidenced = bool(command_payload) and command == "meds"
    # A widget-only turn has no prose to evidence. Flagging "synthesis without
    # citations" against an empty sentence describes an omission that did not happen:
    # the canvas is the answer, and the reader was never shown a claim needing
    # attribution.
    no_prose = not (renumbered or "").strip()
    scope_suppressed = (scope == SCOPE_PATIENT and not record_sources
                        and not lab_evidenced and not payload_evidenced
                        and not no_prose)
    if scope_suppressed:
        scope = None

    # Whether this turn is DRAWN as a literature turn.
    #
    # Two ways it is: the search returned records, or it returned nothing and there is
    # no patient-record evidence to show instead. The second case is a real one —
    # `/pubmed` about a condition this patient does not have still owes the reader the
    # records and the sentence saying none matched.
    #
    # What it must never be is "a dict arrived". An empty result beside a chart answer
    # swapped the canvas to an empty `LiteratureList` and dropped the note citations, so
    # a correct chart answer arrived under a card saying no literature was found about a
    # question nobody asked of the literature (2026-10-22).
    articles = _usable_articles(literature or {})
    draws_literature = bool(articles) or (
        literature is not None and not record_sources and not lab_evidenced
        and not payload_evidenced
    )
    drawn_literature = literature if draws_literature else None

    # Whether the answer was verified against the patient record.
    #
    # The rule is grounding, not citation. A `/labs` answer rests on a trajectory, a
    # `/risk` answer on the model's own estimate, `/meds` on an extraction checked
    # against the note — none of them carries a `^[n]`, and requiring one would withhold
    # the badge from the turns that are most tightly bound to the chart. A pure PubMed
    # turn is the case it must NOT cover: those records are somebody else's patients.
    #
    # The guardrail's own verdict is ANDed by the caller (`http`), because only that
    # layer knows whether the answer survived it or was replaced by an abstention.
    # The estimate is a fifth channel, and it is the one `/risk` has: a widget-only turn
    # carries no prose and no passage, and its evidence is the model's own scoring of
    # this admission. A driver question reaches the same place from free text.
    estimate_evidenced = _usable_estimate(predict_payload(tool_calls)) is not None
    grounded_in_record = (
        bool(record_sources) or lab_evidenced or payload_evidenced or estimate_evidenced
        or bool(context_grounded)
    )
    # A literature turn is about other people's patients, whatever rides beside it — the
    # comparator turn draws this patient's estimate as a baseline row and is still an
    # answer about published cohorts. Note passages and laboratory trajectories are the
    # exception: a turn that cites the record IS grounded in it, whichever surface it drew.
    #
    # Decided from `record_sources` and NOT from the published `sources` list, which now
    # also carries the retrieved articles. Reading that list here would let a PubMed turn
    # vouch for itself: the records are somebody else's patients, and a literature-only
    # turn is not chart-grounded.
    literature_only = draws_literature and not record_sources
    # A widget IS an answer. `/risk` returns no prose by design, so "substantive" cannot
    # mean "has prose" — it means there is something the reader was given that can be
    # vouched for, and a drawn estimate is that.
    substantive = (not no_prose) or payload_evidenced or estimate_evidenced
    verified = bool(
        grounded_in_record and substantive and not scope_suppressed
        and not literature_only
    )

    canvas = compose_risk_canvas(
        predict_payload(tool_calls),
        rag,
        cite=first_citation(renumbered),
        sections=intent,
        source=record_sources[0] if record_sources else None,
        literature=drawn_literature,
        labs=labs,
        medications=command_payload,
        # The narrative chooses its citations from this turn's evidence: the
        # articles for a literature turn, the resolved note passages otherwise.
        narrative=narrative,
        # The patient's own condition selects the cohort rows, whatever the
        # question was about (see `turn_phenotype`), and `command` decides whether
        # this turn is one that carries the table at all (see `_draws_matrix`).
        # Resolved from the chart the conversation has read, so a follow-up and a
        # `/pubmed` turn that came after an `/ehr` still know whose condition the
        # cohorts are being compared against.
        phenotype=turn_phenotype(tool_calls, evidence_calls),
        eligible=admission_eligible(tool_calls),
        # The clinician's own words, when the turn had any: the other half of the
        # condition-lock, so a question about a condition this patient does not have
        # is answered with records and no comparison.
        query=str((literature or {}).get("query") or "") if isinstance(literature, dict) else "",
        command=command,
    )

    # The comparison table's published cohorts, registered as citable sources and
    # numbered after the note passages. Read from the composed canvas rather than from
    # `literature_benchmarks` directly, because whether the table was DRAWN is the
    # question — a condition that was retrieved and not compared owes no source — and
    # the canvas is the only place that decides it.
    #
    # Appended here, after every flag above, so nothing about this turn's grounding is
    # re-read from a list that now counts somebody else's patients.
    # The records the search returned, in the same numbered channel as every other
    # citable source, numbered from 1 and continuing into the comparison table's cohorts.
    #
    # A literature turn published no `sources` before Phase F (2026-10-26): the articles
    # lived only in the `literature` payload, which the canvas drew as boxed cards, so the
    # inline pill had nothing to address and the reference list did not exist on that
    # surface at all. They are registered here so a marker, a hover card and a drawer row
    # name one record — the same treatment `/ehr` and `/meds` citations already had.
    published_sources = literature_sources(articles)
    sources = record_sources + published_sources

    benchmark_sources, benchmark_references = register_benchmark_sources(
        canvas, start=len(sources) + 1,
        existing_pmids=[entry.get("pmid") for entry in sources],
    )
    # One array, one builder (Phase F.1, 2026-10-26). Every source this turn publishes is
    # converted to its reference row here, in the same order and under the same number, so
    # `sources` and `references` are two views of one list rather than two lists that
    # happen to agree. `/pubmed` shipped with three numbered pills and no rows at all
    # because this used to be assembled from the narrative's citations plus the table's
    # cohorts, and a literature turn's records were in neither of those two builders.
    published = sources + benchmark_sources
    references = reference_rows(published)

    return {
        "answer": renumbered,
        "citation_map": remap,
        "intent_sections": list(intent),
        # Note passages and laboratory trajectories first, then the published records the
        # search returned, then the cohorts the comparison table draws. Each group is
        # numbered to continue the one before it, so a marker written anywhere in this turn
        # addresses one array.
        "sources": published,
        # The published records a search RETURNED, passed through whole. Kept separate
        # from `sources` because it is the tool's own answer — `None` (no search) and an
        # empty list (searched, found nothing) must not collapse into each other — while
        # `sources` is the citation channel the records are addressed by.
        "literature": drawn_literature,
        # Whether this answer was checked against, and grounded in, the patient's own
        # record. Stated once, here, because the turns that qualify are decided by
        # clinical rules and a browser re-deriving them would be a second opinion.
        "verified": verified,
        # What the turn rested on.
        "scope": scope,
        "scope_suppressed": scope_suppressed,
        # The rows the console's references drawer lists — `1..K`, one per source, in
        # the same order and under the same numbers. Derived here rather than in the
        # browser because this is the layer that saw the tool calls and resolved the
        # citations; a second derivation downstream would be a second answer to the same
        # question, and the two would drift. See `note_presentation.reference_rows`.
        "references": references,
        "a2ui": canvas,
    }

# ===========================================================================
# Consolidated from `note_presentation.py` (Phase 7).
#
# The records an answer's citations resolve to.
#
# Everything between the model's `^[1]` markers and the rows a reader can click: the
# passage a marker points at, the section label on that row, the drawer record for a
# note citation and the drawer record for an article, and the narrative surface that
# carries them.
#
# Split out of `a2ui.py`, which is the composer. Nothing here builds an envelope or
# decides what a turn is drawn as — it answers "what is this citation, and what does a
# reader see for it", which is a question about the evidence rather than about the
# canvas. `a2ui` re-exports every name here, because the tests and `guardrail` reach
# them through it and the module boundary is not worth a second call path.
#
# `_usable_articles` lives here rather than with the canvas that also reads it: it is
# the rule for what may be cited at all (a record with no pmid cannot be linked), and
# the canvas is a consumer of that rule. The notes that describe an EMPTY or
# UNAVAILABLE search stay with the canvas, because they are wording for a surface
# that did not get any records.
# ===========================================================================

from services.agent.postprocess.citations import (
    citable_passages,
    cited_numbers,
    extract_section,
)

# The fields a literature record is put on the wire with, in the order the card shows
# them.
#
# The abstract was removed from this list on 2026-10-13 with the ruling that "the
# reader's next step is the record at PubMed, not a second copy of its text here" — at
# the time the only surface was a card that showed a title and a journal. Phase F
# (2026-10-26) puts the record behind a citation marker like every other source in the
# product, and a reference that opens onto a title and nothing else is not a reference:
# the reader cannot weigh a claim whose evidence is a heading. The abstract is the
# passage, so it travels.
def _unavailable_text(section: str) -> str:
    """The deterministic message for a question whose target section the note
    does not have. Never mine the content from unrelated narrative."""
    if section == "discharge_medications":
        return "No discharge medication information is available for this patient."
    return f"No {section.replace('_', ' ')} information is available for this patient."


def resolve_source(passages: list[dict], passage_n: int,
                   sections: tuple[str, ...] = ()) -> dict:
    """Resolve one citation to the passage and section body it supports.

    `passage_n` is the model's ORIGINAL 1-based passage number (before
    renumbering). `sections` is the question's section intent; when given, the
    section is located by label or extracted from any whole-note chunk, and
    the model's number is ignored — the model mis-numbers citations (a meds
    answer cites ^[1] while its supporting passage sits elsewhere).

    Returns {"section", "text"}. Callers own the badge number and the query.
    """
    cited = None
    intent_body = None
    matched_section = None
    for sec in sections:
        cited = next((p for p in passages if p.get("section") == sec), None)
        if cited is not None:
            matched_section = sec
            break
        # The index stores whole-note chunks: a passage labeled with a
        # different section still CONTAINS the target section, and the
        # labeled chunk can miss the top-k (near-tied whole-note
        # embeddings). Pull the section body out of the passage text.
        for p in passages:
            body = extract_section(p.get("text", ""), sec)
            if body:
                cited = p
                intent_body = body
                matched_section = sec
                break
        if cited is not None:
            break

    if cited is None and sections:
        # The note has NONE of the targeted sections. The deterministic answer
        # is "not available", not a passage mined from unrelated narrative.
        return {"section": "not available",
                "text": _unavailable_text(sections[0])}

    if cited is None:
        # Clamp to a real passage so an out-of-range number never raises.
        idx = min(max(passage_n, 1), len(passages)) - 1
        cited = passages[idx]
        shown_section = cited.get("section", "discharge note")
        section_text = extract_section(cited.get("text", ""), shown_section)
    elif intent_body is not None:
        shown_section = matched_section
        section_text = intent_body
    else:
        shown_section = matched_section or cited.get("section", "discharge note")
        section_text = extract_section(cited.get("text", ""), shown_section)

    return {"section": shown_section,
            "text": section_text or cited.get("text", ""),
            # What a citation for this passage calls itself, when the passage is not a
            # note section. A laboratory trajectory is evidence of a different kind — it
            # has an analyte rather than a heading — so it names itself here and
            # `note_citations` uses the name instead of a section label it would have to
            # guess at.
            "cite_source": str(cited.get("cite_source") or ""),
            "cite_title": str(cited.get("cite_title") or "")}


#: The badge a published record carries. One word, because a pill is a tag beside a
#: title rather than a sentence about it.
_LITERATURE_BADGE = "PubMed"


def _surface_components(canvas: dict | None):
    """Every component a composed surface carries, in the order it draws them.

    The components are not on the envelope: an A2UI surface is a list of protocol
    messages, and the component tree is the `updateComponents` payload of one of them.
    Reading `canvas["components"]` found nothing and would have gone on finding nothing,
    because the envelope is valid without that key.
    """
    for message in (canvas or {}).get("messages") or []:
        if not isinstance(message, dict):
            continue
        update = message.get("updateComponents")
        if not isinstance(update, dict):
            continue
        for component in update.get("components") or []:
            if isinstance(component, dict):
                yield component


def register_benchmark_sources(canvas: dict | None, start: int,
                              existing_pmids=()) -> tuple[list[dict], list[dict]]:
    """Give the comparison table's published cohorts a citation identity.

    `ComparativeRiskMatrix` draws curated PubMed cohorts beside this patient's estimate,
    and it is built OUTSIDE the prose citation pipeline: the rows come from
    `literature_benchmarks`, selected by the admission's phenotype, not from a passage
    the answer cites. So the table arrived with a journal, a year and a PMID on screen,
    and the turn arrived with no record of any of them — the references drawer listed
    `1..4` (the note sections) and the benchmark, which the whole comparison rests on,
    could not be opened from the turn that showed it.

    This registers each benchmark row as a source at `start` and onwards, and stamps the
    row itself with `source_index` / `source_label` so the surface can draw the pill
    beside it. Two lists are returned because the turn carries two: `sources` is the
    citation channel (numbered, `cite`), and `references` is the drawer's own rows. They
    are built together here so a number written in one cannot disagree with the other.

    Called AFTER every flag derived from `sources` has been computed. The cache of note
    passages is what `grounded_in_record` and `scope_suppressed` ask about — whether the
    answer rests on THIS patient's record — and a published cohort about somebody else's
    patients must not be able to answer yes to that question by being appended to the
    list it is asked of.
    """
    sources: list[dict] = []
    references: list[dict] = []
    seen = {str(pmid).strip() for pmid in existing_pmids if str(pmid or "").strip()}
    index = start
    for component in _surface_components(canvas):
        if component.get("component") != "ComparativeRiskMatrix":
            continue
        for row in component.get("rows") or []:
            if not isinstance(row, dict):
                continue
            pmid = str(row.get("pmid") or "").strip()
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            # What the table calls the row: the journal and the year. It stays the pill's
            # label, because a pill is set inside a sentence or a cell and a full article
            # title is not.
            citation = str(row.get("source") or "").strip() or f"PMID {pmid}"
            label = f"{index} · {citation}"
            url = str(row.get("url") or "").strip() or (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            # The article, which is what the record IS. A row that named only its journal
            # left the drawer showing `5. The Journal of Arthroplasty · 2022` and nothing
            # else — no paper, no authors, nothing to weigh it by.
            title = str(row.get("title") or "").strip() or citation
            authors = str(row.get("authors") or "").strip()
            journal = str(row.get("journal") or "").strip()
            year = str(row.get("year") or "").strip()
            # A citation line, in the order a reader reads one: where it was published,
            # when, by whom, and which record it is.
            journal_citation = " ".join(
                part for part in (
                    f"{journal}." if journal else "",
                    f"{year}." if year else "",
                    authors,
                ) if part
            ).strip()
            subtitle = " · ".join(
                part for part in (journal_citation, f"PMID {pmid}") if part
            )
            # The cohort and the sentence the rate was read from: what the popover and the
            # drawer show for the reader to weigh the comparison rather than take it.
            passage = " — ".join(
                part for part in (str(row.get("cohort_summary") or "").strip(),
                                  str(row.get("excerpt") or "").strip())
                if part
            )
            # Stamped on the row so the surface draws the pill from the same number,
            # rather than resolving a PMID back to an index a second time.
            row["source_index"] = index
            row["source_label"] = label
            sources.append({
                "cite": index,
                "kind": "Literature",
                "badge": _LITERATURE_BADGE,
                # The article, not its journal: this is the field the card's heading and
                # the drawer's row are titled from.
                "title": title,
                "authors": authors,
                "label": label,
                "subtitle": subtitle,
                "journal_citation": journal_citation,
                "section": citation,
                "text": passage,
                "tier": str(row.get("evidence_tier") or "").strip(),
                "pmid": pmid,
                "url": url,
            })
            references.append({
                "index": index,
                "source": _LITERATURE_BADGE,
                # The drawer's own row, shaped like the note rows beside it: a title, a
                # subtitle and the passage. `title` is the article — a row headed with its
                # journal was the naked `5. The Journal of Arthroplasty · 2022` a reader
                # reported, with nothing to weigh it by.
                "title": title,
                "label": label,
                "subtitle": subtitle,
                "authors": authors,
                "excerpt": passage,
                "pmid": pmid,
                "url": url,
            })
            index += 1
    return sources, references


def reference_rows(sources: list[dict]) -> list[dict]:
    """Every published source as the row the references drawer draws.

    One converter, applied to the whole `sources` array, so a turn cannot draw an inline
    pill `1..K` and then list fewer than `K` rows at the foot of the answer. That is not
    a hypothetical: `/pubmed` shipped with three numbered pills and an EMPTY references
    list, because the two came from different builders — `references` was assembled from
    the narrative's own citation list plus the comparison table's cohorts, and a
    literature turn's records were in `sources` and in the narrative's citations but the
    third builder had never been taught about the third kind of source (2026-10-26).

    Whatever a source is — a note passage, a laboratory trajectory, a retrieved article,
    a curated cohort — it has a title, a line to place it, and a passage. Those three are
    what a reference row is, so those three are what this reads, and a source that names
    itself (`cite_title`/`cite_source`) is used as it names itself.

    The `index` is the source's own `cite`, which is the number the marker shows. No
    second numbering exists to drift.
    """
    rows: list[dict] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        index = source.get("cite")
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        published = source.get("kind") == "Literature"
        if published:
            # A published record names itself: the article is the title, the citation
            # line is what places it, and the identifier is what a reader can open.
            title = str(source.get("title") or "").strip() or "Published record"
            subtitle = str(source.get("subtitle") or "").strip()
            self_named = _LITERATURE_BADGE
        else:
            title = str(source.get("cite_title") or "").strip() or _section_label(
                source.get("section"))
            subtitle = ""
            self_named = str(source.get("cite_source") or "").strip() or "Discharge note"
        row = {
            "index": index,
            "source": self_named,
            "title": title,
            "label": str(source.get("label") or "").strip() or f"{index} · {title}",
            "excerpt": str(source.get("text") or ""),
        }
        if subtitle:
            row["subtitle"] = subtitle
        for key in ("pmid", "url", "authors", "journal_citation", "year"):
            value = str(source.get(key) or "").strip()
            if value:
                row[key] = value
        rows.append(row)
    return rows


def resolve_sources(answer: str, citation_map: dict[str, int],
                    rag: dict | None, sections: tuple[str, ...],
                    evidence: list[dict] | None = None) -> list[dict]:
    """One resolved source per citation number in the (renumbered) answer.

    This is the list the browser renders on a footnote click: it looks up the
    clicked number and displays the entry — no vocabulary, no heuristics.
    Section intent is applied only to a single-citation answer; a
    multi-citation answer cites its passages in array order and maps by
    number.

    `evidence` is the whole conversation's tool calls, in the order the model was shown
    them, and it supersedes `rag` when given: a marker addresses ONE array, and that array
    holds every citable unit the conversation produced — four note sections from `/ehr`,
    then ten analytes from a lab drill-down — not only the passages of a single retrieval
    payload. Resolving against the retrieval alone left a lab citation pointing at a note
    passage or at nothing (Phase E, 2026-10-26).
    """
    if evidence is not None:
        passages = [
            passage for call in evidence
            for passage in citable_passages((call or {}).get("response"))
        ]
    else:
        passages = (rag or {}).get("passages") or []
    if not passages:
        return []
    query = (rag or {}).get("query") or "discharge note"
    cites = cited_numbers(answer)
    if not cites:
        return []
    intent = sections if len(cites) == 1 else ()
    return [
        {"cite": n,
         **resolve_source(passages, citation_map.get(str(n), n), intent),
         "query": query}
        for n in cites
    ]


# The section names as a clinician reads them. A separate vocabulary from
# `feature_labels`, which names MODEL features: these name parts of a note, and the
# two happen to share the word "discharge" without meaning the same thing by it.
_SECTION_LABELS = {
    "history_of_present_illness": "History of present illness",
    "brief_hospital_course": "Brief hospital course",
    "discharge_diagnosis": "Discharge diagnosis",
    "discharge_medications": "Discharge medications",
    "medications_on_transfer": "Medications on transfer",
    "discharge_instructions": "Discharge instructions",
    "discharge_condition": "Discharge condition",
    "discharge_disposition": "Discharge disposition",
    "followup_instructions": "Follow-up instructions",
    # Not a note heading: a laboratory trajectory, the other channel a chart answer can
    # rest on. It is here because this map is where a section key becomes words a reader
    # sees, and a lab citation that fell through to the generic branch would read
    # "laboratory trends" in lower case beside "Discharge diagnosis".
    "laboratory_trends": "Laboratory trends",
}


def _section_label(section: object) -> str:
    name = str(section or "")
    return _SECTION_LABELS.get(name, name.replace("_", " ") or "Note passage")


def narrative_surface(answer: str, citations: list[dict]) -> dict:
    """The prose, and the evidence its markers point at, as one component.

    The markers and the sentences they annotate travel together because a marker is a
    control the reader clicks while reading the sentence it sits in. The answer used to
    be drawn as plain text in a separate element with nothing to click, because the
    basic Text component renders Markdown and Markdown cannot host a button.

    `citations` are built by the caller from the evidence this turn actually rests on;
    see `note_citations` and `literature_citations`. Each entry's `index` is the
    number the marker shows, so the two cannot drift.

    There is no `streaming` key. There was one — a boolean the component read to draw
    its markers inert while the answer was still arriving — and no caller could ever set
    it: this envelope is composed once the turn has finished, so the value was `False`
    on every payload that ever reached a browser. The draft that fills the canvas while
    the answer is being written is drawn by the page (Phase 5B), not by this component,
    so the key was removed from the contract rather than left as a promise nothing kept.
    """
    return {"text": answer, "citations": citations}


def note_citations(sources: list[dict]) -> list[dict]:
    """One citation per resolved source, keyed by its renumbered marker.

    The citation layer renumbers `^[n]` to first-appearance order; reading the
    renumbered `cite` here keeps the marker and the popover in step without a second
    mapping to drift.

    A source that names itself wins over the section vocabulary. A note passage does not
    need to — every note passage is a discharge note and differs only by section — but a
    laboratory trajectory is not a note passage at all, and labelling it "Discharge note"
    would tell the reader their creatinine came out of the prose.
    """
    citations = []
    for source in sources:
        title = str(source.get("cite_title") or "") or _section_label(
            source.get("section"))
        citations.append({
            "index": int(source["cite"]),
            # The channel the evidence came from. Note passages all share one channel
            # and differ by section, so the section becomes the title.
            "source": str(source.get("cite_source") or "Discharge note"),
            "title": title,
            # What the marker SAYS, resolved here rather than in the browser. The marker
            # reads `1 · Hospital Course` and not a bare `1`, because a number tells the
            # reader nothing about what they are about to open — and resolving it here
            # keeps the section vocabulary in the one module that already owns it
            # (`_SECTION_LABELS`), rather than a second copy in the client that would
            # drift the first time a section is renamed.
            "label": f"{int(source['cite'])} · {title}",
            "excerpt": str(source.get("text") or ""),
        })
    return citations


#: How long a literature marker's source token may get before it is cut back.
#:
#: A journal name is unbounded — "Journal of the American College of Cardiology" is
#: forty-seven characters — and an inline marker is a pill set inside a sentence, so an
#: unbounded label is a pill whose own box wraps to a second line and reads as a broken
#: one. Asked for by a reader looking at exactly that, 2026-10-16.
#:
#: Twenty-four characters keeps "Home healthcare now" and "Patient education and…"
#: recognisable. The full title is a hover away on the marker's popover and a line below in
#: the record list, which is where a reader goes to weigh a source anyway.
_CITE_SOURCE_CHARS = 24


def _short_source(name: str) -> str:
    """A source name cut back to the marker's budget, at a word boundary.

    At a boundary rather than mid-word, and with an ellipsis rather than silently: a cut
    that lands inside a word reads as a typo, and one with nothing to mark it reads as the
    whole name.
    """
    text = " ".join(str(name or "").split())
    if len(text) <= _CITE_SOURCE_CHARS:
        return text
    cut = text[:_CITE_SOURCE_CHARS]
    space = cut.rfind(" ")
    if space > 8:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def _year_of(pub_date: object) -> str:
    """The four-digit year a PubMed date states, or an empty string.

    PubMed dates are not one format — "2023", "2023 Aug 01", "2022 Sep-Oct" — so this
    reads the year off rather than parsing a date. Nothing here needs a day.
    """
    for token in str(pub_date or "").replace("-", " ").split():
        if len(token) == 4 and token.isdigit():
            return token
    return ""


#: How much of an abstract a hover card carries.
#:
#: The card is 340px wide and floats over the answer being read, so it is a place to judge
#: a source, not a place to read one. 400 characters is the opening of a structured
#: abstract — enough to see what the study did and found — and the record itself is one
#: click away on the title above it. Cut at a sentence end where there is one inside the
#: budget, because a cut mid-clause reads as a truncation error rather than as a summary.
_CITE_EXCERPT_CHARS = 400


def _excerpt(text: object) -> str:
    """The opening of an abstract, bounded and marked as an opening."""
    body = " ".join(str(text or "").split())
    if len(body) <= _CITE_EXCERPT_CHARS:
        return body
    cut = body[:_CITE_EXCERPT_CHARS]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if stop > _CITE_EXCERPT_CHARS // 2:
        return cut[: stop + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip(" ,.;:") + "…"


def _article_record(article: dict, index: int) -> dict:
    """One published record, in the shape every other citable source already has.

    Built once and read by both channels — the narrative's citations and the turn's
    `sources` — so a marker, a hover card and a drawer row cannot disagree about which
    paper they name. Phase F (2026-10-26): a literature turn used to be the one surface
    with its own presentation, a boxed card under the prose and a marker that opened
    PubMed directly; what it never had was the hover card and the reference row that
    every other citation in the product gets.
    """
    journal = str(article.get("journal") or "").strip()
    year = _year_of(article.get("pub_date"))
    authors = str(article.get("authors") or "").strip()
    pmid = str(article.get("pmid") or "").strip()
    title = str(article.get("title") or "Untitled record").strip()
    # The citation line, in the order a reader reads one: where it was published, when,
    # by whom, and which record it is. Every part is optional except the identifier,
    # because a record that states no author must not render as "et al." against nobody.
    journal_citation = " ".join(part for part in (
        f"{journal}." if journal else "",
        f"{year}." if year else "",
        authors,
    ) if part).strip()
    subtitle = " · ".join(part for part in (journal_citation, f"PMID {pmid}")
                           if part)
    # What the pill SAYS. The journal and the year, because a pill is set inside a
    # sentence and a title is not; cut to a marker's length by `_short_source`, which is
    # the same budget the comparative table's pill uses.
    label = f"{index} · " + " · ".join(part for part in (
        _short_source(journal) or "PubMed", year) if part)
    return {
        "index": index,
        "kind": "Literature",
        "badge": _LITERATURE_BADGE,
        "title": title,
        "journal": journal,
        "year": year,
        "authors": authors,
        "label": label,
        "subtitle": subtitle,
        "journal_citation": journal_citation,
        # The passage: what the record reports. A reference that opens onto a title and
        # nothing else cannot be weighed.
        "text": _excerpt(article.get("abstract")),
        "pmid": pmid,
        "url": str(article.get("url") or "").strip() or (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""),
    }


def literature_citations(articles: list[dict]) -> list[dict]:
    """One citation per retrieved article, shaped like every other citation.

    `index` is the 1-based position in the record order the model was given, which is
    exactly what the formatting instruction tells it to cite — the marker in the prose and
    the record beneath it are the same record by construction.

    The marker's DISPLAY is the number, the journal and the year; the identifier stays the
    link target so a reader who wants the paper is one click from the card's title. That
    trade was made on 2026-10-15 (journal over identifier), revisited on 2026-10-16 (the
    pills were reading long, so the journal is cut to `_CITE_SOURCE_CHARS`), and completed
    here: the record now also carries the article, its authors and its abstract, because
    the card behind the pill is the same card a note citation opens.

    `source` is the EYEBROW the card shows under the title. For a note passage that is
    "Discharge note" and the section is the title; for a published record the title is the
    article, so the eyebrow is the citation line itself.
    """
    citations = []
    for position, article in enumerate(articles, start=1):
        record = _article_record(article, position)
        citation = {
            "index": record["index"],
            "source": record["subtitle"],
            "title": record["title"],
            "label": record["label"],
            "subtitle": record["subtitle"],
            "authors": record["authors"],
            "journal_citation": record["journal_citation"],
            "excerpt": record["text"],
        }
        if record["pmid"]:
            citation["pmid"] = record["pmid"]
        if record["url"]:
            citation["url"] = record["url"]
        citations.append(citation)
    return citations


def literature_sources(articles: list[dict], start: int = 1) -> list[dict]:
    """The retrieved records as `sources` entries, numbered from `start`.

    A literature turn had no `sources` at all: the records lived in the `literature`
    payload, which the canvas drew as cards, and the citation channel held only note
    passages. The consequence was a turn whose markers could not address anything — the
    browser resolved `^[PMID: n]` by identifier as a special case, and a `^[1]` written on
    a literature turn pointed at nothing.

    They are `kind="Literature"` for the same reason the comparison table's cohorts are:
    a published record was not retrieved from THIS patient's note, so it carries the
    identifier and the link a reader can open instead of a retrieval query.
    """
    sources = []
    for offset, article in enumerate(articles):
        record = _article_record(article, start + offset)
        sources.append({
            "cite": record["index"],
            "kind": record["kind"],
            "badge": record["badge"],
            "title": record["title"],
            "authors": record["authors"],
            "journal": record["journal"],
            "year": record["year"],
            "label": record["label"],
            "subtitle": record["subtitle"],
            "journal_citation": record["journal_citation"],
            # The section a note passage would name. For a published record it is the
            # citation tram, which is what the table's benchmark rows use too.
            "section": " · ".join(part for part in (record["journal"], record["year"])
                                  if part) or record["badge"],
            "text": record["text"],
            "tier": "",
            "pmid": record["pmid"],
            "url": record["url"],
        })
    return sources


def _narrative(answer: str, sources: list[dict], literature: dict | None) -> dict | None:
    """The narrative surface for this answer, or None when there is no prose.

    Citations come from whichever evidence the turn rests on. A literature turn cites its
    articles and nothing else: those records ARE the evidence, and resolving the markers
    against note passages would leave every marker in a summary unbacked — which the
    component draws as a disabled control, on a summary that was properly grounded.

    The turn is only a literature turn when it produced RECORDS. An empty search result
    is a `dict`, so a bare `is not None` test made every chart answer that happened to
    carry a fruitless PubMed call cite zero articles: the note passages had already been
    resolved, and this function threw them away in favour of an empty list. The reader
    then had a `^[1]` that resolved to nothing beside a card saying no literature was
    found (2026-10-22). Records or note sources — never an empty list over a full one.
    """
    if not answer.strip():
        return None
    articles = _usable_articles(literature or {})
    if articles and literature is not None:
        return narrative_surface(answer, literature_citations(articles))
    return narrative_surface(answer, note_citations(sources))
