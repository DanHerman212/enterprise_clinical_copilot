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
from services.agent.admission_phenotypes import (
    GENERAL_INPATIENT,
    NON_ADMISSION,
    is_matrix_eligible,
    resolve_phenotype,
)

# The citation records live in `note_presentation`. Imported by name here because
# this is where the rest of the package — and the tests — reach for them:
# `guardrail` reads `_usable_articles`, and the canvas and the composer call
# `resolve_sources`, `_narrative` and `_usable_articles` directly.
# The canvas lives in `risk_canvas`; the composer still calls into it, and the
# tests reach `_draws_matrix` here.
from services.agent.risk_canvas import (  # noqa: F401 — re-exported
    _draws_matrix,
    _usable_estimate,
    compose_risk_canvas,
)
from services.agent.note_presentation import (  # noqa: F401 — re-exported
    _narrative,
    _section_label,
    _short_source,
    _unavailable_text,
    _usable_articles,
    literature_citations,
    literature_sources,
    narrative_surface,
    note_citations,
    reference_rows,
    register_benchmark_sources,
    resolve_source,
    resolve_sources,
)

from services.agent.a2ui_envelope import (  # noqa: F401 — re-exported
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
    # vouch for itself: the records are somebody else's patients, and `dev_t1_06` asserts
    # exactly this turn is not chart-grounded.
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