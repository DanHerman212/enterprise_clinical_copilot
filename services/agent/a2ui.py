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

from services.agent.citations import (
    citation_remap,
    first_citation,
    intent_sections,
    renumber_citations,
)
from services.agent.admission_phenotypes import (
    bucket_for,
    is_matrix_eligible,
    NON_ADMISSION,
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
    narrative_surface,
    note_citations,
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


def turn_phenotype(tool_calls: list[dict]) -> str:
    """The phenotypic bucket this turn's admission resolved to, or "".

    Read off the `hadm_id` the tools were actually given, the same way `turn_scope` reads
    the scope, so a tool added later is classified by what it did. The admission id
    travels as an argument rather than being passed alongside the calls because this
    layer's whole job is to derive the surface from the evidence the turn produced.

    Three ways to get nothing back, and the caller cannot tell them apart — which is
    correct, because the consequence is the same and it is the safe one:

      * no tool carried an admission (a condition-only search, a free-text question);
      * the admission is not in the reviewed map, so its condition is unknown;
      * the admission resolved to a bucket whose policy is no comparison, or is not a
        hospital admission at all.

    The condition-lock is enforced here rather than at the matrix, so a mismatched
    comparison cannot be drawn by any caller that forgets a check: a `/pubmed` turn whose
    query is about a condition this patient does not have still carries `hadm_id` for the
    estimate, and the matrix for it is selected by the PATIENT's phenotype. A query about
    heart failure on an orthopedic admission therefore draws the orthopedic rows beside
    her estimate or, when the query is the point, the records alone — never heart failure
    cohorts pinned under an orthopedic patient.
    """
    for call in tool_calls or []:
        hadm_id = (call.get("args") or {}).get("hadm_id")
        if not isinstance(hadm_id, int) or isinstance(hadm_id, bool):
            continue
        # The first admission the turn used is the one it is about: every call in a turn
        # carries the same one, so there is nothing here to reconcile between calls.
        if not is_matrix_eligible(hadm_id):
            return ""
        bucket = bucket_for(hadm_id)
        return bucket.key if bucket else ""
    return ""


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
    literature = literature_payload(evidence_calls)
    # The lab card reads THIS turn's calls, not the conversation's. `evidence_calls` is
    # the right array for what the answer rests on and for how its markers are numbered
    # (TF-10, 2026-10-23), but a chart is not a provenance statement — it is a drawing
    # of a course this turn read. Composed from the replay, a follow-up about anything
    # at all — the discharge diagnosis, a medication — redrew the sparkline panel of a
    # `/labs` turn from three turns ago, and each extra `/labs` turn in the thread added
    # another one (reported from the production UI, 2026-10-25). The comparative matrix
    # beside it is gated the same way and for the same reason (see `_draws_matrix`).
    labs = lab_trend_payload(tool_calls)
    sources = resolve_sources(renumbered, remap, rag, intent)
    narrative = _narrative(renumbered, sources, literature)

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
    scope_suppressed = (scope == SCOPE_PATIENT and not sources
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
        literature is not None and not sources and not lab_evidenced
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
        bool(sources) or lab_evidenced or payload_evidenced or estimate_evidenced
        or bool(context_grounded)
    )
    # A literature turn is about other people's patients, whatever rides beside it — the
    # comparator turn draws this patient's estimate as a baseline row and is still an
    # answer about published cohorts. Note sources are the exception: a turn that cites
    # the record is grounded in it, whichever surface it drew.
    literature_only = draws_literature and not sources
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
        source=sources[0] if sources else None,
        literature=drawn_literature,
        labs=labs,
        medications=command_payload,
        # The narrative chooses its citations from this turn's evidence: the
        # articles for a literature turn, the resolved note passages otherwise.
        narrative=narrative,
        # The patient's own condition selects the cohort rows, whatever the
        # question was about (see `turn_phenotype`), and `command` decides whether
        # this turn is one that carries the table at all (see `_draws_matrix`).
        phenotype=turn_phenotype(tool_calls),
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
    benchmark_sources, benchmark_references = register_benchmark_sources(
        canvas, start=len(sources) + 1,
        existing_pmids=[entry.get("pmid") for entry in sources],
    )

    return {
        "answer": renumbered,
        "citation_map": remap,
        "intent_sections": list(intent),
        # The note passages first, then the cohorts the comparison table draws. The
        # second group is numbered to continue the first, so a marker written anywhere
        # in this turn addresses one array.
        "sources": sources + benchmark_sources,
        # The published records a search RETURNED, passed through whole. Kept separate
        # from `sources` on purpose: that list is the citation channel — numbered, and
        # addressable by a marker — and a returned article is drawn by the canvas rather
        # than cited. The comparison table's curated cohorts are the exception and are
        # in `sources`, because the table names them on screen and a reader can open one.
        "literature": drawn_literature,
        # Whether this answer was checked against, and grounded in, the patient's own
        # record. Stated once, here, because the turns that qualify are decided by
        # clinical rules and a browser re-deriving them would be a second opinion.
        "verified": verified,
        # What the turn rested on, and the rows the console's references drawer lists.
        # Derived here rather than in the browser because this is the layer that saw the
        # tool calls and resolved the citations; a second derivation downstream would be
        # a second answer to the same question, and the two would drift.
        #
        # `references` is the narrative's own citation list, so the marker a reader
        # clicks and the row they find at the foot of the turn are the same record by
        # construction rather than by two lists agreeing today.
        "scope": scope,
        "scope_suppressed": scope_suppressed,
        # The narrative's own citations, plus the cohorts the comparison table drew. A
        # note passage is evidence the answer cites; a benchmark is evidence the SURFACE
        # shows, and both are things a reader can open from the foot of the turn.
        "references": (
            list((narrative or {}).get("citations") or []) + benchmark_references
        ),
        "a2ui": canvas,
    }