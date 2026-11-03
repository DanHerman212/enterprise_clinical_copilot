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

3. **The model's citation numbers are the numbering.** This layer resolves a
   marker against the array the model was shown and draws the evidence it points
   at; it does not rewrite the prose. The SourceCard resolves its passage by
   section intent, not by the model's numbering — see
   `services/agent/postprocess/citations.py`.

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
    RETRIEVAL_TOOLS,
    EvidenceLedger,
    cited_numbers,
    cited_pmids,
    citable_passages,
    first_citation,
    intent_sections,
    literature_articles,
    literature_calls,
)
# The two numbering helpers stay importable from here, and are NOT used by this module:
# callers that keep an answer in the expanded form — a stored transcript, a replayed
# `resolve_sources` call in the tests — need to ask the same question of the same module
# the answer was settled in. This layer no longer calls them (Charter 13 Law 1; audit
# L1-1): a marker is resolved and drawn here, never rewritten.
from services.agent.postprocess.citations import (  # noqa: F401 — re-exported
    citation_remap,
    renumber_citations,
)
from services.agent.presentation.domain import _usable_articles, is_matrix_eligible
# The analyte vocabulary, from the shared leaf rather than from `routing/`: this layer
# may not import the routing package (Charter 13 §5), and the answer it needs — "did the
# question name a measurement?" — is a label question, not a routing decision.
from shared.labels import names_an_analyte

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


def admission_eligible(tool_calls: list[dict], *,
                       is_outpatient: bool | None = None) -> bool:
    """Whether the record this turn is about is an admission at all.

    The encounter, not a table: the orchestrator reads whether this record is a stay and
    passes it in. Until Stage Gate 2 this asked whether the admission id appeared in a
    hand-maintained frozenset of non-admissions — a corpus fact compiled into the
    presentation layer, which is exactly the kind of map the gate removed.

    Defaults to True when the caller has no encounter: a literature or free-text turn that
    names no admission is not making an inpatient claim, so there is nothing to suppress.
    """
    return is_matrix_eligible(is_outpatient=is_outpatient)


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

    Which tools are retrieval tools is `routing.PASSAGE_TOOLS`'s fact, imported rather
    than restated: this module held the only other copy of that grouping outside
    `routing/` (Charter 13 Law 4).
    """
    rag_calls = [call for call in tool_calls or []
                 if call.get("name") in RETRIEVAL_TOOLS
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


def literature_payload(
    tool_calls: list[dict],
    replay_calls: "list[dict] | None" = None,
) -> dict | None:
    """The records this turn's literature searches returned, as ONE payload.

    EVERY search the turn made, unioned by pmid in call order. It used to be the first
    match alone, on the reasoning that "a run normally makes one search, and if the
    model made two the answer was written against the first". On 2026-10-30 a turn made
    four — the model refining its query — and wrote its answer against the last, so the
    composer drew no articles, `sources` carried no published records, and a correct
    citation was removed at settlement for naming a record nobody had retrieved.

    `citations.literature_calls` and `citations.literature_articles` make the choice and
    the union, so the payload here, the PMIDs enforcement accepts and the count the
    numbered channel uses are one set of records.

    `replay_calls` is the replayed turns' calls, passed only by a turn whose own
    question is about published records — the case a declared follow-up is, since
    `graph._freetext_allowed` withholds `search_literature` from it because the earlier
    search is already in the transcript. A chart turn must not reach the replay: that is
    90000058. Defaults to None, which is the behaviour every turn had before this
    parameter.

    No records means the first payload is returned as it stands, because its `error` key
    is what tells the composer and the caption that the tool did not answer — a
    different statement from "nothing was found", and dropping it here would collapse
    the two into the same empty canvas.
    """
    calls = literature_calls(tool_calls, replay_calls)
    if not calls:
        return None
    payloads = [(call.get("response") or {}) for call in calls]
    articles = literature_articles(calls)
    if not articles:
        return payloads[0] or None

    merged = {key: value for key, value in payloads[0].items() if key != "error"}
    merged["articles"] = articles
    merged["returned"] = len(articles)
    # The LAST query, because that is the one the answer was written against and the
    # caption states what was searched. An earlier refinement's wording would describe
    # a search whose records may not be on the screen at all. Read from the payload's own
    # `query` where the tool reported one — that is what was actually searched, and the
    # arguments are the request rather than the result.
    for call in reversed(calls):
        response = call.get("response") or {}
        query = str(response.get("query")
                    or (call.get("args") or {}).get("query") or "").strip()
        if query:
            merged["query"] = query
            break
    return merged


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


def compose_presentation(question: str, answer: str,
                         tool_calls: list[dict],
                         command_payload: dict | None = None,
                         command: str | None = None,
                         context_grounded: bool = False,
                         evidence: list[dict] | None = None,
                         *, is_outpatient: bool | None = None,
                         literature_replay: bool = False,
                         citation_map: dict[str, int] | None = None,
                         question_names_analyte: bool = False,
                         comparators: list[dict] | None = None) -> dict:
    """The full presentation contract for an answered question.

    One call produces every field the browser consumes, so the answer's
    evidence semantics are decided in exactly one place (here) and the BFF is
    a pass-through:

      - answer: the settled answer, VERBATIM — this layer resolves its citations
        and draws them, and does not rewrite the prose the trace recorded
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

    `literature_replay` is the caller saying this turn is about published records AND ran
    no search of its own, which is what a declared follow-up is: `graph._freetext_allowed`
    withholds `search_literature` because the earlier search is already in the transcript.
    The records it cites are then the replayed ones, so they are the ones drawn — in
    `sources`, in `references`, and in the scope badge. Defaults to False, which is the
    rule that a search belongs to the turn that made it (90000058).

    Pure — no I/O, no clock, no randomness — so it is fully testable without
    a browser or a live endpoint.
    """
    # The answer is published unchanged. Rewriting it here — expanding `^[4, 5]` into
    # `^[4] ^[5]`, or reassigning numbers — meant the prose `turn.py` recorded on the
    # trace and the prose the browser rendered were produced by two layers, and they
    # disagreed about the text of one turn (Charter 13 Law 5; audit L1-1/L5). A marker is
    # this layer's to RESOLVE and DRAW, not to rewrite.
    intent = intent_sections(question)
    # Which published records this answer names. Read here, beside the answer, because the
    # reference list is built much further down and the answer is the only thing that says
    # which of the retrieved records were used.
    cited_numbers_of_records = cited_pmids(answer)
    # `evidence` is every call the conversation has shown the model — the replayed turns
    # first, then this one — because that is the array the answer's markers are numbered
    # against (`graph._render_tool_result` counts continuously across both). Resolving
    # against this turn's calls alone left a follow-up that quoted the chart with a
    # marker that pointed at nothing (TF-10, 2026-10-23).
    evidence_calls = list(evidence) if evidence is not None else tool_calls
    # One array, one count, one place the published records start (Phase 4; audit §2.3).
    # Both were derived here before — `sum(len(citable_passages(...)))` — which made the
    # presentation layer a second slot counter beside the guardrail's.
    ledger = EvidenceLedger.of(evidence_calls)
    literature_start = ledger.literature_start
    rag = rag_payload(evidence_calls)
    # The literature surface is the search THIS turn answers from: its own, or — for a
    # turn whose question is about published records — the newest search in the replay.
    #
    # Read from the conversation unconditionally, a `search_literature` call made three
    # turns ago came back on every turn after it: `resolve_sources` bound `^[1]` to a
    # PubMed record the reader was not being shown, the badge flipped to General Medical
    # Evidence over an answer about the discharge note, and `LiteratureList` re-rendered
    # under medication instructions (90000058, 2026-10-26). A search is an event, not a
    # state.
    #
    # Read from this turn alone, the opposite failure: a declared follow-up holds no
    # search — `graph._freetext_allowed` withholds the tool because the earlier result is
    # already in the transcript — so the records it cites reached neither `sources` nor
    # `references`, and a `/pubmed` answer followed by its own chip lost both its pills
    # and its drawer (live, 2026-10-30). `literature_replay` is the caller saying the two
    # facts together: this turn is about published records, and it ran no search of its
    # own. It is the same `wants_literature` gate that let the replay keep the search, so
    # what the model was shown and what the reader is shown cannot drift apart.
    literature = literature_payload(
        tool_calls, evidence_calls if literature_replay else None)
    # The lab card reads THIS turn's calls, not the conversation's. `evidence_calls` is
    # the right array for what the answer rests on and for how its markers are numbered
    # (TF-10, 2026-10-23), but a chart is not a provenance statement — it is a drawing
    # of a course this turn read. Composed from the replay, a follow-up about anything
    # at all — the discharge diagnosis, a medication — redrew the sparkline panel of a
    # `/labs` turn from three turns ago, and each extra `/labs` turn in the thread added
    # another one (reported from the production UI, 2026-10-25). The comparative matrix
    # beside it is gated the same way and for the same reason (see `_draws_matrix`).
    #
    # And it is drawn only when the turn ASKED for it: `/labs`, or a question that names
    # an analyte. A free-text turn that read the panel to answer something else — "which
    # issue should be treated first?" — kept the panel out of an answer that never
    # discussed it, one column wide and out of context (2026-10-31, turn 3 of the
    # 90000023 thread). The evidence is not lost: the series is still in `evidence_calls`,
    # so a marker into it resolves and the scope still reads the chart.
    labs = (lab_trend_payload(tool_calls)
            if command == "labs" or names_an_analyte(question) else None)
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
    record_sources = resolve_sources(answer, citation_map or {}, rag, intent,
                                     evidence=evidence_calls)
    # What the reader is SHOWN as evidence, which for `/labs` is nothing at all
    # (2026-11-03). A laboratory turn already draws every reading it discusses — the
    # measurement, the reference interval, the course and the flag, one row per timepoint —
    # in the table under the sentence. A note citation into a trajectory only re-labels that
    # table as a set of "Clinical Note" records, so the drawer beneath a lab answer listed
    # one synthetic card per analyte and the prose carried a pill whose popover repeated the
    # row it sat beside.
    #
    # `record_sources` is still resolved and still decides what this turn rests on: the
    # scope badge, the grounding verdict and the literature gate are all read from it below.
    # Only the published channel is emptied, so the answer stays a patient-chart answer with
    # no citations rather than becoming an uncited one.
    shown_sources = [] if command == "labs" else record_sources
    narrative = _narrative(answer, shown_sources, literature,
                           literature_start=literature_start)

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
    no_prose = not (answer or "").strip()
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
        cite=first_citation(answer),
        sections=intent,
        source=shown_sources[0] if shown_sources else None,
        literature=drawn_literature,
        labs=labs,
        medications=command_payload,
        # The narrative chooses its citations from this turn's evidence: the
        # articles for a literature turn, the resolved note passages otherwise.
        narrative=narrative,
        # The verified comparator rows, settled and traced by the engine during the
        # `/ehr` turn. This layer draws them and never chooses them: a row here was read
        # from an abstract that this turn retrieved, and the reading was checked against
        # that abstract before it arrived.
        comparators=comparators,
        eligible=admission_eligible(tool_calls, is_outpatient=is_outpatient),
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
    #
    # Only the records this answer NAMES (`citations.cited_pmids`). A search returns three
    # and the answer rests on one; publishing all three told the reader two papers had been
    # used (patient 90000069, turn 3, 2026-10-31 — two rows nothing pointed at).
    published_sources = literature_sources(
        articles, start=literature_start, cited=cited_numbers_of_records)
    sources = shown_sources + published_sources

    comparator_sources, comparator_references = register_comparator_sources(
        canvas, start=len(sources) + 1,
        existing_pmids=[entry.get("pmid") for entry in sources],
    )
    # And the prose is told they exist. The narrative was composed before the table was,
    # so its citation list cannot have anticipated these rows; a badge the briefing writes
    # for a cohort resolves against this array, and without them it had nothing to draw.
    add_comparator_citations(canvas, comparator_sources)
    # One array, one builder (Phase F.1, 2026-10-26). Every source this turn publishes is
    # converted to its reference row here, in the same order and under the same number, so
    # `sources` and `references` are two views of one list rather than two lists that
    # happen to agree. `/pubmed` shipped with three numbered pills and no rows at all
    # because this used to be assembled from the narrative's citations plus the table's
    # cohorts, and a literature turn's records were in neither of those two builders.
    published = sources + comparator_sources
    references = reference_rows(published)

    return {
        "answer": answer,
        # The marker numbers are not reassigned here, so this is the identity over the
        # numbers the prose actually cites — unless settlement compacted them, in which
        # case it published the map and this is that map. Kept because the client
        # translates a clicked pill through it (Charter 13 Law 5).
        "citation_map": (dict(citation_map) if citation_map
                         else {str(n): n for n in cited_numbers(answer)}),
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
    renumbering). `sections` is the question's section intent: when the passages contain
    one of those sections — as a labelled passage, or embedded in a whole-note chunk — that
    body is what the reader is shown, because the model mis-numbers citations and the intent
    knows what the question asked for.

    The intent is a PREFERENCE and never a veto. A citation that indexes a passage the tool
    actually returned resolves to that passage and its own section, whatever the intent says.
    It used to be a veto: with intent given and no matching section in the passages, a valid
    `^[1]` resolved to "not available" — so a `/meds` answer whose only passage was the
    hospital course told the reader no medication information existed while citing a passage
    that named three drugs (90000069, 2026-11-03). "Not available" is now reserved for the
    two cases where there is nothing to show: no passages at all, or a number outside them.

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

    # Nothing to show, and nothing to prefer: an empty retrieval, or a number that indexes
    # no passage. These are the only two "not available" cases.
    if not passages:
        return {"section": "not available",
                "text": _unavailable_text(sections[0] if sections else "")}
    if cited is None and not (1 <= passage_n <= len(passages)):
        return {"section": "not available",
                "text": _unavailable_text(sections[0] if sections else "")}

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


def add_comparator_citations(canvas: dict | None, sources: list[dict]) -> None:
    """Give the prose the cohorts it is now allowed to cite.

    The prose's citation list is resolved before the comparison table is composed, because
    the table needs the narrative to put it under. The table's cohorts are registered after
    that, so a `^[PMID: …]` badge the briefing writes in `Readmission risk & trajectory`
    had no record to resolve against: the number was accepted by settlement (the allowance
    knows the rows), the browser could not draw it, and the reader saw a cohort named in
    prose with no control beside it (reported 2026-11-02).

    So the citations arrive with the sources rather than being anticipated by them: a
    source registered as citable belongs in the array the prose resolves against, under the
    number it was given. One numbering — `cite` — read by the list, the popover and the
    marker.

    Silent when the turn drew no prose: a surface-only answer has no narrative component to
    extend, and nothing is owed.
    """
    if not sources:
        return
    for component in _surface_components(canvas):
        if component.get("component") != "CitedNarrative":
            continue
        citations = component.setdefault("citations", [])
        known = {int(entry.get("index") or 0) for entry in citations}
        for source in sources:
            index = int(source.get("cite") or 0)
            if not index or index in known:
                continue
            citations.append({
                "index": index,
                "source": str(source.get("subtitle") or ""),
                "title": str(source.get("title") or ""),
                "label": str(source.get("label") or ""),
                "pill_label": str(source.get("pill_label") or ""),
                "kind": "literature",
                "subtitle": str(source.get("subtitle") or ""),
                "authors": str(source.get("authors") or ""),
                "journal_citation": str(source.get("journal_citation") or ""),
                "excerpt": str(source.get("text") or ""),
                **({"pmid": str(source["pmid"])} if source.get("pmid") else {}),
                **({"url": str(source["url"])} if source.get("url") else {}),
            })
            known.add(index)
        return


def register_comparator_sources(canvas: dict | None, start: int,
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
            # What the table calls the row: the journal and the year. It stays the ROW's
            # label, because a row is a record and a record is dated.
            citation = str(row.get("source") or "").strip() or f"PMID {pmid}"
            label = f"{index} · {citation}"
            url = str(row.get("url") or "").strip() or (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            # What the pill in the table's source cell prints: the journal alone, the same
            # compact emerald badge the prose draws, from the same resolver. A cell that
            # carried `4 · Journal · Year` was a bibliography line in a table column and
            # stretched the column rather than marking the record (2026-11-02).
            pill_label = _pill_journal(row.get("journal"),
                                       row.get("journal_abbrev"))
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
                # The inline form: the journal alone, exactly as the prose pill prints it.
                "pill_label": pill_label,
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
#
# One entry per canonical section in `services/mcp/retrieval/sections.py` (2026-11-03).
# The registry grew to cover the whole cohort, and every name it can produce has to be
# readable here: the fallback below turns an unmapped key into its own underscores, so an
# unmapped section is not a crash but a citation labelled "admission diagnosis" in lower
# case beside ones that are capitalised, which is the sort of detail a reader sees and a
# test does not. `tests/agent/test_section_labels.py` holds the two maps to the registry.
_SECTION_LABELS = {
    "name": "Patient name",
    "unit_no": "Unit number",
    "admission_date": "Admission date",
    "discharge_date": "Discharge date",
    "date_of_birth": "Date of birth",
    "sex": "Sex",
    "service": "Service",
    "allergies": "Allergies",
    "attending": "Attending physician",
    "chief_complaint": "Chief complaint",
    "major_procedure": "Major surgical or invasive procedure",
    "history_of_present_illness": "History of present illness",
    "review_of_systems": "Review of systems",
    "past_medical_history": "Past medical history",
    "past_surgical_history": "Past surgical history",
    "social_history": "Social history",
    "family_history": "Family history",
    "physical_exam": "Physical exam",
    "pertinent_results": "Pertinent results",
    "brief_hospital_course": "Brief hospital course",
    "medications_on_admission": "Medications on admission",
    "medications_on_transfer": "Medications on transfer",
    "discharge_medications": "Discharge medications",
    "discharge_disposition": "Discharge disposition",
    "admission_diagnosis": "Admission diagnosis",
    "discharge_diagnosis": "Discharge diagnosis",
    "discharge_condition": "Discharge condition",
    "discharge_instructions": "Discharge instructions",
    "discharge_summary": "Discharge summary",
    "followup_instructions": "Follow-up instructions",
    "facility": "Facility",
    "complications": "Complications",
    "consults": "Consultations",
    "assessment_and_plan": "Assessment and plan",
    "pathology": "Pathology",
    "infections": "Infections",
    "transfusions": "Blood transfusions",
    "indications": "Indications for the surgery",
    # Not a note heading: a laboratory trajectory, the other channel a chart answer can
    # rest on. It is here because this map is where a section key becomes words a reader
    # sees, and a lab citation that fell through to the generic branch would read
    # "laboratory trends" in lower case beside "Discharge diagnosis".
    "laboratory_trends": "Laboratory trends",
}


def _section_label(section: object) -> str:
    name = str(section or "")
    return _SECTION_LABELS.get(name, name.replace("_", " ") or "Note passage")


#: What a chart section's INLINE pill says, as opposed to what the references list calls it.
#:
#: What a note section's INLINE pill says, as opposed to what the references list calls it.
#:
#: The pill is the section's own name in Title Case. It was abbreviated for a while — `HPI`,
#: `Discharge dx` — to keep a printed `/ehr` from being a paragraph of badges, and a reader
#: reported the opposite defect: `dx` is a chart shorthand, `HPI` is a ward shorthand, and
#: neither is what the section is called (2026-11-02). The pill is now the heading itself,
#: compacted by its styling rather than by its vocabulary, so the words on the badge and
#: the words in the note agree.
#:
#: A section this map has not caught falls back to `_section_label`, title-cased, because a
#: new section arriving as an empty pill is worse than one arriving long.
_PILL_SECTION_LABELS = {
    "name": "Patient Name",
    "unit_no": "Unit Number",
    "admission_date": "Admission Date",
    "discharge_date": "Discharge Date",
    "date_of_birth": "Date of Birth",
    "sex": "Sex",
    "service": "Service",
    "allergies": "Allergies",
    "attending": "Attending Physician",
    "chief_complaint": "Chief Complaint",
    "major_procedure": "Major Procedure",
    "history_of_present_illness": "History of Present Illness",
    "review_of_systems": "Review of Systems",
    "past_medical_history": "Past Medical History",
    "past_surgical_history": "Past Surgical History",
    "social_history": "Social History",
    "family_history": "Family History",
    "physical_exam": "Physical Exam",
    "pertinent_results": "Pertinent Results",
    "brief_hospital_course": "Brief Hospital Course",
    "medications_on_admission": "Medications on Admission",
    "medications_on_transfer": "Medications on Transfer",
    "discharge_medications": "Discharge Medications",
    "discharge_disposition": "Discharge Disposition",
    "admission_diagnosis": "Admission Diagnosis",
    "discharge_diagnosis": "Discharge Diagnosis",
    "discharge_condition": "Discharge Condition",
    "discharge_instructions": "Discharge Instructions",
    "discharge_summary": "Discharge Summary",
    "followup_instructions": "Follow-up Instructions",
    "facility": "Facility",
    "complications": "Complications",
    "consults": "Consultations",
    "assessment_and_plan": "Assessment and Plan",
    "pathology": "Pathology",
    "infections": "Infections",
    "transfusions": "Blood Transfusions",
    "indications": "Indications for the Surgery",
    "laboratory_trends": "Laboratory Trends",
}


#: Joining words a Title Case phrase lower-cases unless they open or close it.
_TITLE_CASE_MINOR = frozenset({
    "of", "and", "the", "in", "on", "at", "to", "a", "an", "for", "with", "by",
})


def _title_case(text: object) -> str:
    """A phrase in Title Case, leaving words that already carry capitals alone.

    The second rule is what keeps this usable on journal names: `JAMA`, `PLoS` and
    `eClinicalMedicine` are spelled by their publishers, and a cased-every-word pass would
    produce `Jama`, `Plos` and `Eclinicalmedicine`. A word with a capital anywhere but its
    first letter is a name, so it is passed through untouched; everything else is cased.
    """
    words = " ".join(str(text or "").split()).split(" ")
    cased: list[str] = []
    for position, word in enumerate(words):
        if not word:
            continue
        if word.isupper() or any(character.isupper() for character in word[1:]):
            cased.append(word)
            continue
        lowered = word.lower()
        if 0 < position < len(words) - 1 and lowered in _TITLE_CASE_MINOR:
            cased.append(lowered)
            continue
        cased.append(word[:1].upper() + word[1:])
    return " ".join(cased)


def _pill_section_label(section: object) -> str:
    name = str(section or "")
    return _PILL_SECTION_LABELS.get(name) or _title_case(_section_label(section))



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
            # reads `Brief Hospital Course` and not a bare `1`, because a number tells the
            # reader nothing about what they are about to open — and resolving it here
            # keeps the section vocabulary in the one module that already owns it
            # (`_SECTION_LABELS`), rather than a second copy in the client that would
            # drift the first time a section is renamed.
            #
            # `label` is the REFERENCES LIST's line, number and all. `pill_label` is what
            # the inline control prints: the section name alone, because the sentence
            # already tells the reader which source it is a footnote to and the number
            # lives one line below in the list.
            "label": f"{int(source['cite'])} · {title}",
            "pill_label": _pill_section_label(source.get("section")),
            # Which channel the pill belongs to, so the surface can colour a chart source
            # differently from a published one. Decided here: a browser that classifies
            # its own evidence has taken a citation decision. `note` is a passage out of
            # this patient's own record; `literature` is somebody else's patients.
            "kind": "note",
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


def _pill_journal(name: object, abbreviation: object = "") -> str:
    """A journal as an inline pill spells it: the compact name, Title Case, no year.

    A pill sits inside a sentence, so it carries the journal and nothing else. Not the
    record number, which belongs to the bibliography a line below, and not the year —
    `Spine · 2020` is a second citation line rather than a marker, and beside a rate cell
    it was wide enough to stretch the comparison table's source column (2026-11-02).

    The compact form is the record's own ISO abbreviation: `JAMA Netw Open`, `N Engl J
    Med`, `Spine`. A clinician cites a paper by that string and recognises it faster than
    a title. Truncating the full title instead, which is what this used to do, produced
    `Journal of the American…` — a fragment of a title rather than the name of a journal.

    A record that states neither is `PubMed`, which is true and short rather than blank.
    """
    full = " ".join(str(name or "").split())
    short = " ".join(str(abbreviation or "").split())
    chosen = short or full
    return _title_case(chosen) if chosen else "PubMed"


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
    journal_abbrev = str(article.get("journal_abbrev") or "").strip()
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
    # What the REFERENCES LIST calls the record: the number, the journal and the year. The
    # list is a bibliography, and a bibliography is numbered and dated.
    label = f"{index} · " + " · ".join(part for part in (
        _short_source(journal) or "PubMed", year) if part)
    # What the INLINE pill prints: the journal alone. Two strings because they are read in
    # two places — a numbered line in a bibliography, and a marker inside a sentence that
    # the reader is meant to pass over and come back to.
    pill_label = _pill_journal(journal, journal_abbrev)
    return {
        "index": index,
        "kind": "Literature",
        "badge": _LITERATURE_BADGE,
        "title": title,
        "journal": journal,
        "journal_abbrev": journal_abbrev,
        "year": year,
        "authors": authors,
        "label": label,
        "pill_label": pill_label,
        "subtitle": subtitle,
        "journal_citation": journal_citation,
        # The passage: what the record reports. A reference that opens onto a title and
        # nothing else cannot be weighed.
        "text": _excerpt(article.get("abstract")),
        "pmid": pmid,
        "url": str(article.get("url") or "").strip() or (
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""),
    }


def literature_citations(articles: list[dict], start: int = 1) -> list[dict]:
    """One citation per retrieved article, shaped like every other citation.

    `index` is the article's position in the record order the model was given, which is
    exactly what the formatting instruction tells it to cite — the marker in the prose and
    the record beneath it are the same record by construction.

    `start` is where this turn's note citations end. A turn resting on both the chart and
    the literature publishes one `sources` array — note passages first, then the records
    — and `literature_sources` numbers the records to continue it. The citation list has
    to use the same numbers or the pill and the drawer row disagree; from 1 always, the
    third article of a mixed turn shared the number of the third note passage.

    The marker's DISPLAY is the journal, resolved to the record's own compact name; the
    identifier stays the link target so a reader who wants the paper is one click from the
    card's title. That trade was made on 2026-10-15 (journal over identifier), revisited on
    2026-10-16 (the pills were reading long, so the journal was cut to `_CITE_SOURCE_CHARS`),
    and settled on 2026-11-02: a pill is the journal alone — `JAMA Netw Open`, `Spine` — and
    the year belongs to the bibliography. The record also carries the article, its authors
    and its abstract, because the card behind the pill is the same card a note citation
    opens.

    `source` is the EYEBROW the card shows under the title. For a note passage that is
    "Discharge note" and the section is the title; for a published record the title is the
    article, so the eyebrow is the citation line itself.
    """
    citations = []
    for position, article in enumerate(articles, start=start):
        record = _article_record(article, position)
        citation = {
            "index": record["index"],
            "source": record["subtitle"],
            "title": record["title"],
            "label": record["label"],
            # The inline control prints the journal alone: the number is the references
            # list's, and the year is what made the pill a citation line rather than a
            # marker. `kind` is what the surface colours it by — a published cohort and a
            # note section are weighed differently.
            "pill_label": record["pill_label"],
            "kind": "literature",
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


def literature_sources(articles: list[dict], start: int = 1,
                       *, cited: list[str] | None = None) -> list[dict]:
    """The retrieved records as `sources` entries, numbered from `start`.

    A literature turn had no `sources` at all: the records lived in the `literature`
    payload, which the canvas drew as cards, and the citation channel held only note
    passages. The consequence was a turn whose markers could not address anything — the
    browser resolved `^[PMID: n]` by identifier as a special case, and a `^[1]` written on
    a literature turn pointed at nothing.

    They are `kind="Literature"` for the same reason the comparison table's cohorts are:
    a published record was not retrieved from THIS patient's note, so it carries the
    identifier and the link a reader can open instead of a retrieval query.

    `cited` filters the list to the records the answer actually names (see
    `citations.cited_pmids`). A search returns what it returns; a reference list is what
    the answer rests on, and publishing three rows over an answer that cites one tells the
    reader two papers were used when they were not (patient 90000069, turn 3,
    2026-10-31). The NUMBER is kept as the record's position in the array, so a surviving
    record still carries the number its pill shows; dropping the neighbours leaves a gap,
    which is a smaller falsehood than a row nothing points at.

    An empty `cited` publishes every record. That is not the same statement as "the answer
    cited none of these": a turn that names no record at all still read what the search
    returned, and deleting the whole list would say nothing was retrieved.
    """
    wanted = {str(pmid) for pmid in cited or ()}
    sources = []
    for offset, article in enumerate(articles):
        number = start + offset
        if wanted and str((article or {}).get("pmid") or "") not in wanted:
            continue
        record = _article_record(article, number)
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


def _narrative(answer: str, sources: list[dict], literature: dict | None,
               *, literature_start: int = 1) -> dict | None:
    """The narrative surface for this answer, or None when there is no prose.

    Citations come from every kind of evidence the turn rests on, and a mixed turn
    carries BOTH channels. `sources` holds the resolved note passages and `literature`
    the retrieved records; on a turn that summarised an article and then the admission,
    the article's badge and the note badges address the same `sources` array, which is
    why the record numbers continue after the note numbers (`literature_sources(start=…)`
    and `literature_citations(start=…)` agree by the same argument).

    This chose one channel or the other until 2026-10-30, on the reasoning that a
    literature turn cites its articles "and nothing else". That holds for a turn whose
    answer is only about the literature; on a mixed answer it handed the component an
    article-only citation list, and every note marker in the second paragraph became an
    unresolvable control — the grey badges a reader saw and could not click.

    The turn is only a literature turn when it produced RECORDS. An empty search result
    is a `dict`, so a bare `is not None` test made every chart answer that happened to
    carry a fruitless PubMed call cite zero articles: the note passages had already been
    resolved, and this function threw them away in favour of an empty list. The reader
    then had a `^[1]` that resolved to nothing beside a card saying no literature was
    found (2026-10-22).
    """
    if not answer.strip():
        return None
    notes = note_citations(sources)
    articles = _usable_articles(literature or {})
    if not articles or literature is None:
        return narrative_surface(answer, notes)
    return narrative_surface(
        answer, notes + literature_citations(articles, start=literature_start))
