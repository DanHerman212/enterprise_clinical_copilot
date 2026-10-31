"""The four target architecture contracts, locked before any production code moves.

Phase 2 of the backend alignment. These tests encode the *target* behaviour from
`docs/architecture_alignment/01a-audit_report.md` and Charter 13, and they are written
to FAIL against the un-remediated code — that is what makes them a lock rather than a
description. Each failure names the audit finding it pins, so Phase 3 and Phase 4 can
work from the red output.

Two of them assert BASELINE behaviour and pass today (`_prose_settled` honouring a
flag, and a chart turn keeping its lab card). They are here so the remediation cannot
"fix" a violation by deleting the feature beside it: the replacement swap stays visible,
and a chart turn keeps drawing its sparklines.

Expected state after this phase: 7 failing, 2 passing. Nothing here is xfail-marked,
because a test that is expected to fail forever cannot tell you when it stops failing.
"""

from __future__ import annotations

import inspect

from services.agent.engine import graph as graph_module
from services.agent.engine import mcp_client
from services.agent.ingress.http import _prose_settled
from services.agent.postprocess import citations
from services.agent.postprocess.guardrail import guard_answer
from services.agent.presentation.a2ui import compose_presentation
from services.agent.presentation.canvas import compose_risk_canvas
from services.agent.routing import routing

ARTICLE = {
    "pmid": "35010760",
    "title": "Accidental hypothermia",
    "journal": "N Engl J Med",
    "pub_date": "2022 Jan",
    "abstract": "Rewarming is the priority; no one is dead until they are warm and dead.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/35010760/",
}

SEARCH = {
    "name": "search_literature",
    "args": {"query": "accidental hypothermia"},
    "response": {"query": "accidental hypothermia", "returned": 1, "articles": [ARTICLE]},
}

PASSAGES = [
    {"section": "brief_hospital_course", "text": "Admitted with hypothermia."},
    {"section": "discharge_diagnosis", "text": "Hypothermia."},
]

CHART = {
    "name": "rag_search",
    "args": {"hadm_id": 90000059, "query": "hypothermia"},
    "response": {"query": "hypothermia", "returned": 2, "passages": PASSAGES},
}

LABS_CALL = {
    "name": "get_lab_trends",
    "args": {"hadm_id": 90000059},
    "response": {"trends": [{"label": "Creatinine", "points": [[0, 1.1], [1, 1.3]]}],
                 "synthetic": True},
}

#: What `compose_risk_canvas` takes: the tool's RESPONSE, not the call record.
LABS = LABS_CALL["response"]


# --------------------------------------------------------------------------- #
# Contract 1 — a withheld reply is resolved in postprocessing, visibly
# --------------------------------------------------------------------------- #

def test_a_withheld_reply_is_resolved_by_postprocess_not_the_engine():
    """Charter 13 Law 2 (audit L2-1).

    The engine streams `"Accidental hypothermia is"` and the provider stops with
    `finish_reason=SAFETY`. Resolving that — discard the fragment, serve the withheld
    sentence, record the reason — is postprocessing's job, because it is prose
    replacement and it has to happen for both the streaming and blocking paths. Today
    the decision lives in `engine/deterministic.py:611-629`, so `guard_answer` cannot
    even be told why the text stopped.
    """
    assert "finish_reason" in inspect.signature(guard_answer).parameters, (
        "Charter 13 Law 2: `postprocess.guard_answer` must resolve a withheld or "
        "truncated reply (audit L2-1). The engine currently does it at "
        "engine/deterministic.py:611-629."
    )


def test_the_withheld_sentence_is_served_with_its_reason_in_the_flags():
    """Charter 13 Law 2 — the sentence AND the flag, so the client can reconcile.

    Both halves matter: the sentence is what the reader sees, the flag is what
    `_prose_settled` reads to decide whether the swap was visible. A replacement with
    no flag is a silent withdrawal of prose the reader already watched arrive.
    """
    assert "finish_reason" in inspect.signature(guard_answer).parameters, (
        "Charter 13 Law 2: `guard_answer` cannot resolve a withheld reply until it is "
        "told why the text stopped (audit L2-1)")

    partial = "Accidental hypothermia is"
    state = {
        "messages": [_ai_message(partial)],
        "tool_calls": [SEARCH],
        "replayed_tool_calls": [],
    }

    guarded = guard_answer(
        partial, [SEARCH], None, finish_reason="SAFETY", state=state)

    assert guarded["answer"].strip() == citations_sentence(), (
        "a reply withheld by the provider must be replaced by the withheld-summary "
        "sentence, not served as the partial fragment")
    assert "safety_withheld" in guarded["flags"], (
        "the replacement must be visible in guardrail_flags, or the client reports a "
        "silent swap for prose that was taken away")


def test_prose_settled_treats_a_withheld_flag_as_a_visible_swap():
    """Baseline, and it passes today: any flag makes the swap visible.

    Pinned because the remediation moves the decision, and the cheapest way to make
    Contract 1 green would be to stop flagging it. The reader still gets told.
    """
    swap = _prose_settled(
        ["Accidental hypothermia is"],
        {"answer": citations_sentence(),
         "guardrail_flags": ["safety_withheld"]},
        "trace-under-test",
    )

    assert swap == "visible"


def citations_sentence():
    """The withheld sentence, imported late so a rename fails here and nowhere else."""
    from services.agent.postprocess import answer

    return answer.SAFETY_WITHHELD_SUMMARY.strip()


def _ai_message(content: str):
    from langchain_core.messages import AIMessage

    return AIMessage(content=content)


# --------------------------------------------------------------------------- #
# Contract 2 — a literature question is scoped to literature, and cannot loop
# --------------------------------------------------------------------------- #

def test_a_literature_question_binds_no_lab_or_medication_tool():
    """Charter 13 Principle 1 + Phase 4 (audit §2.1).

    `_freetext_allowed` unions `LITERATURE_TOOLS` into the whole chart set, so a PubMed
    question keeps `get_lab_trends` and `discharge_medications` bound. The observed
    consequence is a five-call turn on one literature question (`079be88f…`: one
    `rag_search` and four `search_literature`).

    The passage tools stay: the question is answered against the patient's record and
    the literature, so `rag_search` and `rag_search_sections` are how the record half
    is read. The two that go are the ones that answer a DIFFERENT question — a lab
    panel and a medication list — which is why they are named one at a time.
    """
    allowed = graph_module._freetext_allowed(risk_driver_question=False,
                                             wants_literature=True)

    expected = ((set(routing.LITERATURE_TOOLS) | set(routing.PASSAGE_TOOLS))
                - {"discharge_medications"})
    assert set(allowed) == expected, (
        "a literature turn must bind literature plus record retrieval and nothing else "
        "(Phase 4; audit §2.1). Bound today: " + ", ".join(sorted(allowed)))
    assert "get_lab_trends" not in allowed
    assert "discharge_medications" not in allowed


def test_the_literature_tool_has_its_own_call_cap():
    """Phase 4 (audit §2.1): the only bound today is five calls of ANY tool.

    One literature turn made four `search_literature` calls with refined wording. A
    per-tool cap is what stops the third and fourth from being spent on a question the
    first two already answered, and it has to be a declared constant rather than a
    number written inside the loop.
    """
    limit = getattr(mcp_client, "PER_TOOL_CALL_LIMIT", None)

    assert isinstance(limit, int), (
        "engine/mcp_client.py must declare PER_TOOL_CALL_LIMIT; MAX_TOOL_CALLS_PER_TURN "
        "bounds the turn, not one tool (audit §2.1)")
    assert 1 <= limit <= 2, (
        f"the observed turn made four redundant searches; a cap of {limit} would not "
        "have stopped it")


# --------------------------------------------------------------------------- #
# Contract 3 — one ledger, and the trace records what the caller was served
# --------------------------------------------------------------------------- #

def test_citations_owns_a_canonical_evidence_ledger():
    """Charter 13 Law 1 (audit §2.3, §3).

    Four modules resolve the turn's evidence today — `citations` counts it, `engine`
    numbers it for the model, `presentation` re-counts it and re-orders markers, and
    `evaluation` re-derives it. The ledger is the single structure Phase 5 builds so
    that a marker has one meaning: the display number the reader sees.
    """
    assert hasattr(citations, "EvidenceLedger"), (
        "postprocess/citations.py must expose the canonical EvidenceLedger "
        "(Phase 5; audit §2.3)")


def test_presentation_does_not_alter_the_answer_it_is_given():
    """Charter 13 Law 5, the caveat in the audit: trace output == served prose.

    `turn.py` records `guarded["answer"]` on the trace, and `compose_presentation`
    publishes a RENUMBERED answer as the payload's `answer` (`a2ui.py:363`). The two
    differ whenever the model writes a cluster — `^[4, 5]` becomes `^[4] ^[5]` — so the
    audit trail and the browser disagree about the text of one turn.
    """
    settled = "Hypothermia was documented.^[4, 5]"

    presentation = compose_presentation(
        "what happened during the admission?", settled, [CHART])

    assert presentation["answer"] == settled, (
        "presentation must publish the settled prose unchanged (Charter 13 Law 5; "
        "audit L1-1/L5): the trace records the input, the caller reads the output")
    # Baseline that must survive the fix: the marker cluster reaches the client
    # unaltered. It is ONE marker here — `^[4, 5]` is a single cluster, and the count is
    # one. Expanding it to `^[4] ^[5]` is the rewrite the assertion above forbids, and it
    # is already done before this layer sees the text: `citations.enforce_citations`
    # emits one marker per unit, so a cluster never survives settlement. What this pins
    # is that the cluster form itself is not mistaken for two markers and rewritten.
    assert presentation["answer"].count("^[") == 1
    assert "^[4, 5]" in presentation["answer"]


# --------------------------------------------------------------------------- #
# Contract 4 — a canvas is scoped to its turn
# --------------------------------------------------------------------------- #

def test_a_literature_turn_does_not_draw_a_lab_card():
    """Phase 6 (audit §2.2).

    `compose_risk_canvas` appends the lab card whenever a `get_lab_trends` payload is
    in hand (`canvas.py:~305-315`) and the literature branch appends after it without
    removing it. A turn that asked about the literature and also called a lab tool
    therefore draws a sparkline panel under a literature answer — reachable exactly
    because of Contract 2's violation.

    The payloads are the tool RESPONSES, which is what the composer takes: a call
    record would silently satisfy every `isinstance(x, dict)` test in here and prove
    nothing. The first run of this test passed the call records and measured nothing.

    `narrative` is supplied because `compose_presentation` always supplies one, and the
    records reach the reader through it: a literature turn publishes its articles as
    numbered pills in the prose, so with prose present the canvas draws `CitedNarrative`
    and no second copy of the list (Phase F). The fixture has to be the call the product
    makes, or it measures a shape no caller produces.
    """
    envelope = compose_risk_canvas(
        None, None, literature=SEARCH["response"], labs=LABS,
        narrative={"text": "One record supports this.^[1]", "citations": []},
        command="literature")
    kinds = _component_kinds(envelope)

    assert "LabTrendCard" not in kinds, (
        "a literature turn must not carry a chart surface (audit §2.2). "
        "Components: " + ", ".join(kinds))
    assert "LiteratureList" in kinds or "CitedNarrative" in kinds, (
        "the literature answer itself must still be drawn; components: "
        + ", ".join(kinds))


def test_a_chart_turn_keeps_its_lab_card():
    """Baseline, passing today: the fix must scope the card, not delete it.

    The lab drill-down IS the answer on its own command, so removing the card from the
    canvas to satisfy the test above would trade one wrong surface for a missing one.
    """
    envelope = compose_risk_canvas(None, None, labs=LABS, command="labs")
    kinds = _component_kinds(envelope)

    assert "LabTrendCard" in kinds


def _component_kinds(envelope: dict) -> list[str]:
    kinds: list[str] = []
    for message in envelope.get("messages") or []:
        for component in (message.get("updateComponents") or {}).get("components") or []:
            kinds.append(str(component.get("component") or ""))
    return kinds


# --------------------------------------------------------------------------- #
# Contract 5 — the reader gets 1..K; the array keeps its numbering
# --------------------------------------------------------------------------- #

REPLAY = [
    {"name": "rag_search_sections", "args": {}, "response": {"passages": [
        {"section": "brief_hospital_course", "text": "Course."},
        {"section": "discharge_diagnosis", "text": "Diagnosis."},
        {"section": "discharge_medications", "text": "Meds."},
        {"section": "discharge_instructions", "text": "Instructions."},
    ]}},
    {"name": "rag_search", "args": {}, "response": {"passages": [
        {"section": f"section_{i}", "text": f"passage {i}"} for i in range(9)
    ]}},
]
LAB_PANEL = {"name": "get_lab_trends", "args": {}, "response": {"trends": [
    {"label": "RDW", "series": [{"day": 0, "value": 12.0}]},
    {"label": "Sodium", "series": [{"day": 0, "value": 140.0}]},
]}}
LATER_PASSAGE = {"name": "rag_search", "args": {}, "response": {"passages": [
    {"section": "discharge_instructions", "text": "Follow up in one week."},
]}}


def test_settlement_compacts_the_markers_while_the_array_keeps_its_positions():
    """Charter 13 Law 1 and Law 5, from the reader's side.

    Turn 3 of the 90000023 thread was served `^[14]` and `^[15]`: the positions of two
    laboratory series in a conversation array 13 units deep. The numbers were correct and
    unreadable. Settlement now renumbers the served prose to the order it cites, and
    publishes the map back to the array so a pill, a drawer row and an evaluation source
    keep naming the same unit.

    Both halves are asserted. A fix that compacted the prose without the map would point
    every pill at the wrong passage — the failure this contract exists to prevent.
    """
    from services.agent.postprocess import answer as answer_text

    state = {
        "messages": [_ai_message("Notes do not rank them ^[14], ^[15].")],
        "tool_calls": [LAB_PANEL, LATER_PASSAGE],
        "replayed_tool_calls": REPLAY,
        "deterministic_answer": None,
    }

    # The array is 16 units, so the two cited ones are the panel's analytes at 14 and 15.
    assert citations.citation_bound(state["tool_calls"],
                                    state["replayed_tool_calls"]) == 16

    answer_text.sanitize_answer_markers(state)

    assert state["messages"][-1].content == "Notes do not rank them ^[1], ^[2]."
    assert state["citation_display_map"] == {"1": 14, "2": 15}


def test_a_free_text_turn_that_read_the_panel_does_not_draw_it():
    """The card answers "show me the course"; a question about something else did not ask.

    Turn 3 read the panel and drew it under an answer that never discussed it. The tool
    stays bound — the prompt prefers a series for every quantitative value, so withholding
    it would trade one defect for a worse one — and the SURFACE is what is scoped.
    """
    drawn = compose_presentation(
        "which issue should be treated first?",
        "The record does not rank the conditions.^[1]",
        [LAB_PANEL],
    )
    assert "LabTrendCard" not in _component_kinds(drawn["a2ui"])

    # The baseline that must survive: a question that names an analyte still gets it.
    named = compose_presentation(
        "how did her sodium move?",
        "Sodium rose.^[1]",
        [LAB_PANEL],
    )
    assert "LabTrendCard" in _component_kinds(named["a2ui"])


# --------------------------------------------------------------------------- #
# Contract 6 — settlement acts on prose, not on the message's container
# --------------------------------------------------------------------------- #

def test_settlement_settles_a_streamed_answer_whose_content_is_blocks():
    """Charter 13 Law 1: the bound applies to every served answer, whatever its shape.

    A model message's `content` is not always a string. Gemini answers in content blocks,
    so the streamed free-text turn accumulates `[{'type': 'text', 'text': '…'}]` — the
    shape `engine/model.text_of` exists to unwrap. Settlement gated every step on
    `isinstance(content, str)`, so on every streamed turn it enforced nothing: markers
    past the end of the array were served, and no numbering was applied. Patient 90000023
    turn 3 was served `^[23]` against a shorter array (`agent-00227-n7g`, 2026-10-31).

    Both halves are asserted: the in-range marker is renumbered for the reader, and the
    out-of-range one is removed and reported. A test that only checked the first would
    pass against a settlement that still did no enforcement.
    """
    import asyncio
    from unittest.mock import patch

    from services.agent.engine import graph as graph_module

    replay = [{"name": "rag_search_sections", "args": {}, "response": {"passages": [
        {"section": f"s{i}", "text": f"section {i}"} for i in range(4)
    ]}}]
    this_turn = [{"name": "rag_search", "args": {}, "response": {"passages": [
        {"section": f"p{i}", "text": f"passage {i}"} for i in range(5)
    ]}}]
    blocks = [{"type": "text", "text": "The notes do not rank them ^[5]. "
                                      "The record lists them ^[99]."}]

    class _Stub:
        async def ainvoke(self, *_args, **_kwargs):
            return {"messages": [_ai_message(blocks)], "tool_calls": this_turn,
                    "replayed_tool_calls": replay}

    with patch.object(graph_module, "build_graph", lambda *_a, **_k: _Stub()):
        state = asyncio.run(graph_module.ask(
            object(), "which issue first? For admission 90000023."))

    assert state["messages"][-1].content == (
        "The notes do not rank them ^[1]. The record lists them.")
    assert state["citation_display_map"] == {"1": 5}
    assert state["guardrail_flags"] == ["citation_out_of_range:^99"]

