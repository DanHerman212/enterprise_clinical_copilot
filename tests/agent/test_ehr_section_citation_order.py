"""TF-7: a summary's numbered sources are the sections in the order it writes them.

`/ehr` on 90000015 cross-wired `^[3]` and `^[4]` (Gate 1 re-verification,
`phase04-gate1-canary-20261023-161739`). Retrieval returned the discharge
instructions before the discharge diagnosis; the formatter is told to write the diagnosis
third and the instructions fourth; the model numbered its markers by the position of
the section it was writing, and the source list kept the retrieval order. A reader
clicking `^[3]` inside the Discharge diagnosis paragraph landed on the medication list.

The fix orders the passages canonically before the citation layer, the model or the
envelope sees them, so position and provenance are one fact. These tests pin that:
retrieval order in, canonical order out, on both the recorded evidence and the
sources a reader clicks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from services.agent.engine import deterministic
from services.agent.engine import model as model_gateway
from services.agent.engine import graph
from services.agent.engine import model
from services.agent.routing import routing
from services.agent.engine import prompts
from services.agent.presentation.a2ui import compose_presentation
from services.agent.engine.deterministic import run_command
from services.mcp.retrieval.sections import SUMMARY_SECTIONS

SUMMARIZE = routing.command_for("summarize")

#: The order `rag_search_sections` actually returned for 90000015: the sectioned call
#: happens to put the discharge instructions ahead of the discharge diagnosis, which is
#: the whole reason the markers crossed.
RETRIEVAL_ORDER = (
    ("history_of_present_illness", "A 61-year-old woman admitted for rehabilitation."),
    ("brief_hospital_course", "Admitted 08/24/2007 to the transitional care unit."),
    ("discharge_instructions", "Continue Coumadin; INR every Monday and Thursday."),
    ("discharge_diagnosis", "Status post right total knee replacement."),
)

ANSWER = (
    "**Hospital course.** She was admitted for rehabilitation ^[1]. "
    "The stay was uneventful ^[2].\n\n"
    "**Discharge diagnosis.** Status post right total knee replacement ^[3].\n\n"
    "**Discharge instructions.** Continue Coumadin ^[4]."
)


class _SectionToolbox:
    """Returns the passages for whichever tool the command declares.

    Any tool, not only the sectioned search: the point of these tests is the order the
    recorded evidence ends up in, and the labs command reads its own tool.
    """

    def __init__(self, passages=RETRIEVAL_ORDER) -> None:
        self.passages = passages

    async def call(self, name, args):
        if name not in ("rag_search_sections", "get_lab_trends"):
            # The risk estimate a summary turn also asks for is not a passage and must
            # not be counted as one; the record-wide fallback is only reached when the
            # sectioned call returns nothing, which no test here arranges.
            return {"returned": 0, "passages": []}
        return {"returned": len(self.passages),
                "passages": [{"section": section, "text": text}
                             for section, text in self.passages]}


class _Model:
    def __init__(self, reply: str = ANSWER) -> None:
        self.reply = reply
        self.calls = 0
        self.seen: list[list] = []

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the formatting turn was given tools")

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        self.seen.append(list(messages))
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.reply)


def _run(hadm_id: int = 90000015, *, cabinet=None, reply: str = ANSWER):
    box = cabinet or _SectionToolbox()
    model = _Model(reply)
    with patch.object(model_gateway, "build_llm", lambda name: model):
        state = asyncio.run(run_command(
            box, SUMMARIZE, hadm_id, question="/ehr", model="test-model",
        ))
    return state, box, model


def test_the_recorded_passages_come_back_in_canonical_section_order():
    state, _, _ = _run()

    sections = [passage["section"]
                for call in state["tool_calls"]
                for passage in (call.get("response") or {}).get("passages") or []]
    assert sections == list(SUMMARY_SECTIONS)


def test_the_sources_a_reader_clicks_are_in_the_same_order_as_the_sections():
    state, _, _ = _run()
    out = compose_presentation("/ehr", state["messages"][-1].content,
                               state["tool_calls"], None, "summarize")

    # The note passages first, in canonical order — the citation channel. The comparison
    # table's published cohort follows them (2026-10-26, Phase B): the matrix names a
    # journal, a year and a PMID on screen, and without this the turn held no record of
    # any of them and the benchmark could not be opened from the turn that showed it.
    note_sources = [s for s in out["sources"] if s.get("kind") != "Literature"]
    assert [source["section"] for source in note_sources] == list(SUMMARY_SECTIONS)
    # And the marker numbers run 1..4 against that order, so `^[3]` is the diagnosis.
    assert [source["cite"] for source in note_sources] == [1, 2, 3, 4]
    assert out["sources"][2]["section"] == "discharge_diagnosis"
    # The benchmark is numbered to continue the same array, so a marker written anywhere
    # in this turn addresses one list.
    assert out["sources"][-1]["kind"] == "Literature"
    assert out["sources"][-1]["cite"] == 5


def test_the_diagnosis_passage_is_not_displaced_by_the_instructions():
    """The swap itself: `^[3]` must not resolve to the medication text."""
    state, _, _ = _run()
    out = compose_presentation("/ehr", state["messages"][-1].content,
                               state["tool_calls"], None, "summarize")

    third = out["sources"][2]
    assert third["section"] == "discharge_diagnosis"
    assert "Coumadin" not in third["text"]
    assert "knee replacement" in third["text"]


def test_an_unnamed_section_keeps_its_place_after_the_named_ones():
    """Ordering is a stable sort, so a heading the tuple does not name is not dropped."""
    cabinet = _SectionToolbox(passages=(
        ("discharge_instructions", "Continue Coumadin."),
        ("social_history", "Lives alone."),
        ("history_of_present_illness", "Admitted for rehabilitation."),
    ))
    state, _, _ = _run(cabinet=cabinet)

    sections = [passage["section"]
                for call in state["tool_calls"]
                for passage in (call.get("response") or {}).get("passages") or []]
    assert sections == ["history_of_present_illness", "discharge_instructions",
                        "social_history"]


def test_a_single_passage_is_left_alone():
    """One passage cannot be out of order, and the ordering must not renumber it."""
    cabinet = _SectionToolbox(passages=(("discharge_diagnosis", "One section."),))
    state, _, _ = _run(cabinet=cabinet)

    sections = [passage["section"]
                for call in state["tool_calls"]
                for passage in (call.get("response") or {}).get("passages") or []]
    assert sections == ["discharge_diagnosis"]


def test_the_ordering_is_scoped_to_the_summary_command():
    """`/labs` and `/meds` read their passages in retrieval order on purpose."""
    labs = routing.command_for("labs")
    cabinet = _SectionToolbox(passages=(("sodium", "136"), ("potassium", "4.1")))
    model = _Model(reply="Sodium 136, potassium 4.1.")
    with patch.object(model_gateway, "build_llm", lambda name: model):
        state = asyncio.run(run_command(
            cabinet, labs, 90000015, question="/labs", model="test-model",
        ))

    sections = [passage["section"]
                for call in state["tool_calls"]
                for passage in (call.get("response") or {}).get("passages") or []]
    assert sections == ["sodium", "potassium"]


def test_the_formatter_is_told_to_cite_the_passage_not_the_section_position():
    # One assertion per line of the instruction, because the prose wraps and a
    # substring that spans a wrap asserts nothing.
    assert "position of the section you are writing" in prompts.EHR_FORMATTING
    assert "carries a `cite` value" in prompts.EHR_FORMATTING


def test_each_passage_carries_the_number_its_marker_must_use():
    """The label that removes the counting step, and the reason TF-7 took two attempts.

    The model used to read an unlabelled array and count positions; on a four-section
    summary it put `^[3]` on the instructions and `^[4]` on the diagnosis, both of which
    begin with "discharge". The rendered view now states the number.
    """
    from services.agent.engine.graph import _render_tool_result  # noqa: PLC0415

    rendered = _render_tool_result({"passages": [{"section": "a"}, {"section": "b"},
                                                 {"section": "c"}]})
    assert [entry["cite"] for entry in __import__("json").loads(rendered)["passages"]] \
        == [1, 2, 3]
    # And the payload the citation layer resolves against is untouched.
    payload = {"passages": [{"section": "a"}, {"section": "b"}]}
    _render_tool_result(payload)
    assert all("cite" not in passage for passage in payload["passages"])


def test_the_payload_hook_leaves_an_unnamed_section_alone():
    payload = {"passages": [{"section": "zeta"}, {"section": "alpha"}]}
    ordered = deterministic._ordered_summary_payload("rag_search", payload)

    assert [p["section"] for p in ordered["passages"]] == ["zeta", "alpha"]


def test_a_response_without_sections_is_untouched():
    payload = {"passages": [{"text": "no section"}, {"text": "also none"}]}
    assert deterministic._ordered_summary_payload("rag_search", payload) is payload


def test_the_formatter_reads_the_same_order_the_citation_layer_numbers():
    """The regression this fix exists for.

    Reordering the recorded payload alone left the model reading retrieval order while
    the envelope numbered canonically, so `^[3]` still resolved to the wrong section —
    in the other direction. The ordering has to happen before the tool result the model
    is shown is rendered.
    """
    state, _, model = _run()
    shown = model.seen[0][-1].content

    positions = [shown.index(section) for section in SUMMARY_SECTIONS]
    assert positions == sorted(positions), (
        "the formatter was shown the sections out of canonical order")
    recorded = [p["section"]
                for call in state["tool_calls"]
                for p in (call.get("response") or {}).get("passages") or []]
    assert recorded == list(SUMMARY_SECTIONS)


def test_tf8_prohibits_a_closing_vigilance_sentence():
    assert "Do not close with clinical advice" in prompts.EHR_FORMATTING
    assert "vigilance" in prompts.EHR_FORMATTING
