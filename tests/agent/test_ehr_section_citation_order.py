"""A summary's numbered sources are its sections, in the order the note states them.

TF-7 (2026-10-23) was a cross-wiring: retrieval returned the discharge instructions before
the discharge diagnosis, the model numbered its markers by the position of the section it
was writing, and the source list kept the retrieval order — so a reader clicking `^[3]`
inside the Discharge diagnosis paragraph landed on the medication list.

The fix at the time ordered the passages into a canonical four-section sequence before
anyone saw them. That is gone (2026-11-03): the retrieval tool now returns the note's own
sections in document order, so the canonical re-sort was a second opinion about how a chart
reads, and it hid every section outside the four. What survives from TF-7 is the invariant,
which is what this file pins: the order the formatter is shown, the order the citation
layer numbers, and the order the tool returned are one order. A re-sort anywhere downstream
breaks it, and these tests are written so a re-sort would visibly change the order — the
passages below are deliberately NOT in canonical sequence.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from services.agent.engine import deterministic
from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.engine import prompts
from services.agent.engine.deterministic import run_command
from services.agent.presentation.a2ui import compose_presentation
from services.agent.routing import routing

SUMMARIZE = routing.command_for("summarize")

#: The note's own order, with the discharge instructions before the discharge diagnosis.
#: Deliberately not the canonical sequence: if anything downstream still sorted sections
#: into that sequence, every assertion in this file would fail rather than agree.
DOCUMENT_ORDER = (
    ("history_of_present_illness", "A 61-year-old woman admitted for rehabilitation."),
    ("brief_hospital_course", "Admitted 08/24/2007 to the transitional care unit."),
    ("discharge_instructions", "Continue Coumadin; INR every Monday and Thursday."),
    ("discharge_diagnosis", "Status post right total knee replacement."),
)

ANSWER = (
    "**Presentation & diagnosis.** She was admitted for rehabilitation ^[1].\n\n"
    "**Hospital course & procedures.** The stay was uneventful ^[2].\n\n"
    "**Discharge plan & medications.** Continue Coumadin ^[3]. Status post right total "
    "knee replacement ^[4]."
)


class _SectionToolbox:
    """Returns the passages for whichever tool the command declares.

    Any tool, not only the sectioned search: the point of these tests is the order the
    recorded evidence ends up in, and the labs command reads its own tool.
    """

    def __init__(self, passages=DOCUMENT_ORDER) -> None:
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


def _recorded_sections(state) -> list[str]:
    return [passage["section"]
            for call in state["tool_calls"]
            for passage in (call.get("response") or {}).get("passages") or []]


def test_the_recorded_passages_keep_the_document_order():
    state, _, _ = _run()

    assert _recorded_sections(state) == [section for section, _ in DOCUMENT_ORDER]


def test_the_sources_a_reader_clicks_are_in_the_same_order_as_the_passages():
    state, _, _ = _run()
    out = compose_presentation("/ehr", state["messages"][-1].content,
                               state["tool_calls"], None, "summarize")

    # The note passages, in the order the tool returned them — the citation channel. The
    # comparison table's cohorts are whatever literature retrieval verified on the turn;
    # this cabinet serves no literature, so nothing follows them. What is pinned here is
    # the note channel's own numbering, which must not depend on whether a cohort
    # appended itself afterwards.
    note_sources = [s for s in out["sources"] if s.get("kind") != "Literature"]
    assert [source["section"] for source in note_sources] == [
        section for section, _ in DOCUMENT_ORDER]
    # And the marker numbers run 1..4 against that order, so `^[3]` is the instructions and
    # `^[4]` is the diagnosis — the order the note reads in, not a canonical one.
    assert [source["cite"] for source in note_sources] == [1, 2, 3, 4]
    assert out["sources"][3]["section"] == "discharge_diagnosis"
    assert not [s for s in out["sources"] if s.get("kind") == "Literature"]


def test_a_marker_resolves_to_the_passage_at_that_position():
    """The TF-7 failure mode stated directly: `^[4]` must not resolve to the instructions."""
    state, _, _ = _run()
    out = compose_presentation("/ehr", state["messages"][-1].content,
                               state["tool_calls"], None, "summarize")

    fourth = out["sources"][3]
    assert fourth["section"] == "discharge_diagnosis"
    assert "knee replacement" in fourth["text"]
    assert "Coumadin" not in fourth["text"]


def test_a_section_the_summary_does_not_write_still_keeps_its_place():
    """Order is the document's, so an unnamed section is neither dropped nor moved."""
    cabinet = _SectionToolbox(passages=(
        ("discharge_instructions", "Continue Coumadin."),
        ("social_history", "Lives alone."),
        ("history_of_present_illness", "Admitted for rehabilitation."),
    ))
    state, _, _ = _run(cabinet=cabinet)

    assert _recorded_sections(state) == [
        "discharge_instructions", "social_history", "history_of_present_illness"]


def test_no_command_re_orders_its_passages():
    """`/labs` reads its passages in retrieval order, and so does everything else now."""
    labs = routing.command_for("labs")
    cabinet = _SectionToolbox(passages=(("sodium", "136"), ("potassium", "4.1")))
    model = _Model(reply="Sodium 136, potassium 4.1.")
    with patch.object(model_gateway, "build_llm", lambda name: model):
        state = asyncio.run(run_command(
            cabinet, labs, 90000015, question="/labs", model="test-model",
        ))

    assert _recorded_sections(state) == ["sodium", "potassium"]


def test_the_formatter_is_told_to_cite_the_passage_not_the_section_position():
    # One assertion per line of the instruction, because the prose wraps and a
    # substring that spans a wrap asserts nothing. The wording of the attribution rule
    # changed on 2026-11-03 (two rules rather than one long paragraph); what is pinned
    # here is the requirement it states — the passage's own `cite` value, not the
    # position of the section being written.
    assert "position of the section you are writing" in prompts.EHR_FORMATTING
    assert "the `cite` value that passage carries" in prompts.EHR_FORMATTING


def test_the_instruction_states_only_the_two_legal_marker_forms():
    """`^[predict_readmission]` was invented because the rule never said what a marker is.

    The model was told to cite the passage a detail comes from, and a tool result with no
    cite value gave it nothing to cite — so it wrote the tool's name into a marker, which
    settlement deletes, leaving the sentence unattributed (2026-11-03). The instruction now
    names both legal forms and says plainly that a result without a `cite` value takes no
    marker.
    """
    text = " ".join(prompts.EHR_FORMATTING.split())

    assert "^[PMID: 24108298]" in text
    assert "exactly two legal forms" in text
    assert "carries no `cite` value is not citable" in text
    assert "put no marker on it" in text


def test_each_passage_carries_the_number_its_marker_must_use():
    """The label that removes the counting step, and the reason TF-7 took two attempts.

    The model used to read an unlabelled array and count positions; on a four-section
    summary it put `^[3]` on the instructions and `^[4]` on the diagnosis, both of which
    begin with "discharge". The rendered view now states the number.
    """
    import json

    rendered = graph._render_tool_result({"passages": [{"section": "a"}, {"section": "b"},
                                                       {"section": "c"}]})
    assert [entry["cite"] for entry in json.loads(rendered)["passages"]] == [1, 2, 3]
    # And the payload the citation layer resolves against is untouched.
    payload = {"passages": [{"section": "a"}, {"section": "b"}]}
    graph._render_tool_result(payload)
    assert all("cite" not in passage for passage in payload["passages"])


def test_the_formatter_reads_the_same_order_the_citation_layer_numbers():
    """The regression TF-7 exists for, and the one a re-sort would reintroduce.

    Re-ordering the recorded payload alone left the model reading one order while the
    envelope numbered another, so a marker resolved to the wrong section — in whichever
    direction the disagreement ran. The order has to be the same at all three points: what
    the model is shown, what the citation layer numbers, and what the tool returned.
    """
    state, _, model = _run()
    shown = model.seen[0][-1].content

    positions = [shown.index(section) for section, _ in DOCUMENT_ORDER]
    assert positions == sorted(positions), (
        "the formatter was shown the sections in an order the note does not read in")
    assert _recorded_sections(state) == [section for section, _ in DOCUMENT_ORDER]


def test_the_re_ordering_hook_is_gone():
    """`_ordered_summary_payload` put a summary's passages into a canonical sequence.

    It is deleted rather than left unused, because the tool now returns the document's own
    order and a second ordering in this layer would be a second opinion about the chart.
    """
    assert not hasattr(deterministic, "_ordered_summary_payload")


def test_tf8_prohibits_a_closing_vigilance_sentence():
    assert "Do not close with clinical advice" in prompts.EHR_FORMATTING
    assert "vigilance" in prompts.EHR_FORMATTING
