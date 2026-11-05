"""The two seams `15-defect_remediation.md` opened: the manifest on `/ehr`, and the
numbering of a replayed answer's markers.

**Manifest parity.** `/ehr` states where the patient went; the discharge disposition is an
administrative field the pre-flight block holds and the notes only sometimes state. The block
reached the free-text loop's system prompt and nothing else, so a command was formatted without
it and then graded against it — `dev_t1_02` and `dev_t2_01` served byte-identical `/ehr` answers,
against identical manifests, and were scored fidelity 3 and fidelity 1 in the same run.

**Replay numbering.** The served page numbers a turn's citations `1..K` in the order they are
cited; every `cite` in a tool result is a position in the conversation's array. The answer
shipped without the map between them, so the only safe thing a replay could do with a prior
answer was delete its numbers (`citations.strip_numeric_markers`) — which also deleted the
pattern a follow-up imitates, and a turn that answered without citing because the earlier turns
cited nothing is the cost (`dev_t2_02` turn 3).

The map now travels with the answer, and `graph.replay_messages` restores the prior turn's
markers into the array's own numbering.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from services.agent.engine import deterministic
from services.agent.engine import graph
from services.agent.engine import model as model_gateway
from services.agent.engine.deterministic import run_command
from services.agent.postprocess import citations
from services.agent.routing import encounter, routing

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# The map itself
# ---------------------------------------------------------------------------


def test_a_turn_local_map_lands_on_the_numbers_its_evidence_carries():
    """`{"1": 4, "2": 5}` from a turn starting at 3 restores to `^[7]`, `^[8]`."""
    restored = citations.restore_global_markers(
        "discharged on warfarin ^[1], dietary instruction ^[2].", {"1": 4, "2": 5}, 3)

    assert restored == "discharged on warfarin ^[7], dietary instruction ^[8]."


def test_a_reference_to_an_earlier_turns_passage_survives_the_round_trip():
    """A display number measured from this turn's first unit can be negative, and that is
    the case a citation into an earlier turn produces. It has to come back as the position
    it actually addressed, not be dropped."""
    # Turn starts at 7; the answer cites its own first unit and turn 1's second passage.
    restored = citations.restore_global_markers(
        "the operation was uncomplicated ^[1] and the history states angina ^[2].",
        {"1": 1, "2": -5}, 7)

    assert restored == "the operation was uncomplicated ^[8] and the history states angina ^[2]."


def test_a_pmid_badge_is_left_exactly_as_written():
    """It names a record, not a position, so compaction never touched it and restoration
    must not either."""
    restored = citations.restore_global_markers(
        "the published rate was 4.1% ^[PMID: 35186616] and the ratio ^[1].",
        {"1": 2}, 0)

    assert "^[PMID: 35186616]" in restored
    assert restored.endswith("^[2].")


def test_a_display_number_the_map_does_not_hold_is_removed_not_kept():
    """A marker with no position behind it must not be printed: a pill that resolves to the
    wrong passage is the failure the whole mechanism exists to prevent."""
    restored = citations.restore_global_markers(
        "the course was uncomplicated ^[9] and the diagnosis was stated ^[1].",
        {"1": 4}, 0)

    assert "^[9]" not in restored
    assert "^[4]" in restored


def test_a_map_with_nothing_restorable_falls_back_to_deleting_them():
    """A turn stored before the map existed replays exactly as it did before it existed."""
    answer = "the course was uncomplicated ^[1] and the diagnosis was stated ^[2]."
    assert citations.restore_global_markers(answer, {}) == \
        citations.strip_numeric_markers(answer)
    assert citations.restore_global_markers(answer, None) == \
        citations.strip_numeric_markers(answer)


def test_an_answer_with_no_markers_is_returned_unchanged():
    answer = "The discharge summary does not document a diet."
    assert citations.restore_global_markers(answer, {"1": 3}, 5) == answer


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------


class _NoToolbox:
    """Every replayed call carries its payload, so nothing is re-resolved."""

    async def call(self, name, args):  # pragma: no cover - must not run
        raise AssertionError(f"replay re-resolved {name!r}; the payload was carried")


def _payload(*sections: str) -> dict:
    return {"returned": len(sections),
            "passages": [{"section": section, "text": f"{section} text", "id": section}
                         for section in sections]}


def _replay(turns):
    return asyncio.run(graph.replay_messages(_NoToolbox(), turns))


def _answers(messages) -> list[str]:
    """The replayed answer messages only. A replayed tool call is an `AIMessage` too, with
    no content — it is the assistant turn that asked for the call."""
    return [message.content for message in messages
            if isinstance(message, AIMessage) and str(message.content).strip()]


def test_a_replayed_answer_keeps_its_markers_in_the_conversations_numbering():
    """Turn 1's own numbering is the array's, because it starts at 0."""
    messages, calls, _ = _replay([{
        "question": "/ehr",
        "answer": "the operation was uncomplicated ^[1] and the diagnosis was stated ^[2].",
        "tool_calls": [{"name": "rag_search", "args": {"hadm_id": 1},
                        "payload": _payload("brief_hospital_course",
                                            "discharge_diagnosis")}],
        "citation_map": {"1": 1, "2": 2},
    }])

    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content.endswith("^[1] and the diagnosis was stated ^[2].")
    assert len(calls) == 1

def test_a_second_turn_is_restored_from_where_its_own_units_begin():
    """Turn 2's map is measured from its first unit, so the replay has to add the offset it
    is numbering that turn from — otherwise `^[1]` addresses turn 1's first passage."""
    messages, _, _ = _replay([
        {
            "question": "/ehr",
            "answer": "the operation was uncomplicated ^[1].",
            "tool_calls": [{"name": "rag_search", "args": {"hadm_id": 1},
                            "payload": _payload("brief_hospital_course")}],
            "citation_map": {"1": 1},
        },
        {
            "question": "What did the medications say?",
            "answer": "she was discharged on warfarin ^[1].",
            "tool_calls": [{"name": "rag_search", "args": {"hadm_id": 1},
                            "payload": _payload("discharge_medications")}],
            "citation_map": {"1": 1},
        },
    ])

    answers = _answers(messages)
    assert answers[0].endswith("^[1].")
    assert answers[1].endswith("^[2]."), answers[1]


def test_a_turn_with_no_map_is_replayed_with_its_numbers_deleted():
    """The behaviour before the field existed, kept as the fallback rather than assumed
    away: a caller that stores turns and not maps is a supported caller."""
    messages, _, _ = _replay([{
        "question": "/ehr",
        "answer": "the operation was uncomplicated ^[1].",
        "tool_calls": [{"name": "rag_search", "args": {"hadm_id": 1},
                        "payload": _payload("brief_hospital_course")}],
    }])

    assert messages[-1].content == "the operation was uncomplicated."


def test_the_turn_index_is_not_confused_with_the_offset():
    """`turn_offset` is captured before a turn's own calls are rendered. Taking it after
    would put turn 2's first marker one past its own first passage, which resolves to a
    unit the answer never addressed."""
    messages, _, _ = _replay([
        {
            "question": "/ehr",
            "answer": "course ^[1] and course again ^[2].",
            "tool_calls": [{"name": "rag_search", "args": {"hadm_id": 1},
                            "payload": _payload("brief_hospital_course",
                                                "history_of_present_illness")}],
            "citation_map": {"1": 1, "2": 2},
        },
        {
            "question": "And the diagnosis?",
            "answer": "the diagnosis was stated ^[1].",
            "tool_calls": [{"name": "rag_search", "args": {"hadm_id": 1},
                            "payload": _payload("discharge_diagnosis")}],
            "citation_map": {"1": 1},
        },
    ])

    answers = _answers(messages)
    assert answers[1].endswith("^[3]."), answers[1]


# ---------------------------------------------------------------------------
# The manifest on `/ehr`
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Records the prompt the formatting turn was given, then answers."""

    def __init__(self, reply: str = "**Presentation & diagnosis.** A short stay.") -> None:
        self.reply = reply
        self.prompts: list[list] = []

    def bind_tools(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the formatting turn was given tools")

    async def ainvoke(self, messages, config=None):
        self.prompts.append(list(messages))
        return AIMessage(content=self.reply)


class _PassageToolbox:
    async def call(self, name, args):
        return {"returned": 1, "passages": [{
            "section": "brief_hospital_course",
            "text": "Transferred to a rehabilitation facility on postoperative day four.",
            "id": "p1",
        }]}


MANIFEST = (
    "<admission_metadata>\n"
    "encounter: inpatient\n"
    "discharge_location: Home with home health\n"
    "</admission_metadata>"
)


def _run(hadm_id: int, *, admission_context: str = ""):
    used = _RecordingModel()
    encounter.set_current(encounter.INPATIENT)
    with patch.object(model_gateway, "build_llm", lambda name: used):
        state = asyncio.run(run_command(
            _PassageToolbox(), routing.command_for("summarize"), hadm_id,
            question="/ehr", model="test-model",
            admission_context=admission_context,
        ))
    return state, used


def test_the_ehr_formatting_turn_is_shown_the_manifest():
    """The block is the authority the passages are read against, and the instruction that
    says so (`EHR_FORMATTING`) can only be applied to a block the turn has been shown."""
    _, used = _run(90000015, admission_context=MANIFEST)

    assert used.prompts, "the formatting turn did not run"
    human = [message for message in used.prompts[0] if isinstance(message, HumanMessage)]
    assert human, "the formatting turn was given no question"
    assert "discharge_location: Home with home health" in human[0].content


def test_the_manifest_comes_after_the_evidence_it_qualifies():
    """Last, not first: it is the block that outranks the passages, and a reader of the
    prompt meets the evidence before the rule that decides between two of it."""
    _, used = _run(90000015, admission_context=MANIFEST)

    content = [m for m in used.prompts[0] if isinstance(m, HumanMessage)][0].content
    assert content.index("rehabilitation facility") < content.index("discharge_location")


def test_a_turn_with_no_manifest_adds_no_block():
    _, used = _run(90000015)

    content = [m for m in used.prompts[0] if isinstance(m, HumanMessage)][0].content
    assert "<admission_metadata>" not in content


def test_the_prompt_carries_the_rule_that_resolves_the_two_statements():
    """A block nobody is told to prefer is two facts in a window and no instruction."""
    instruction = deterministic.formatting_for("summarize")

    assert "discharge_location" in instruction
    assert "EARLIER encounter" in instruction


def test_the_block_is_rendered_by_the_same_function_the_free_text_path_uses():
    """One renderer, so the two paths cannot be shown different manifests."""
    encounter.set_current(encounter.INPATIENT)

    assert graph._context_block(MANIFEST, 90000015) == deterministic._context_block(
        MANIFEST, 90000015)
