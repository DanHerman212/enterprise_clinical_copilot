"""Stage 5D: the admission suffix stops at the model boundary, and capitals are read.

Two failures and one instruction are pinned here.

The failure: turn 2 of `dev_t3_03` was served "This session is bound to one encounter
and cannot answer questions about other admission IDs" for a question about the patient
in front of it. The question the model read ended with the sentence `compose_question`
writes for the wire — `For admission 90000108.` — and the model had no way to know that
number was its own. The prompt has since told it not to guess; these tests hold the two
mechanical halves of that: the suffix never reaches the model, and any admission id a
caller types is read regardless of case.

The instruction: `CORE_HUB_PROMPT` must say that a foreign id is stopped before the model
sees it, because the alternative is the model drawing the boundary itself and getting it
wrong.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from services.agent.engine import graph, prompts
from services.agent.routing import routing

ADMISSION = 90000108


# --- reading an admission reference, whatever case it is typed in --------------------

@pytest.mark.parametrize("question", [
    "Summarise admission 90000015.",
    "Summarise Admission 90000015.",
    "Summarise ADMISSION 90000015.",
    "What did Patient 90000015 have?",
    "What did PATIENT 90000015 have?",
    "Read record 90000015",
    "Read RECORD 90000015",
    "Switch context to hadm_id 90000015",
    "Switch context to HADM_ID 90000015",
])
def test_a_capitalised_reference_is_read_as_an_admission(question):
    assert routing.requests_foreign_admission(question, ADMISSION) is True


@pytest.mark.parametrize("question", [
    "What is the risk for admission 90000108?",
    "What is the risk for Admission 90000108?",
    "Summarise admission 90000108.",
])
def test_the_bound_admission_is_never_foreign(question):
    assert routing.requests_foreign_admission(question, ADMISSION) is False


@pytest.mark.parametrize("question", [
    "Is a platelet count of 150000 concerning?",
    "Her white count was 110000 on admission.",
    "The cohort of 200000 patients was followed for a year.",
])
def test_a_bare_count_is_not_an_admission(question):
    """Six digits is the floor because everything shorter is a quantity, and a cue noun
    is required because a bare run of digits is a measurement."""
    assert routing.requests_foreign_admission(question, ADMISSION) is False


def test_a_recordless_turn_has_no_scope_to_violate():
    assert routing.requests_foreign_admission("admission 90000015", None) is False


# --- removing the sentence the service appended, and nothing else -------------------

def test_the_composed_suffix_is_removed_after_a_question_mark():
    assert routing.strip_admission_suffix(
        "Was his hemoglobin lower when he came in or when he left? "
        "For admission 90000108.", ADMISSION) == (
        "Was his hemoglobin lower when he came in or when he left?")


def test_the_composed_suffix_is_removed_after_a_full_stop():
    assert routing.strip_admission_suffix(
        "Summarise this admission. For admission 90000108.", ADMISSION) == (
        "Summarise this admission.")


def test_the_composed_suffix_is_removed_after_a_bare_command():
    """The case that has no punctuation to anchor on: `/ehr For admission 90000069.`"""
    assert routing.strip_admission_suffix(
        "/ehr For admission 90000069.", 90000069) == "/ehr"


def test_a_callers_own_sentence_is_left_alone():
    """"for admission 90000015?" is a clinician's question, not the composed tail."""
    for question in (
        "readmission risk score for admission 90000015?",
        "What is the readmission risk for admission 90000015?",
        "Is the risk for Admission 90000108 high?",
    ):
        assert routing.strip_admission_suffix(question, 90000015) == question


def test_a_question_that_never_had_the_suffix_comes_back_identical():
    question = "What medications is she going home on?"
    assert routing.strip_admission_suffix(question, ADMISSION) == question


def test_a_text_with_nothing_but_the_suffix_is_not_emptied():
    """A bare admission sentence is the one input that could reduce to an empty message,
    and an empty human message is not a question."""
    assert routing.strip_admission_suffix("For admission 90000108.", ADMISSION) == (
        "For admission 90000108.")


def test_a_foreign_suffix_is_kept_when_the_binding_is_known():
    """It is the caller's own text, and the request has already refused on it."""
    question = "Summarise this. For admission 90000015."
    assert routing.strip_admission_suffix(question, ADMISSION) == question


def test_an_empty_question_is_returned_unchanged():
    assert routing.strip_admission_suffix("", ADMISSION) == ""
    assert routing.strip_admission_suffix(None, ADMISSION) == ""


# --- the boundary itself -----------------------------------------------------------

class _CapturingGraph:
    """A graph that records the messages it was invoked with."""

    def __init__(self, reply: str = "The notes record a left heart catheterisation."):
        self.messages: list = []
        self._reply = reply

    async def ainvoke(self, payload, *_args, **_kwargs):
        self.messages = list(payload["messages"])
        return {
            "messages": [AIMessage(content=self._reply)],
            "tool_calls": [{"name": "rag_search_sections", "args": {},
                            "response": {"hadm_id": ADMISSION, "passages": []}}],
            "replayed_tool_calls": [],
            "evidence_state": None,
            # The real graph seeds the state with the question it was invoked with, and
            # the wire value is what every later reader of the state expects to find.
            "question": payload.get("question"),
            "replay_positions": 0,
        }


def test_the_model_is_not_shown_the_admission_sentence(monkeypatch):
    stub = _CapturingGraph()
    monkeypatch.setattr(graph, "build_graph", lambda *_a, **_k: stub)

    state = asyncio.run(graph.ask(
        object(), "Was his hemoglobin lower when he came in? For admission 90000108.",
        hadm_id=ADMISSION))

    humans = [m for m in stub.messages if isinstance(m, HumanMessage)]
    assert humans[-1].content == "Was his hemoglobin lower when he came in?"
    assert "For admission" not in " ".join(str(m.content) for m in stub.messages), (
        "the composed admission sentence reached the model's context window")
    # The wire value is untouched: routing, the settlement and the chip comparison read it.
    assert state["question"] == (
        "Was his hemoglobin lower when he came in? For admission 90000108.")


def test_a_replayed_turn_is_not_shown_the_admission_sentence():
    """The single-turn replay helper is the path a turn without tool calls takes."""
    messages = graph.replayed_turns([{
        "question": "/ehr For admission 90000069.",
        "answer": "A summary.",
    }])

    assert messages[0].content == "/ehr"


def test_replay_messages_strips_the_sentence_from_each_turn(monkeypatch):
    messages, _calls, _literature = asyncio.run(graph.replay_messages(
        object(),
        [{"question": "Summarise this admission. For admission 90000069.",
          "answer": "A summary.", "tool_calls": []}],
        hadm_id=90000069,
    ))

    humans = [m for m in messages if isinstance(m, HumanMessage)]
    assert humans[0].content == "Summarise this admission."


# --- the instruction the model reads ------------------------------------------------

def test_the_prompt_says_a_foreign_id_is_stopped_before_the_model():
    text = " ".join(prompts.CORE_HUB_PROMPT.split())

    assert "Foreign admission ids are stopped before they reach you" in text
    assert "never refuse it as another admission" in text
    assert "patient_scope_refused" in text


def test_the_prompt_no_longer_asks_the_model_to_judge_a_numeric_id():
    """The sentence that produced `dev_t3_03` turn 2's refusal is gone."""
    text = " ".join(prompts.CORE_HUB_PROMPT.split())

    assert "a question naming a different one is one this session does not answer" not in text
    assert "An admission the session did not bind is one you cannot reach" not in text


def test_the_hub_still_fits_its_cap():
    assert len(prompts.CORE_HUB_PROMPT.split()) <= 1500
