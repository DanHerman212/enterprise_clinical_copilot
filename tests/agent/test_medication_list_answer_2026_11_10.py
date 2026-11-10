"""A medication list the chart returned may never be settled into a blank answer.

The live failure this file exists for (2026-11-10): the free-text question "What
medications is admission 90000015 going home on?" called `discharge_medications`, the
dedicated section came back with the drug list, and the model's reply arrived empty with
the provider's dangerous-content filter flagged. The settlement read an empty reply as a
decline, found nothing to decline against because the only retrieval was the medication
tool, and served no text at all — so the route reported a question the service had
answered as a refused one.

Two things went wrong and neither is the model's judgement:

  * the medication rules asked the model to verify the transfer list against both sections
    without saying which section to enumerate, and the reply it wrote while deciding was
    the one the filter took away (`engine/prompts.py`);
  * a turn holding a retrieved medication list had no fallback, because the only sentence
    written for "the reply did not come back" belongs to a literature turn
    (`postprocess/answer.py`).

Both are pinned here, because the live suite can only observe them one admission at a
time and this is the pair that has to hold for every admission.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from services.agent.engine import prompts
from services.agent.postprocess import answer as answer_text

ADMISSION = 90000015


def _state(tool_calls, *, text: str = "", finish_reason: str = "SAFETY") -> dict:
    """One settled turn, shaped the way the engine hands it to the settlement."""
    return {
        "messages": [
            AIMessage(content=text, response_metadata={"finish_reason": finish_reason})
        ],
        "tool_calls": tool_calls,
    }


def _medication_call(*, passages: int = 2, error: str | None = None) -> dict:
    if error is not None:
        response: dict = {"error": error}
    else:
        response = {
            "hadm_id": ADMISSION,
            "query": "discharge medications",
            "returned": passages,
            "dedicated_section_present": bool(passages),
            "passages": [
                {"id": f"chunk-{index}", "section": "discharge_medications"}
                for index in range(passages)
            ],
        }
    return {"name": "discharge_medications", "args": {"hadm_id": ADMISSION},
            "response": response}


def _served(state: dict) -> str:
    return str(state["messages"][-1].content)


def test_a_withheld_medication_summary_points_at_the_list_instead_of_vanishing():
    """The turn has the list. Serving nothing says the question was refused."""
    state = _state([_medication_call()])

    answer_text.reject_model_reply(state)

    served = _served(state)
    assert served.strip(), (
        "a turn holding the patient's discharge medication list settled to an empty "
        "answer; the reader is told the question was refused while the list is in hand"
    )
    assert "/meds" in served, (
        "the sentence has to name the command that renders the list, or it says that "
        "something exists without saying where to read it"
    )


def test_a_list_from_a_replayed_turn_counts_as_retrieved():
    """A follow-up replays the earlier call rather than re-running it."""
    state = _state([])
    state["replayed_tool_calls"] = [_medication_call()]

    answer_text.reject_model_reply(state)

    assert _served(state).strip()


def test_a_turn_with_no_medication_list_still_serves_no_text():
    """The fallback is for a list that exists, not for every empty turn.

    A search that returned nothing is a turn that may not assert anything, and the
    sentence written for it would name a list the record does not hold.
    """
    state = _state([{"name": "rag_search", "args": {"hadm_id": ADMISSION},
                     "response": {"hadm_id": ADMISSION, "passages": []}}])

    answer_text.reject_model_reply(state)

    assert _served(state) == ""


def test_a_failed_medication_call_is_not_a_list_to_point_at():
    state = _state([_medication_call(error="unknown_patient")])

    answer_text.reject_model_reply(state)

    assert _served(state) == ""


def test_a_medication_call_that_returned_no_passages_is_not_a_list_to_point_at():
    state = _state([_medication_call(passages=0)])

    answer_text.reject_model_reply(state)

    assert _served(state) == ""


def test_the_medications_spoke_says_which_section_to_enumerate():
    """The provenance rule and the enumeration rule are one instruction.

    The Sprint 2 sentences are unchanged: both sections are checked, and a drug that was
    not continued is never reported as a prescription. What was missing is the half that
    says what to write down — and without it the reply the model wrote while deciding was
    the one the provider's filter withheld, on a question the same rules had answered the
    day before.
    """
    text = " ".join(prompts.MEDS_SPOKE.split())

    assert "verify it against BOTH sections" in text
    assert "`discharge_instructions` for a continuation at home" in text
    assert "`brief_hospital_course` for a stop" in text
    assert "NOT continued" in text

    assert "enumerating the regimen `discharge_instructions` gives" in text, (
        "the medications spoke no longer says which section to enumerate, and a rule "
        "that only says what to check leaves the model deciding what to write"
    )
    assert "never to add one" in text, (
        "the second section may withhold a drug from the list; if it can also add one, "
        "the transfer list becomes a discharge regimen"
    )
