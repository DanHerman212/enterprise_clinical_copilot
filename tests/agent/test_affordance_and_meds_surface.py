"""Phase 01 follow-ups: the declared affordance, and the free-text medication surface.

Two decisions from the architect (2026-10-22), both asserted here:

1. A risk-driver bar click declares `intent="risk_driver"` instead of being recognised
   by its label, and that intent binds the predictor exactly as the wording does.
2. `discharge_medications` on the free-text path draws SOURCE CARDS, not a medication
   grid: the loop produces prose, not extraction rows, so `command_payload` is None and
   the grid had nothing verified to draw — it emitted "the medication extraction did not
   run for this answer" under a correct answer.
"""

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pytest  # noqa: E402

from services.agent.ingress import contracts
from services.agent.routing import routing  # noqa: E402
from services.agent.postprocess.guardrail import _passage_sections  # noqa: E402


# --- 1. the declared affordance ----------------------------------------------

def test_a_declared_driver_intent_binds_the_predictor():
    assert routing.wants_risk_driver("What is the oncology history for this patient?",
                                     "risk_driver") is True
    # The same words from a person are an ordinary history question.
    assert routing.wants_risk_driver("What is the oncology history for this patient?") is False
    # And the wording path still works without an intent.
    assert routing.wants_risk_driver("Why does her RDW drive readmission risk?") is True


def test_the_request_accepts_the_intent_and_carries_it():
    parsed = contracts.parse_agent_request(
        {"question": "What is the oncology history for this patient?",
         "hadm_id": 90000015, "intent": "risk_driver"},
        max_question_chars=4000)

    assert parsed["intent"] == "risk_driver"


def test_an_ordinary_request_carries_no_intent_key():
    """Absent, not null: "did a control send this?" is a key question, not a value one."""
    parsed = contracts.parse_agent_request(
        {"question": "What is the oncology history for this patient?",
         "hadm_id": 90000015},
        max_question_chars=4000)

    assert "intent" not in parsed


@pytest.mark.parametrize("intent", ["driver", "RISK_DRIVER", "", 7])
def test_an_unknown_intent_is_refused(intent):
    """A closed vocabulary. A surface that thinks it declared something must find out."""
    with pytest.raises(contracts.AgentRequestError):
        contracts.parse_agent_request(
            {"question": "q", "hadm_id": 90000015, "intent": intent},
            max_question_chars=4000)


# --- 2. the free-text medication surface -------------------------------------

MEDS_CALL = {
    "name": "discharge_medications",
    "args": {"hadm_id": 90000015},
    "response": {"hadm_id": 90000015, "returned": 1, "passages": [
        {"id": "n_discharge_medications_1",
         "text": "Discharge Medications: oxycodone SR 20 mg q.12h."},
    ]},
}


def test_a_free_text_medication_answer_resolves_citations_from_its_passages():
    """The passages are the evidence, so the source card can point at them."""
    from services.agent.presentation import a2ui

    merged = a2ui.rag_payload([MEDS_CALL])
    assert merged["passages"][0]["id"] == "n_discharge_medications_1"

    answer = "Oxycodone SR 20 mg is listed on discharge ^[1]."
    sources = a2ui.resolve_sources(
        a2ui.renumber_citations(answer), a2ui.citation_remap(answer), merged, ())

    assert len(sources) == 1
    assert "oxycodone" in sources[0]["text"].lower()


def test_the_guardrail_sees_the_medication_passages():
    """Per-dose verification reads the same passages the answer was written from."""
    passages, med_text = _passage_sections([MEDS_CALL])

    assert len(passages) == 1
    assert passages[0]["section"] == "discharge_medications"
    assert "oxycodone SR 20 mg" in med_text

def test_a_meds_command_answer_resolves_its_own_marker():
    """`/meds` writes `^[1]` in code; only now does the marker have a passage to point at.

    The deterministic narrative began citing the passage it reports on, but
    `rag_payload` did not read the medication tool's passages, so `sources` was empty
    and the marker addressed nothing. Including `discharge_medications` in the payload
    is what closes it — one passage, one source card, for the command and for free text.
    """
    from services.agent.presentation import a2ui

    presentation = a2ui.compose_presentation(
        "/meds",
        "1 discharge medication, as recorded in the discharge note: oxycodone SR ^[1].",
        [MEDS_CALL],
        {"medications": [{"medication": "oxycodone SR", "dosage": "20 mg",
                          "frequency": "q.12h."}],
         "dropped": [], "returned_passages": 1},
        "meds",
    )

    assert presentation["sources"], "the marker has no resolved source"
    assert "oxycodone" in presentation["sources"][0]["text"].lower()
