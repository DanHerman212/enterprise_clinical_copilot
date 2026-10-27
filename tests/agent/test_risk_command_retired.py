"""`/risk` is retired: the estimate belongs to the `/ehr` turn, not to a command of its
own.

The charter item (2026-10-26): the `/risk` chip exists but produces nothing when a patient
is submitted with a blank composer, and a reader who lands on that should learn that the
command does not exist rather than get a spinner and no answer. The score itself is not
going anywhere — `/ehr` runs `predict_readmission` as a supplementary call beside the
summary that explains it, and a driver bar inside that chart still declares
`intent="risk_driver"` to ask why a factor counts.

So there are two things to hold at once, and these tests hold both: the command
vocabulary is the four commands, and every path that produced or drew the estimate
without it still works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent.engine import graph
from services.agent import http as srv
from services.agent.routing import routing as questions
from services.agent.routing import routing  # noqa: E402


def _ask(body: dict):
    return TestClient(srv.app, raise_server_exceptions=False).post("/ask", json=body)


# ------------------------------------------------------------------- the vocabulary


def test_the_command_vocabulary_is_the_four_commands():
    assert [command.slash for command in routing.COMMANDS] == [
        "/labs", "/meds", "/ehr", "/pubmed"]


def test_slash_risk_names_no_command():
    assert routing.command_for_slash("/risk") is None


def test_a_typed_slash_risk_is_refused_and_lists_the_commands_that_exist():
    response = _ask({"question": "/risk", "hadm_id": 90000015})

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "unknown_command"
    # The message is the reader's: it names what they typed and what does exist.
    assert "There is no /risk command" in body["message"]
    for command in ("/ehr", "/labs", "/meds", "/pubmed"):
        assert command in body["message"]


def test_a_chip_named_risk_is_refused_before_anything_runs():
    response = _ask({"chip": "risk", "hadm_id": 90000015})

    assert response.status_code == 400
    assert response.json()["error"] == "unknown_chip"


def test_a_patient_with_no_question_is_refused():
    """The default that used to answer this was the risk command's own wording. With the
    command gone there is nothing to ask, and the honest answer is the refusal."""
    response = _ask({"hadm_id": 90000015})

    assert response.status_code == 400
    assert "Send a 'question'" in response.json()["message"]


def test_a_patient_with_no_question_composes_nothing():
    """The default sentence that used to answer this WAS the risk command's question.
    With the command gone there is no default to compose, and the module says so."""
    assert not hasattr(questions, "DEFAULT_TEMPLATE")

    with pytest.raises(ValueError):
        questions.own_words(question=None, hadm_id=90000015)


# ------------------------------------------------------- what the removal must not touch


def test_the_ehr_turn_still_requests_the_estimate():
    """The score is drawn by `/ehr`, beside the narrative that explains it — which is why
    the command of its own was retired rather than the capability."""
    ehr = routing.command_for("summarize")

    assert "predict_readmission" in {spec.tool for spec in ehr.supplementary}


def test_a_driver_question_still_binds_the_predictor():
    """A bar clicked inside `/ehr`'s risk chart declares `intent="risk_driver"`, and that
    declaration is what keeps the predictor in front of the model on a typed why-question."""
    assert "predict_readmission" in graph._freetext_allowed(
        routing.wants_risk_driver("Why does oncology history matter here?",
                                  "risk_driver"),
        wants_literature=False)
