"""The routing boundary: a command is a command, and a question is a question.

Phase 01. Before this, `http._route_free_text` rewrote any free-text question that
contained "medications", "sodium" or "creatinine" into a command, and the command path
ignores `turns` by design — so a follow-up was answered with Turn 1's static caption.
Every turn-2 census failure on 2026-10-22 had that shape: the reader asked whether the
discharge list held an anticoagulant and was handed the medication count again.

The boundary is structural now, not lexical:

* an explicit command — a chip selection, or a leading `/slash` token that
  `parse_agent_request` resolves — runs the deterministic path;
* every other question, on any turn, reaches the reasoning loop with its transcript.

There is no word list left, so there is nothing to tune and nothing to hardcode.
"""

import asyncio
import json
import types

import pytest

from services.agent import contracts, http, routing

WIRE = {"/meds": "meds", "/labs": "labs", "/ehr": "summarize",
        "/pubmed": "literature"}


@pytest.mark.parametrize("slash,wire", sorted(WIRE.items()))
def test_only_a_slash_command_becomes_a_command(slash, wire):
    """The deterministic path is entered by naming it, never by resembling it.

    The command goes on the wire alone: text beside a command is a second intention,
    and the contract refuses that as `ambiguous_intent` (asserted in
    `test_eval_contract`). Only `/pubmed` takes a topic, and it is tested there.
    """
    body = {"question": f"{slash} heart failure readmission"
            if slash == "/pubmed" else slash, "hadm_id": 90000023}
    parsed = contracts.parse_agent_request(body, max_question_chars=4000)
    assert parsed["kind"] == wire


@pytest.mark.parametrize("question", [
    # Each of these was rewritten into /meds or /labs before Phase 01.
    "Show me the discharge medications and dosing for this admission",
    "Are any high-risk or anticoagulant medications listed on discharge?",
    "Which lab was most abnormal at discharge?",
    "What was her creatinine?",
    "What did her hemoglobin and sodium do during this admission?",
    "What is the oncology history for this patient?",
])
def test_a_question_is_never_rewritten_into_a_command(question):
    parsed = contracts.parse_agent_request({"question": question, "hadm_id": 90000023},
                                           max_question_chars=4000)
    assert parsed["kind"] is None


def test_the_request_seam_leaves_the_parsed_request_alone():
    """The seam is where the rewrite happened, so the seam is what is asserted."""
    async def body():
        return {"question": "Show me the discharge medications", "hadm_id": 90000023}

    parsed, error = asyncio.run(
        http._question_or_error(types.SimpleNamespace(json=body)))

    assert error is None
    assert parsed["kind"] is None
    assert "routed_to" not in parsed


def test_a_follow_up_keeps_its_transcript_at_the_seam():
    async def body():
        return {"question": "Are any high-risk or anticoagulant medications listed?",
                "hadm_id": 90000023,
                "turns": [{"question": "/meds",
                           "answer": "13 discharge medications, as recorded in the "
                                     "discharge note: oxycodone SR, Celebrex.",
                           "tool_calls": []}]}

    parsed, error = asyncio.run(
        http._question_or_error(types.SimpleNamespace(json=body)))

    assert error is None
    assert parsed["kind"] is None, "a follow-up is a question, not the command it follows"
    assert parsed["turns"], "the transcript has to reach the loop intact"


def test_no_router_remains():
    """A word list that no longer exists cannot be tuned, hardcoded or drift.

    What Phase 01 removed is the COMMAND router: no free-text question is turned into a
    command by a keyword table. The literature cue set came back on 2026-10-22 by
    architect directive, and it is not the same thing — it does not choose a command, it
    decides whether one optional TOOL is bound for the turn (`asks_for_literature`), and
    the deterministic path is never reachable through it.
    """
    assert not hasattr(http, "_route_free_text")
    assert not hasattr(routing, "classify_free_text")
    assert not hasattr(routing, "_FREE_TEXT_ROUTES")
    # Restored deliberately, and only as a tool gate.
    assert hasattr(routing, "asks_for_literature")
    assert "search_literature" not in routing.FREETEXT_TOOLS


def test_a_mistyped_command_is_refused_before_the_chain_runs():
    """`/ehrr` used to run an 18-step turn and raise a `Reconciled` badge."""
    response = http._unknown_command({"question": "/ehrr", "kind": None})

    assert response is not None
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"] == "unknown_command"
    # The message has to name the way out, or the caller is told only that they failed.
    for slash in ("/ehr", "/labs", "/meds", "/pubmed"):
        assert slash in body["message"]


def test_a_retired_command_is_refused_by_name():
    """`/risk` was a command until 2026-10-26. A caller who still types it is asking for
    something that no longer exists, and the honest answer names it rather than running a
    free-text turn over a line that starts with a slash."""
    response = http._unknown_command({"question": "/risk", "kind": None})

    assert response is not None
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"] == "unknown_command"
    assert "There is no /risk command" in body["message"]
    assert "/risk" not in [command.slash for command in routing.COMMANDS]


@pytest.mark.parametrize("parsed", [
    {"question": "/ehr", "kind": None},          # a real command, typed
    {"question": "/ehrr", "kind": "summarize"},  # a chip already resolved it
    {"question": "What is the readmission risk?", "kind": None},
    # A prose line that opened with a live command's name — "/risk of falls overnight" —
    # used to be exempt because the token named a command. No token does now, so there is
    # no such case left: a slash the caller typed is a command request, and the only
    # exemptions are a command that exists and a chip that already resolved one.
])
def test_nothing_else_is_refused_as_an_unknown_command(parsed):
    assert http._unknown_command(parsed) is None


def test_an_outpatient_record_gets_no_readmission_estimate():
    """A speech-therapy progress note has no admission to be readmitted after."""
    summarize = routing.command_for("summarize")

    outpatient_calls = routing.supplementary_arguments(summarize, 90000054)
    inpatient_calls = routing.supplementary_arguments(summarize, 90000015)

    assert all(name != "predict_readmission" for name, _ in outpatient_calls)
    assert any(name == "predict_readmission" for name, _ in inpatient_calls)


@pytest.mark.parametrize("question,expected", [
    # Answerable from the pre-flight manifest, so the graph must not force a note
    # search. This is the question that produced 17.29 days and then had it
    # replaced by "the discharge note does not document this" (2026-10-17).
    ("what is the patient length of stay?", True),
    ("What is the admission type for this patient?", True),
    ("Where was she discharged to?", True),
    ("What is her insurance?", True),
    ("How old is this patient?", True),
    ("How many prior admissions has she had?", True),
    ("How long was the stay?", True),
    # Not administrative: the notes are the evidence, so the demand still applies.
    ("What did the discharge note say about her anxiety?", False),
    ("Why did her knee pain increase during the stay?", False),
    # Token matching, not substrings: "dosage" is not "age", "closely" is not "los".
    ("What dosage of metoprolol was she on?", False),
    ("How closely did the teams work together?", False),
])
def test_asks_for_administrative_detail(question, expected):
    assert routing.asks_for_administrative_detail(question) is expected


# --- which turns may reach the risk model ------------------------------------

@pytest.mark.parametrize("question,expected", [
    # The question that failed on 2026-10-21: it asks why the model weighs a value,
    # which the chart cannot answer.
    ("What in this patient's chart explains creatinine as a readmission risk driver?",
     True),
    ("Why does her RDW drive readmission risk?", True),
    ("Explain how prior inpatient days contribute as a risk driver", True),
    # A score question is /risk's, and must not hand the free-text loop the predictor.
    ("What is her readmission risk?", False),
    ("Show me the risk score for this admission", False),
    # Why-questions that are not about risk stay on the chart tools.
    ("Why was she started on metoprolol?", False),
    ("What explains the drop in hemoglobin?", False),
])
def test_only_a_risk_driver_question_reaches_the_predictor(question, expected):
    """Binding the predictor everywhere would let free text produce a score; binding it
    nowhere leaves a driver question unanswerable. The predicate is the boundary."""
    assert routing.asks_why_risk_driver(question) is expected


# --- a refusal is a turn, and a turn is traceable -----------------------------

def _fake_span(monkeypatch, trace_id="aaaa1111"):
    """Make `observability.observation` yield a span with a known trace id."""
    import contextlib
    import types

    from services.agent import observability

    seen: dict = {}

    @contextlib.contextmanager
    def fake_observation(name, *, as_type="span", trace_metadata=None):
        seen["name"] = name
        seen["metadata"] = trace_metadata or {}
        yield types.SimpleNamespace(trace_id=trace_id)

    monkeypatch.setattr(observability, "observation", fake_observation)
    return seen


def test_an_unknown_command_refusal_carries_a_trace_id(monkeypatch):
    """`/`, `/labb`, `/medss`, `/ehrr` on 2026-10-21 were right and untraceable.

    The four probes were answered correctly (400, no tool calls) and reported no trace
    id, so the eval had nowhere to put their scores and called them unattributed. A
    refusal is a decision, and a decision is worth a trace.
    """
    import json

    seen = _fake_span(monkeypatch)

    response = http._unknown_command({"question": "/labb", "kind": None,
                                      "run_tag": "pilot-run", "hadm_id": 90000023})

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"] == "unknown_command"
    assert body["langfuse_trace_id"] == "aaaa1111"

    # Tagged like an answered turn, so the run filter finds it.
    assert seen["name"] == "command.unknown"
    assert "run:pilot-run" in seen["metadata"]["langfuse_tags"]
    assert seen["metadata"]["langfuse_session_id"] == "pilot-run:hadm_90000023"


def test_a_known_command_is_still_not_refused(monkeypatch):
    _fake_span(monkeypatch)
    assert http._unknown_command({"question": "/labs", "kind": None}) is None
    assert http._unknown_command({"question": "a question", "kind": None}) is None


def test_an_unserved_admission_refusal_carries_a_trace_id(monkeypatch):
    """The sibling exit: composed like an answer, but opened outside any span."""
    import asyncio

    _fake_span(monkeypatch)

    async def fake_compose(*_args, **_kwargs):
        return {"error": "admission_not_found"}

    monkeypatch.setattr(http, "_compose_success", fake_compose)

    payload = asyncio.run(http._unserved_admission(
        {"question": "q", "hadm_id": 90009999, "run_tag": "pilot-run"}, "trace-1"))

    assert payload is not None
    assert payload["langfuse_trace_id"] == "aaaa1111"
