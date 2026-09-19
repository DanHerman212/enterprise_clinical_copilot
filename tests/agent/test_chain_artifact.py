"""The chain's identity, and the record of what each execution used (Gap 2).

Offline: the chain is stubbed, so these prove the *record* rather than the
answer. The two things worth guarding are that the model cannot be changed by
the environment, and that an execution which failed still leaves a record —
a log that only covers the answers that worked cannot explain the ones that
did not.
"""

import ast
import importlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import chain, http as srv, stages  # noqa: E402


@asynccontextmanager
async def _fake_toolbox():
    yield None


# Files allowed to build a client for the pinned model somewhere other than the pinned
# endpoint. One entry, and it is a diagnostic whose entire purpose is asking where a model
# answers — so a location it can vary is the feature, not a mistake. The test checks that
# its default is still the pinned endpoint, so the exemption cannot quietly go stale.
NON_MODEL_LOCATION_EXEMPT = {"scripts/agent/check_gemini.py"}


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


def _state():
    """A completed chain that survives the real guardrails and composition."""
    return {
        "messages": [
            HumanMessage(content="What do the notes say about the diagnosis?"),
            AIMessage(content="The note describes pneumonia. ^[1]"),
        ],
        "tool_calls": [{
            "name": "rag_search",
            "args": {"hadm_id": 90000009, "query": "diagnosis"},
            "response": {"hadm_id": 90000009, "returned": 1, "passages": [
                {"id": "MT-1-DS_discharge_diagnosis_1",
                 "section": "discharge_diagnosis",
                 "text": "DISCHARGE DIAGNOSES: 1. Pneumonia."},
            ]},
        }],
    }


def _records(caplog):
    """Every execution record in the captured log, parsed."""
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("{"):
            continue
        try:
            parsed = json.loads(message)
        except ValueError:
            continue
        if parsed.get("event") == "agent_execution":
            out.append(parsed)
    return out


# --- the identity -----------------------------------------------------------

def test_the_model_is_pinned_and_ignores_the_environment(monkeypatch):
    """The pin is the whole point: an environment default lets a deploy change
    the model with no commit anywhere, which is what makes an answer
    unreproducible. Reloaded with the variable set, the value must not move."""
    import services.mcp.config as config

    monkeypatch.setenv("GEMINI_MODEL", "gemini-imaginary-1")
    try:
        importlib.reload(config)
        assert config.GEMINI_MODEL != "gemini-imaginary-1"
        assert config.GEMINI_MODEL == "gemini-3.1-flash-lite"
    finally:
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        importlib.reload(config)


def test_the_environment_is_really_consulted(monkeypatch):
    """The tests either side of this one are only worth anything if reload re-reads env.

    A reload that quietly returned the same module object would make every "ignores the
    environment" assertion pass while testing nothing. So the mechanism is pinned down
    here, using a setting that is still meant to be optional. If every setting below ever
    becomes pinned, delete this test rather than weakening it.
    """
    import services.mcp.config as config

    monkeypatch.setenv("RAG_TOP_K", "9")
    try:
        importlib.reload(config)
        assert config.DEFAULT_TOP_K == 9
    finally:
        monkeypatch.delenv("RAG_TOP_K", raising=False)
        importlib.reload(config)


def test_the_embedding_space_is_not_a_setting(monkeypatch):
    """Gap 5, and the same argument as the model pin above.

    The embedding parameters decide which vector space the index was built in. Left
    settable, a deploy could point the query embedder at a different model and get
    plausible neighbours from the wrong space — no error, no commit, no artifact to
    review. They are no longer settings at all, so the variables do nothing.
    """
    import services.mcp.config as config
    from services.mcp.retrieval import embed

    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-imaginary-1")
    monkeypatch.setenv("EMBEDDING_DIM", "999")
    try:
        importlib.reload(config)
        assert not hasattr(config, "EMBEDDING_MODEL")
        assert not hasattr(config, "EMBEDDING_DIM")
        assert embed.EMBEDDING_MODEL == "gemini-embedding-001"
        assert embed.OUTPUT_DIMENSIONALITY == 768
    finally:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_DIM", raising=False)
        importlib.reload(config)


def test_the_region_is_pinned_and_ignores_the_environment(monkeypatch):
    """The other half of gap 7.

    `LOCATION` names where the vector index, the prediction endpoint, the feature bundle
    and the BigQuery dataset live, so it is a fact about the world rather than a setting.
    While it was an environment read, a deploy could point the app at a region its
    resources were not in — which fails at request time with a 404, not at deploy time —
    and the region carries a residency question that belongs in a commit.
    """
    import services.mcp.config as config

    monkeypatch.setenv("LOCATION", "europe-west4")
    try:
        importlib.reload(config)
        assert config.LOCATION != "europe-west4"
        assert config.LOCATION == "us-east1"
    finally:
        monkeypatch.delenv("LOCATION", raising=False)
        importlib.reload(config)


def test_the_output_allowance_is_pinned_and_ignores_the_environment(monkeypatch):
    """Thinking and the answer share this allowance, and running out of it is silent.

    A deploy that lowered it would get an empty answer with `finish_reason=MAX_TOKENS`
    and no exception — a wrong-looking result rather than a failed deployment — which is
    why it is a reviewed change and not a deploy-time input.
    """
    import services.mcp.config as config

    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "64")
    try:
        importlib.reload(config)
        assert config.GEMINI_MAX_OUTPUT_TOKENS != 64
        assert config.GEMINI_MAX_OUTPUT_TOKENS >= 2048
    finally:
        monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)
        importlib.reload(config)


def _model_clients_outside_the_model_region(repo):
    """Files that generate with the pinned model but point a client somewhere else.

    Read from the syntax tree rather than the text, so a comment that mentions the old
    region cannot produce a false alarm and a value passed through a variable cannot hide
    from it. Files that never mention the pinned model are skipped, which is what keeps
    the embedding clients — legitimately in the project's region — out of the result.
    """
    offenders = []
    for folder in ("services", "evaluation", "scripts"):
        for path in (repo / folder).rglob("*.py"):
            source = path.read_text()
            if "GEMINI_MODEL" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                passed = next((k.value for k in node.keywords if k.arg == "location"), None)
                if passed is None:
                    continue
                if isinstance(passed, ast.Name) and passed.id == "GEMINI_LOCATION":
                    continue
                offenders.append(f"{path.relative_to(repo)}:{node.lineno}")
    return offenders


def test_no_call_site_points_the_model_at_the_project_region():
    """Two call sites were left behind by the swap, and both failed only when called.

    `evaluation/agent/judge.py` and `scripts/agent/check_gemini.py` open their own clients
    for the pinned model, and both took their location from the project's region. That was
    correct while the model and the project happened to share a region and wrong the moment
    the swap divided them: the model is not served in us-east1 at all. The suite did not
    notice, because neither file is reached by a test that calls the model, and the second
    of them was worse than the first — a diagnostic whose whole job is to answer "is the
    model reachable?" would have answered 404 by default.
    """
    repo = Path(__file__).resolve().parents[2]
    for exempt in NON_MODEL_LOCATION_EXEMPT:
        # The exemption is only valid while the file's own default is the pinned endpoint,
        # so it is checked rather than trusted.
        assert "GEMINI_LOCATION" in (repo / exempt).read_text(), (
            f"{exempt} is exempt from this rule only because its default is the pinned "
            "endpoint. It no longer mentions GEMINI_LOCATION, so the exemption is stale."
        )
    offenders = [
        found
        for found in _model_clients_outside_the_model_region(repo)
        if found.split(":")[0] not in NON_MODEL_LOCATION_EXEMPT
    ]
    assert offenders == [], (
        "these files build a client for the pinned model in a location that does not "
        f"serve it — use GEMINI_LOCATION: {offenders}"
    )


def test_the_pinned_model_is_the_one_on_record():
    """The pin and its written reason move together.

    A test rather than a comment because a comment does not stop anyone: changing
    the model now means editing the record, and the record is where the reason and
    the evidence live.
    """
    import services.mcp.config as config

    assert config.MODEL_CHOICE["model"] == config.GEMINI_MODEL
    assert config.MODEL_CHOICE["decided"]
    assert config.MODEL_CHOICE["tier"]


def test_the_record_names_what_a_comparison_would_be_against():
    # The cheaper tier is named so the next person does not have to rediscover it, and
    # the evidence slot exists so that filling it in is a deliberate edit.
    import services.mcp.config as config

    assert config.MODEL_CHOICE["cheaper_alternative"] != config.GEMINI_MODEL
    # Empty is allowed — nothing below this pin has been checked — but then the record
    # still has to name what the pin moved from, or the comparison it owes has no other
    # side to be against.
    assert config.MODEL_CHOICE["previous"] != config.GEMINI_MODEL
    assert "evidence" in config.MODEL_CHOICE


def test_the_retirement_is_recorded_with_the_models_that_replace_it():
    # An empty tuple here is a finding rather than an omission: Google records the
    # retirement date for this model and names no successor yet, and the record says
    # so. That is a different state from not having looked.
    import services.mcp.config as config

    assert config.MODEL_CHOICE["retires"]
    replacements = config.MODEL_CHOICE["replacements"]
    assert isinstance(replacements, tuple)
    assert config.GEMINI_MODEL not in replacements


def test_a_pin_change_cannot_stay_uncompared_until_the_next_retirement():
    """The retirement guard forces the swap, not the comparison.

    The last swap happened under the deadline with no comparison behind it, and the
    test that woke us up would have been satisfied by swapping the model and nothing
    else — which is how the same thing happens twice. So the comparison falls due
    earlier than the swap does, and `MODEL_CHOICE['evidence']['comparison']` is what
    settles it. If this is the test that woke you up, the plan is in the layer 4
    document, sections 6.6 and 6.10.
    """
    from datetime import date, timedelta

    import services.mcp.config as config

    retires = date.fromisoformat(config.MODEL_CHOICE["retires"])
    due = retires - timedelta(days=config.COMPARISON_DUE_DAYS)
    recorded = config.MODEL_CHOICE["evidence"]["comparison"]
    remaining = (due - date.today()).days
    when = f"falls due on {due}" if remaining >= 0 else f"fell due on {due}, {abs(remaining)} days ago"
    assert recorded or remaining > 0, (
        f"The pin {config.GEMINI_MODEL} has no comparison recorded and the evidence "
        f"{when}. Run the comparison and record it in "
        f"MODEL_CHOICE['evidence']['comparison']."
    )


def test_the_comparison_falls_due_before_the_swap_does():
    # Otherwise the two guards collapse into one and the comparison can always be
    # deferred to the same last-minute position.
    import services.mcp.config as config

    assert config.COMPARISON_DUE_DAYS > config.MIGRATION_LEAD_DAYS


def test_the_pinned_model_is_not_near_its_retirement():
    """The date is an input the suite enforces, not a note in a comment.

    A retired model does not warn: the calls simply stop working. So this fails
    `MIGRATION_LEAD_DAYS` before the date, which is when there is still time to run
    the comparison the migration needs and to schedule the swap. If this is the test
    that woke you up, the plan is in the layer 4 document, section 6.6.
    """
    from datetime import date, timedelta

    import services.mcp.config as config

    retires = date.fromisoformat(config.MODEL_CHOICE["retires"])
    last_safe_day = retires - timedelta(days=config.MIGRATION_LEAD_DAYS)
    assert date.today() < last_safe_day, (
        f"{config.GEMINI_MODEL} retires on {retires} "
        f"({(retires - date.today()).days} days away). The migration is due: swap "
        f"the model, run the comparison, and record the evidence in MODEL_CHOICE."
    )


def test_the_lead_time_cannot_be_set_to_nothing():
    # A guard that can be switched off quietly is not a guard.
    import services.mcp.config as config

    assert config.MIGRATION_LEAD_DAYS >= 7


def test_the_chain_exposes_its_model_and_identity():
    assert chain.MODEL_ID == chain.MODEL_ID.strip()
    assert chain.CODE_REVISION.strip()
    # The record names the fields; the shape is asserted below against a real
    # record rather than against this tuple alone.
    assert "code_revision" in chain.RECORD_FIELDS
    assert "model" in chain.RECORD_FIELDS


def test_the_identity_comes_from_the_deployment():
    # The deploy's own value wins, and Cloud Run's revision name is the fallback.
    assert chain.resolve_code_revision({"CODE_REVISION": "abc1234"}) == "abc1234"
    assert chain.resolve_code_revision(
        {"K_REVISION": "agent-00032-7cv"}
    ) == "agent-00032-7cv"
    assert chain.resolve_code_revision(
        {"CODE_REVISION": "abc1234", "K_REVISION": "agent-00032-7cv"}
    ) == "abc1234"


def test_a_run_that_is_not_a_deployment_says_so():
    assert chain.resolve_code_revision({}) == "local"
    assert chain.resolve_code_revision(
        {"CODE_REVISION": "   ", "K_REVISION": ""}
    ) == "local"


def test_an_unexpanded_substitution_is_not_an_identity():
    # A $COMMIT_SHA that a build never substituted must not be reported as the
    # revision this answer came from — that is the failure this replaced.
    assert chain.resolve_code_revision({"CODE_REVISION": "$COMMIT_SHA"}) == "local"
    assert chain.resolve_code_revision(
        {"CODE_REVISION": "$COMMIT_SHA", "K_REVISION": "agent-00032-7cv"}
    ) == "agent-00032-7cv"


# --- the record -------------------------------------------------------------

def test_the_record_carries_the_identity_and_the_steps(caplog):
    with caplog.at_level("INFO", logger="services.agent.chain"):
        record = chain.record_execution(
            trace="abc123",
            question="What do the notes say?",
            stages=[
                {"stage": "planning", "tool": None, "ms": 12},
                {"stage": "tool", "tool": "rag_search", "ms": 480},
            ],
            duration_ms=910,
            outcome="ok",
            tool_calls=["rag_search"],
            guardrail_flags=["med_dose_mismatch:5 mg", "risk_number_unsupported:0.14"],
        )

    assert record["code_revision"] == chain.CODE_REVISION
    assert record["model"] == chain.MODEL_ID
    assert record["trace"] == "abc123"
    assert record["outcome"] == "ok"
    assert record["question_chars"] == len("What do the notes say?")
    assert record["duration_ms"] == 910
    assert [stage["stage"] for stage in record["stages"]] == ["planning", "tool"]
    assert record["tool_calls"] == ["rag_search"]
    # The names, so the line says which guard acted and not only that one did.
    assert record["guardrail_flags"] == [
        "med_dose_mismatch:5 mg", "risk_number_unsupported:0.14",
    ]
    # The declared field list is the record's shape, not an approximation of it.
    # A field written to the record and missing from the list is how the list
    # went stale twice: nothing failed, and a reader could not tell which of the
    # two was wrong.
    assert set(record) == set(chain.RECORD_FIELDS)
    assert "error" not in record
    # One line, and it is JSON: queryable by whatever storage layer 10 picks.
    assert len(caplog.records) == 1
    assert json.loads(caplog.records[0].getMessage()) == record


def test_a_failed_execution_still_records_what_it_had(caplog):
    """A record that only exists for answers that worked cannot explain the
    ones that did not."""
    with caplog.at_level("INFO", logger="services.agent.chain"):
        record = chain.record_execution(
            trace="-",
            question="Why?",
            stages=[{"stage": "planning", "tool": None, "ms": 5}],
            duration_ms=100,
            outcome="timeout",
            error="timeout",
        )

    assert record["outcome"] == "timeout"
    assert record["error"] == "timeout"
    assert record["tool_calls"] == []
    assert record["guardrail_flags"] == []
    # `error` is the one field a failure adds, and it is the only difference.
    assert set(record) == set(chain.RECORD_FIELDS) | {"error"}


def test_the_record_carries_no_question_or_answer_text(caplog):
    """The text is patient-derived and a log store is not the repository's
    privacy regime. Length and shape are recorded instead; if the text is ever
    needed for an incident review that is a deliberate decision with retention,
    not a side effect of adding a log line."""
    secret = "What medications was admission 90000009 discharged on?"

    with caplog.at_level("INFO", logger="services.agent.chain"):
        chain.record_execution(
            trace="-", question=secret, stages=[], duration_ms=1, outcome="ok",
        )

    line = caplog.records[0].getMessage()
    assert secret not in line
    assert "90000009" not in line
    assert json.loads(line)["question_chars"] == len(secret)


# --- one record per execution, both routes ----------------------------------

def test_a_blocking_execution_records_once(caplog):
    async def fake_ask(box, question, on_event=None, question_kind=None, turns=None):
        on_event(stages.planning_event())
        on_event(stages.tool_event("rag_search"))
        return _state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "ok"
    # The stages the model ran, with timings, not just a count.
    assert [stage["stage"] for stage in records[0]["stages"]] == ["planning", "tool"]
    assert records[0]["stages"][1]["tool"] == "rag_search"
    assert records[0]["tool_calls"] == ["rag_search"]


def test_a_failed_blocking_execution_still_records_once(caplog):
    async def boom(box, question, on_event=None, question_kind=None, turns=None):
        raise RuntimeError("upstream died")

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", boom), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask", json={"question": "diagnosis?"})

    assert resp.status_code == 502
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "error"
    assert records[0]["error"] == "agent_failed"


def test_a_streamed_execution_records_once(caplog):
    async def fake_ask(box, question, on_event=None, question_kind=None, turns=None):
        on_event(stages.planning_event())
        on_event(stages.tool_event("rag_search"))
        return _state()

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    assert "event: answer" in resp.text
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "ok"


def test_a_failed_streamed_execution_still_records_once(caplog):
    async def boom(box, question, on_event=None, question_kind=None, turns=None):
        on_event(stages.tool_event("rag_search"))
        raise RuntimeError("upstream died")

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", boom), \
         caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask/stream", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    records = _records(caplog)
    assert len(records) == 1
    assert records[0]["outcome"] == "error"


def test_a_rejected_request_is_not_an_execution(caplog):
    """Nothing ran, so there is nothing to record: the chain never started."""
    with caplog.at_level("INFO", logger="services.agent.chain"):
        resp = _client().post("/ask", json={"chip": "bogus", "hadm_id": 7})

    assert resp.status_code == 400
    assert _records(caplog) == []
