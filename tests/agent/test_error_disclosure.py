"""Offline tests for error-detail / info disclosure (Cluster G) — no cloud creds.

Covers:
  - ECC-06: /ask 502 bodies carry a stable code + correlation id, never raw
    exception text; /health no longer discloses project/region/MCP URL
  - ECC-08: tool_calls in the /ask response are trimmed to name + response
  - ECC-21: MCP tool errors return generic messages, not exception text
"""

import importlib
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from services.agent import http as srv  # noqa: E402

# The tools package re-exports the tool coroutines under the same names as the
# submodules, so import the MODULES explicitly for patching.
pr = importlib.import_module("services.mcp.tools.prediction")
rs = importlib.import_module("services.mcp.tools.retrieval")


@asynccontextmanager
async def _fake_toolbox():
    yield None


def _client():
    return TestClient(srv.app, raise_server_exceptions=False)


# --- /health (ECC-06) --------------------------------------------------------

def test_health_does_not_disclose_topology():
    body = _client().get("/health").json()
    assert body["status"] == "ok"
    for leaked in ("project", "location", "mcp_url"):
        assert leaked not in body


def test_health_reports_which_code_is_running():
    # The one field that is deliberately there: without it, "did my deploy land?"
    # is answered by reading a revision list and guessing.
    assert _client().get("/health").json()["code_revision"] == srv.chain.CODE_REVISION


# --- /ask 502 (ECC-06) -------------------------------------------------------

def test_ask_failure_returns_generic_body_with_correlation_id():
    async def boom(box, question, on_event=None, question_kind=None, turns=None):
        raise RuntimeError("https://secret-mcp-url/ask audience=projects/12345")

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", boom):
        resp = _client().post("/ask", json={"question": "risk for 90000009?"})

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "agent_failed"
    assert len(body["correlation_id"]) == 12
    # No exception text anywhere in the response.
    text = resp.text
    assert "secret-mcp-url" not in text
    assert "RuntimeError" not in text
    assert "audience" not in text


# --- request id in logs (Layer 2, Gap 2) --------------------------------------

def test_ask_logs_the_forwarded_cloud_trace_id(caplog):
    """Django forwards X-Cloud-Trace-Context; the agent's log line for the
    request must carry the trace id (the part before the slash) so the two
    services' entries pair up."""
    async def boom(box, question, on_event=None, question_kind=None, turns=None):
        raise RuntimeError("nope")

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", boom), \
         caplog.at_level("ERROR", logger="services.agent.http"):
        _client().post(
            "/ask", json={"question": "risk for 90000009?"},
            headers={"X-Cloud-Trace-Context": "105445aa7843bc8bf206b12000100000/1;o=1"},
        )

    assert "trace=105445aa7843bc8bf206b12000100000" in caplog.text
    assert "/1;o=1" not in caplog.text


# --- tool_calls trim (ECC-08) --------------------------------------------------

def test_ask_response_carries_each_tool_call_with_its_arguments():
    """A turn is stored so it can be replayed, and a replayed call needs its
    arguments: without them the replayed call misstates what was asked.

    This used to assert the opposite (name and response only, no args). The
    arguments are the caller's own question and admission, and the passages the
    call returned already cross this boundary, so carrying them exposes nothing
    new while making a faithful replay possible.
    """
    state = {
        "messages": [HumanMessage(content="q"),
                     AIMessage(content="The note describes pneumonia. ^[1]")],
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

    async def fake_ask(box, question, on_event=None, question_kind=None, turns=None):
        return state

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask):
        resp = _client().post("/ask", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    calls = resp.json()["tool_calls"]
    assert calls == [{
        "name": "rag_search",
        "args": {"hadm_id": 90000009, "query": "diagnosis"},
        "response": state["tool_calls"][0]["response"],
        "derivable": True,
    }]


# --- generic tool errors (ECC-21) ----------------------------------------------

def test_rag_embed_failure_is_generic():
    class _Raising:
        @property
        def models(self):
            raise RuntimeError("403 for table trim-icon.readmission.hybrid_notes")

    with patch.object(rs, "_embed_client", lambda: _Raising()):
        result = rs._search(90000009, "meds", 5)

    assert result["error"] == "embed_failed"
    assert "trim-icon" not in result["message"]
    assert "RuntimeError" not in result["message"]


def test_rag_index_failure_is_generic():
    class _RaisingEndpoint:
        def find_neighbors(self, **kwargs):
            raise RuntimeError("IAM audience projects/778397675435 rejected")

    class _Emb:
        values = [0.1]

    class _Models:
        def embed_content(self, **kwargs):
            class _Resp:
                embeddings = [_Emb()]
            return _Resp()

    class _EmbedClient:
        models = _Models()

    with patch.object(rs, "_embed_client", lambda: _EmbedClient()), \
         patch.object(rs, "_index_endpoint", lambda: _RaisingEndpoint()):
        result = rs._search(90000009, "meds", 5)

    assert result["error"] == "search_failed"
    assert "778397675435" not in result["message"]
    assert "RuntimeError" not in result["message"]


def test_predict_fetch_failure_is_generic():
    class _RaisingSource:
        def fetch(self, hadm_id):
            raise RuntimeError("BigQuery table trim-icon.readmission.hybrid_features 403")

    with patch.object(pr, "feature_order", lambda: ["f1"]), \
         patch.object(pr, "_source", lambda: _RaisingSource()):
        result = pr._predict(90000009)

    assert result["error"] == "feature_fetch_failed"
    assert "trim-icon" not in result["message"]
    assert "RuntimeError" not in result["message"]


def test_predict_prediction_failure_is_generic():
    class _Source:
        def fetch(self, hadm_id):
            return {"f1": 1.0}

    def raising_predict(vec):
        raise RuntimeError("endpoint projects/778397675435/locations/us-east1 down")

    with patch.object(pr, "feature_order", lambda: ["f1"]), \
         patch.object(pr, "_source", lambda: _Source()), \
         patch.object(pr, "predict_one", raising_predict):
        result = pr._predict(90000009)

    assert result["error"] == "prediction_failed"
    assert "778397675435" not in result["message"]
    assert "RuntimeError" not in result["message"]


# --- detail that was interpolated on purpose (gap 4) ----------------------------
#
# The paths above never put `str(exc)` in a message. These did: the detail was
# written into the sentence by hand, which is why they survived the ECC-21 pass.
# Each one now logs the detail and returns the sentence.

def test_incomplete_features_lists_no_columns(caplog):
    class _Source:
        def fetch(self, hadm_id):
            return {"f1": 1.0}  # f2 missing

    with patch.object(pr, "feature_order", lambda: ["f1", "f2"]), \
         patch.object(pr, "_source", lambda: _Source()), \
         caplog.at_level("WARNING"):
        result = pr._predict(90000009)

    assert result["error"] == "incomplete_features"
    assert "f2" not in result["message"]
    assert "f2" in caplog.text  # the operator still learns which column


def test_the_record_carries_the_tool_error_code(caplog):
    """F2: a refused call has to be countable, not only visible to a reader.

    The record exists so an isolation refusal can be alerted on; a code that
    reaches only the prompt and the browser cannot be.
    """
    state = {
        "messages": [HumanMessage(content="q"),
                     AIMessage(content="No notes were found.")],
        "tool_calls": [
            {"name": "rag_search", "args": {"hadm_id": 1, "query": "q"},
             "response": {"hadm_id": 1, "error": "isolation_violation",
                          "message": "A retrieved note did not belong to the "
                                     "requested admission; no passage was served."}},
            {"name": "predict_readmission", "args": {"hadm_id": 1},
             "response": {"hadm_id": 1, "probability": 0.2}},
        ],
    }

    fields = srv._response_fields(state)

    assert fields["tool_errors"] == ["isolation_violation"]


def test_the_record_stays_silent_when_every_call_succeeded():
    state = {
        "messages": [AIMessage(content="ok")],
        "tool_calls": [{"name": "rag_search", "args": {},
                        "response": {"returned": 0, "passages": []}}],
    }

    assert srv._response_fields(state)["tool_errors"] == []
