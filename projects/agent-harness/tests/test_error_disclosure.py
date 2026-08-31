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

from agent import server as srv  # noqa: E402

# The tools package re-exports the tool coroutines under the same names as the
# submodules, so import the MODULES explicitly for patching.
pr = importlib.import_module("mcp_server.tools.predict")
rs = importlib.import_module("mcp_server.tools.rag_search")


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


# --- /ask 502 (ECC-06) -------------------------------------------------------

def test_ask_failure_returns_generic_body_with_correlation_id():
    async def boom(box, question):
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


# --- tool_calls trim (ECC-08) --------------------------------------------------

def test_ask_response_trims_tool_calls_to_name_and_response():
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

    async def fake_ask(box, question):
        return state

    with patch.object(srv, "toolbox", _fake_toolbox), \
         patch.object(srv, "ask", fake_ask):
        resp = _client().post("/ask", json={"question": "diagnosis?"})

    assert resp.status_code == 200
    calls = resp.json()["tool_calls"]
    assert calls == [{"name": "rag_search", "response": state["tool_calls"][0]["response"]}]
    assert "args" not in calls[0]


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
