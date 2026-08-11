"""Offline tests for rag_search — no cloud credentials required.

The tool's cloud touchpoints (_embed_client, _index_endpoint, _fetch_texts)
are stubbed so we test the *logic*: the hadm_id restrict is always applied,
the response shape is right, empty is a real answer, and failures are
structured errors rather than stack traces.
"""

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `mcp_server.tools.rag_search` is shadowed by the function of the same name
# (tools/__init__ re-exports it), so import the module explicitly.
rs = importlib.import_module("mcp_server.tools.rag_search")

HADM_A = 20724182


class _FakeNeighbor:
    def __init__(self, id_: str, distance: float):
        self.id = id_
        self.distance = distance


class _FakeEndpoint:
    """Records the last query so the test can assert the restrict was applied."""

    def __init__(self, neighbors: list[_FakeNeighbor]):
        self._neighbors = neighbors
        self.last_deployed_id = None
        self.last_filter = None
        self.last_queries = None

    def find_neighbors(self, *, deployed_index_id, queries, num_neighbors, filter=None):
        self.last_deployed_id = deployed_index_id
        self.last_queries = queries
        self.last_filter = filter
        return [self._neighbors]


class _FakeEmbeddings:
    """Stub for genai.Client().models.embed_content(...)."""

    def __init__(self):
        self.last_contents = None
        self.last_config = None

    def embed_content(self, *, model, contents, config):
        self.last_contents = contents
        self.last_config = config

        class _Emb:
            values = [0.1] * rs.EMBEDDING_DIM

        class _Resp:
            embeddings = [_Emb()]

        return _Resp()


class _FakeEmbedClient:
    def __init__(self):
        self.models = _FakeEmbeddings()


def _run(coro):
    return asyncio.run(coro)


def _run_search(endpoint: _FakeEndpoint, note_texts: dict | None = None, **kwargs):
    """Run rag_search with all cloud touchpoints faked."""
    embed = _FakeEmbedClient()
    note_texts = note_texts or {}

    def fake_fetch(note_ids):
        # Only return texts that were explicitly provided — a missing note_id
        # stays missing so the missing_text error path can be exercised.
        return {nid: note_texts[nid] for nid in note_ids if nid in note_texts}

    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: embed), \
         patch.object(rs, "_fetch_texts", fake_fetch):
        return _run(rs.rag_search(**kwargs))


def test_restrict_always_applied():
    """The hadm_id restrict must be passed to the index, server-side."""
    endpoint = _FakeEndpoint([
        _FakeNeighbor("13479418-DS-24_brief_hospital_course_2", 0.27),
    ])
    result = _run_search(
        endpoint,
        note_texts={"13479418-DS-24": "real note text"},
        hadm_id=HADM_A, query="sepsis treatment", top_k=5,
    )

    assert result["hadm_id"] == HADM_A
    assert result["returned"] == 1
    assert result["passages"][0]["section"] == "brief_hospital_course"
    assert result["passages"][0]["text"] == "real note text"
    # The restrict must carry THIS patient's hadm_id.
    assert endpoint.last_filter is not None
    assert len(endpoint.last_filter) == 1
    assert endpoint.last_filter[0].name == rs.RESTRICT_NAMESPACE
    assert endpoint.last_filter[0].allow_tokens == [str(HADM_A)]


def test_empty_is_a_real_answer():
    """No neighbors -> empty passages, not an error."""
    endpoint = _FakeEndpoint([])
    result = _run_search(endpoint, hadm_id=HADM_A, query="nothing should match")
    assert result == {"hadm_id": HADM_A, "query": "nothing should match",
                      "returned": 0, "passages": []}


def test_bad_hadm_id_is_structured_error():
    """Non-positive hadm_id is a structured error, not a crash."""
    result = _run_search(_FakeEndpoint([]), hadm_id=-5, query="sepsis")
    assert result["error"] == "bad_request"
    assert "hadm_id" in result["message"]


def test_empty_query_is_structured_error():
    result = _run_search(_FakeEndpoint([]), hadm_id=HADM_A, query="   ")
    assert result["error"] == "bad_request"


def test_top_k_out_of_range_is_structured_error():
    result = _run_search(_FakeEndpoint([]), hadm_id=HADM_A, query="sepsis", top_k=50)
    assert result["error"] == "bad_request"


def test_missing_bigquery_text_errors_not_drops():
    """A returned ID missing from BigQuery must error, not silently drop."""
    endpoint = _FakeEndpoint([
        _FakeNeighbor("99999999-DS-1_brief_hospital_course_1", 0.25),
    ])
    result = _run_search(endpoint, note_texts={}, hadm_id=HADM_A, query="sepsis")
    assert result["error"] == "missing_text"
    assert "99999999-DS-1" in result["message"]


def test_section_parsing_recovers_multiword_section():
    """brief_hospital_course has underscores; parsing must not split it."""
    note_id, section = rs._parse_datapoint_id(
        "13479418-DS-24_brief_hospital_course_2")
    assert note_id == "13479418-DS-24"
    assert section == "brief_hospital_course"


def test_section_parsing_unknown_returns_none():
    note_id, section = rs._parse_datapoint_id("12345-DS-9_unknown_section_1")
    assert note_id is None
    assert section is None


def test_index_endpoint_reconstructs_with_full_name():
    """find_neighbors crashes on .list() lazy objects (no _public_match_client);
    _index_endpoint must re-construct by the full resource name."""
    resource_name = ("projects/778397675435/locations/us-east1/"
                     "indexEndpoints/4335185232320790528")

    class _Lazy:
        display_name = rs.INDEX_ENDPOINT_NAME
        # resource_name assigned below (class body can't see function locals)

    _Lazy.resource_name = resource_name

    _real = object()
    with patch.object(rs.aiplatform, "init"), \
         patch.object(rs.aiplatform.MatchingEngineIndexEndpoint, "list",
                      return_value=[_Lazy()]), \
         patch.object(rs.aiplatform.MatchingEngineIndexEndpoint, "__new__",
                      return_value=_real) as new_:
        result = rs._index_endpoint()

    assert result is _real
    assert new_.call_args[0][1] == resource_name
