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


# --- Section-anchored retrieval fallback (2026-08-17 regression) ------------
#
# Root cause: the meds chip's free-text query "discharge medications" embedded
# far from the patient's discharge_medications chunks (query-side drift), so
# rag_search returned 0 even though the chunks were indexed. The fix retries
# with the section's ACTUAL text as the query. These tests pin that behavior.


class _Recorder:
    """Fake endpoint that records queries and returns neighbors per call."""

    def __init__(self, first_round: list, second_round: list):
        self._rounds = [first_round, second_round]
        self.calls = 0
        self.queries = []
        self.last_filter = None

    def find_neighbors(self, *, deployed_index_id, queries, num_neighbors, filter=None):
        self.calls += 1
        self.queries.append(queries[0])
        self.last_filter = filter
        return [self._rounds[min(self.calls - 1, 1)]]


_NOTE = ("MEDICATIONS ON ADMISSION\nSome home meds.\n\n"
         "Discharge Medications:\n1. Docusate Sodium 100 mg Capsule PO BID\n"
         "2. Albuterol 90 mcg Aerosol Q4H PRN\n\n"
         "Discharge Instructions:\nFollow up with PCP.")


def test_meds_query_falls_back_to_section_anchor():
    """A meds query with 0 direct hits must retry with the section's text.

    Regression for hadm 23613002: "discharge medications" returned 0 though
    the discharge_medications chunks are indexed. The fallback re-queries with
    the section body and returns the meds passage.
    """
    meds_neighbor = _FakeNeighbor("13219116-DS-18_discharge_medications_1", 0.28)
    endpoint = _Recorder([], [meds_neighbor])
    embed = _FakeEmbedClient()

    def fake_fetch(note_ids):
        return {nid: _NOTE for nid in note_ids}

    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: embed), \
         patch.object(rs, "_fetch_texts", fake_fetch), \
         patch.object(rs, "_fetch_note", lambda hadm: _NOTE):
        result = _run(rs.rag_search(hadm_id=23613002, query="discharge medications"))

    # Fell back to the section-anchored query (second call) and returned the meds passage.
    assert endpoint.calls == 2
    assert result["returned"] == 1
    assert result["passages"][0]["section"] == "discharge_medications"
    assert result["passages"][0]["text"] == _NOTE
    # The fallback embeds the section's body, not the original short phrase:
    # the last embed call's contents carry the section text.
    assert embed.models.last_contents is not None
    assert "Docusate Sodium" in embed.models.last_contents[0]
    # The response `query` stays the caller's original phrase (the UI renders it
    # as the source-card title) — the section body must NOT leak into it.
    assert result["query"] == "discharge medications"


def test_non_section_query_still_returns_empty():
    """A query with no section intent and 0 hits stays empty (no fabrication)."""
    endpoint = _Recorder([], [])
    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts", lambda note_ids: {}):
        result = _run(rs.rag_search(hadm_id=23613002, query="what did the nurse chart say"))

    assert result == {"hadm_id": 23613002, "query": "what did the nurse chart say",
                      "returned": 0, "passages": []}
    assert endpoint.calls == 1


def test_meds_fallback_keeps_hadm_restrict():
    """The fallback re-query must still apply the hadm restrict server-side."""
    endpoint = _Recorder([], [_FakeNeighbor("13219116-DS-18_discharge_medications_1", 0.28)])
    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts", lambda note_ids: {nid: _NOTE for nid in note_ids}), \
         patch.object(rs, "_fetch_note", lambda hadm: _NOTE):
        _run(rs.rag_search(hadm_id=23613002, query="discharge medications"))
    assert endpoint.calls == 2
    assert endpoint.last_filter is not None
    assert endpoint.last_filter[0].allow_tokens == ["23613002"]
