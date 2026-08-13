"""The rag_search tool.

Retrieves cited passages from a patient's discharge notes for a free-text
query, using the deployed Vector Search index.

Returns plain JSON, never A2UI (same rule as predict.py). The agent composes
presentation from this payload.

The `hadm_id` restrict is applied server-side, always: it is passed to the
index query itself, not applied as a post-filter on the results. That is R1
from the build guide and it is the difference between a demo and a liability.

Contract:
    rag_search(hadm_id: int, query: str, top_k: int = 5) -> dict
    {
      "hadm_id": 20924467,
      "query": "...",
      "returned": 2,
      "passages": [
        {"id": "noteid_section_ordinal", "section": "brief_hospital_course",
         "text": "...", "score": 0.27}
      ]
    }

Empty is a real answer: {"passages": [], "returned": 0}. Never fabricate.
"""

import asyncio
import re
from functools import lru_cache
from typing import Any

from google import genai
from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
)
from google.genai import types

from ..config import (
    DEPLOYED_INDEX_ID,
    DISCHARGE_TABLE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_ENDPOINT_NAME,
    LOCATION,
    PROJECT,
    RESTRICT_NAMESPACE,
)

# The narrative sections we index (matches chunk_notes.py DEFAULT_SECTIONS).
# A returned datapoint id is "{note_id}_{section}_{ordinal}" and section names
# contain underscores (brief_hospital_course), so we match the exact section
# token to recover both note_id and section instead of blindly splitting on '_'.
_KNOWN_SECTIONS = (
    "history_of_present_illness",
    "past_medical_history",
    "family_history",
    "social_history",
    "physical_exam",
    "brief_hospital_course",
    "discharge_condition",
    "discharge_diagnosis",
    "discharge_medications",
    "medications_on_admission",
    "discharge_disposition",
    "discharge_instructions",
)
_SECTION_RE = re.compile(r"_(?P<section>(" + "|".join(_KNOWN_SECTIONS) + r"))_")


@lru_cache(maxsize=1)
def _index_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """The Vector Search index endpoint, resolved by display name once.

    NOTE: .list() returns lazily-loaded objects whose match client
    (_public_match_client) is not initialized, so find_neighbors would crash.
    Re-construct with the full resource name, which is what initializes it.
    """
    aiplatform.init(project=PROJECT, location=LOCATION)
    for ep in aiplatform.MatchingEngineIndexEndpoint.list():
        if ep.display_name == INDEX_ENDPOINT_NAME:
            return aiplatform.MatchingEngineIndexEndpoint(ep.resource_name)
    raise RuntimeError(
        f"No index endpoint with display_name={INDEX_ENDPOINT_NAME!r} "
        f"in {PROJECT}/{LOCATION}. Redeploy with scripts/deploy_index.py."
    )


@lru_cache(maxsize=1)
def _embed_client() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


@lru_cache(maxsize=1)
def _bigquery() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _error(hadm_id: int, code: str, message: str) -> dict[str, Any]:
    """A failure the agent can read and explain, rather than a stack trace."""
    return {"hadm_id": hadm_id, "error": code, "message": message}


def _parse_datapoint_id(datapoint_id: str) -> tuple[str | None, str | None]:
    """Recover (note_id, section) from "{note_id}_{section}_{ordinal}"."""
    match = _SECTION_RE.search(datapoint_id)
    if not match:
        return None, None
    note_id = datapoint_id[:match.start()]
    return note_id, match.group("section")


def _fetch_texts(note_ids: list[str]) -> dict[str, str]:
    """note_id -> full note text, one batched BigQuery query."""
    if not note_ids:
        return {}
    deduped = sorted(set(note_ids))
    params = [bigquery.ArrayQueryParameter("note_ids", "STRING", deduped)]
    rows = _bigquery().query(
        f"SELECT note_id, text FROM `{DISCHARGE_TABLE}` "
        "WHERE note_id IN UNNEST(@note_ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    ).result()
    return {str(r["note_id"]): r["text"] for r in rows}


def _search(hadm_id: int, query: str, top_k: int) -> dict[str, Any]:
    """Blocking implementation. Wrapped in a thread by the tool below."""
    if not isinstance(hadm_id, int) or hadm_id <= 0:
        return _error(hadm_id, "bad_request", "hadm_id must be a positive integer")
    if not query or not query.strip():
        return _error(hadm_id, "bad_request", "query must be non-empty")
    if not (1 <= top_k <= 20):
        return _error(hadm_id, "bad_request", "top_k must be between 1 and 20")

    # Embed the query.
    try:
        resp = _embed_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIM,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        query_vec = [float(v) for v in resp.embeddings[0].values]
    except Exception as exc:  # network, auth, model unreachable
        return _error(hadm_id, "embed_failed", f"{type(exc).__name__}: {exc}")

    # Query the index with the hadm_id restrict applied server-side.
    try:
        res = _index_endpoint().find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_vec],
            num_neighbors=top_k,
            filter=[Namespace(RESTRICT_NAMESPACE, [str(hadm_id)])],
        )
    except Exception as exc:
        return _error(hadm_id, "search_failed", f"{type(exc).__name__}: {exc}")

    neighbors = res[0] if res else []
    if not neighbors:
        return {"hadm_id": hadm_id, "query": query, "returned": 0, "passages": []}

    # Resolve text by note_id in one batched query.
    parsed = [_parse_datapoint_id(nb.id) for nb in neighbors]
    note_ids = [p[0] for p in parsed if p[0]]
    texts = _fetch_texts(note_ids)

    passages = []
    for nb, (note_id, section) in zip(neighbors, parsed):
        text = texts.get(note_id or "")
        # A returned ID with no text in BigQuery must error, not silently drop
        # (a dropped passage looks like a retrieval gap and is hard to debug).
        if note_id and text is None:
            return _error(
                hadm_id, "missing_text",
                f"Index returned id {nb.id} but no note {note_id} in {DISCHARGE_TABLE}",
            )
        passages.append({
            "id": nb.id,
            "section": section,
            "text": text,
            "score": round(float(nb.distance), 4),
        })

    return {"hadm_id": hadm_id, "query": query, "returned": len(passages),
            "passages": passages}


async def rag_search(hadm_id: int, query: str, top_k: int = 5) -> dict[str, Any]:
    """Retrieve cited passages from a patient's discharge notes for a query.

    Searches the patient's own notes only (the hadm_id restrict is applied
    server-side). Returns passages with the section name and text; an empty
    result means no supporting passages were found.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
        query: free-text clinical question or search phrase.
        top_k: max passages to return (1-20, default 5).
    """
    # Vertex/gRPC calls are synchronous; under the HTTP transport a blocking
    # tool stalls the event loop, so the work goes to a worker thread.
    return await asyncio.to_thread(_search, hadm_id, query, top_k)


# One (query, expected section) per major discharge-note section, in the order
# a summary should cite them. Broad single queries retrieve poorly against this
# index (query-side embedding drift vs the index snapshot); these focused
# queries return their section among the top few — empirically verified 2026-08-13.
SUMMARY_SECTIONS = (
    ("admission course and chief complaint", "brief_hospital_course"),
    ("primary discharge diagnosis", "discharge_diagnosis"),
    ("discharge medications", "discharge_medications"),
    ("discharge instructions", "discharge_instructions"),
)


def _search_sections(hadm_id: int) -> dict[str, Any]:
    """One passage per major note section, merged in a fixed order.

    Deterministic section coverage for summary questions without depending on
    the model to issue and merge multiple calls: run one focused query per
    section, prefer the returned passage whose section matches the intent, and
    return a single deduped, section-ordered passages list. The agent cites
    ^[n] in this order, so each section maps to a distinct citation number.
    """
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query, expected in SUMMARY_SECTIONS:
        try:
            res = _search(hadm_id, query, top_k=3)
        except Exception:
            continue  # a failed section query must not sink the whole summary
        if res.get("error"):
            continue
        passages = res.get("passages") or []
        if not passages:
            continue
        # Prefer the passage whose section matches the intent; fall back to the
        # top passage (a wrong-section hit is better than nothing).
        pick = next((p for p in passages if p["section"] == expected), passages[0])
        if pick["id"] in seen:
            continue
        seen.add(pick["id"])
        merged.append(pick)
    return {
        "hadm_id": hadm_id,
        "query": "discharge notes",
        "returned": len(merged),
        "passages": merged,
    }


async def rag_search_sections(hadm_id: int) -> dict[str, Any]:
    """Retrieve one cited passage per major discharge-note section (hospital
    course, discharge diagnosis, discharge medications, discharge
    instructions), merged in that fixed order. Use for summarization questions
    so the answer can cite each section distinctly.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
    """
    return await asyncio.to_thread(_search_sections, hadm_id)
