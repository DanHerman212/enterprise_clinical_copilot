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
import logging
import re
from functools import lru_cache
from typing import Any

from google import genai
from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
)
from google.genai import types

from rag.sections import parse_note
from rag.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_PACK_TO,
    INDEX_SECTIONS,
    chunk_note,
)

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
from ._validation import valid_hadm_id

# The narrative sections we index — single-sourced from rag.chunking, the same
# tuple the ingest pipeline whitelists, so build and serving can never drift.
# A returned datapoint id is "{note_id}_{section}_{ordinal}" and section names
# contain underscores (brief_hospital_course), so we match the exact section
# token to recover both note_id and section instead of blindly splitting on '_'.
_KNOWN_SECTIONS = INDEX_SECTIONS
_SECTION_RE = re.compile(r"_(?P<section>(" + "|".join(_KNOWN_SECTIONS) + r"))_")

_LOG = logging.getLogger(__name__)


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


class IsolationViolation(Exception):
    """A retrieved note resolved to a DIFFERENT admission (R1 breach)."""


def _fetch_texts(note_ids: list[str], hadm_id: int) -> dict[str, str]:
    """note_id -> full note text, one batched BigQuery query.

    Second, independent enforcement of R1 (ECC-19): the index restrict is the
    first layer, but any upstream defect (wrong/missing restrict token, stale
    index, datapoint-id truncation) would flow another patient's text straight
    into a citation. Every resolved row's hadm_id is re-checked here at the
    BigQuery layer and a mismatch is a hard error, never a served passage.
    """
    if not note_ids:
        return {}
    deduped = sorted(set(note_ids))
    params = [bigquery.ArrayQueryParameter("note_ids", "STRING", deduped)]
    rows = _bigquery().query(
        f"SELECT note_id, hadm_id, text FROM `{DISCHARGE_TABLE}` "
        "WHERE note_id IN UNNEST(@note_ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    ).result()
    texts: dict[str, str] = {}
    foreign: list[str] = []
    for r in rows:
        if int(r["hadm_id"]) != hadm_id:
            foreign.append(str(r["note_id"]))
        else:
            texts[str(r["note_id"])] = r["text"]
    if foreign:
        raise IsolationViolation(
            f"Index returned note(s) {foreign} that do not belong to "
            f"admission {hadm_id} — index restrict defect; refusing to serve."
        )
    return texts


def _fetch_note_row(hadm_id: int) -> tuple[str | None, str | None]:
    """(note_id, text) for one admission, or (None, None).

    ORDER BY note_id (ECC-27): BigQuery gives LIMIT 1 no ordering guarantee,
    so with multiple note rows the cited note would be nondeterministic
    across calls.
    """
    rows = _bigquery().query(
        f"SELECT note_id, text FROM `{DISCHARGE_TABLE}` "
        "WHERE hadm_id = @hadm_id ORDER BY note_id LIMIT 1",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("hadm_id", "INTEGER", hadm_id),
        ]),
    ).result()
    for row in rows:
        return str(row["note_id"]), row["text"]
    return None, None


def _chunk_texts_for(note_id: str, text: str) -> dict[str, str]:
    """Index-form chunk id -> chunk text, deterministically re-chunked.

    The index stores section-level chunk embeddings (datapoint id =
    "{note_id}_{section}_{ordinal}"). Re-running the same deterministic chunker
    over the note — with the SAME parameters the ingest pipeline used, or the
    ids don't line up — reproduces each chunk's exact text, so a passage can
    return the SECTION it cites instead of the whole note — whole-note text is
    what leaks unrelated sections (allergies, activity) into a citation.
    """
    out: dict[str, str] = {}
    if not text:
        return out
    for chunk in chunk_note({"hadm_id": 0, "note_id": note_id, "text": text},
                            max_chars=DEFAULT_MAX_CHARS, pack_to=DEFAULT_PACK_TO):
        out[chunk.chunk_id.replace(":", "_")] = chunk.text
    return out


def _search(hadm_id: int, query: str, top_k: int, *,
            is_retry: bool = False) -> dict[str, Any]:
    """Blocking implementation. Wrapped in a thread by the tool below.

    `is_retry` marks the single permitted section-anchored retry: the retried
    call must never anchor again, or a section body that itself matches a
    section intent recurses without bound — every level a billed embed +
    index query + BigQuery fetch.
    """
    if not valid_hadm_id(hadm_id):
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

    # A query that clearly targets one note section (medications, instructions,
    # hospital course, diagnoses) can still rank a sibling section first — the
    # near-tie chunk embeddings. When the intended section did NOT win rank 1,
    # retry ONCE with the section's ACTUAL text as the query so the matching
    # chunk ranks first. This generalizes the zero-hit query-side-drift
    # fallback below to wrong-rank results (2026-08-24, recall@1 82% -> ~100%).
    anchor = None if is_retry else _section_for_query(query)
    if neighbors and anchor is not None:
        if _parse_datapoint_id(neighbors[0].id)[1] != anchor:
            body = _section_bodies(hadm_id).get(anchor)
            if body:
                retried = _search(hadm_id, body, top_k, is_retry=True)
                # Keep the original hits unless the retry actually improved on
                # them — an empty or failed retry must not destroy real recall.
                if not retried.get("error") and retried.get("returned", 0) > 0:
                    return {**retried, "query": query}

    if not neighbors:
        # Free-text queries embed far from the stored chunks for some patients
        # (query-side drift, root-caused 2026-08-17 on hadm 23613002: the meds
        # chip's "discharge medications" query returned 0 though the chunks are
        # indexed). When the query clearly targets a discharge-note section, retry
        # with that section's ACTUAL text as the query — the section body embeds
        # next to the chunk it came from, so retrieval no longer depends on query
        # luck. Keeps the hadm restrict and the same index (RAG intact). The
        # returned `query` stays the caller's original phrase (the UI shows it as
        # the source-card title), not the section body used internally.
        if anchor is not None:
            body = _section_bodies(hadm_id).get(anchor)
            if body:
                res = _search(hadm_id, body, top_k, is_retry=True)
                if res.get("error"):
                    return res
                return {**res, "query": query}
        return {"hadm_id": hadm_id, "query": query, "returned": 0, "passages": []}

    # Resolve text by note_id in one batched query, re-checking that every
    # note belongs to THIS admission (second R1 layer — ECC-19).
    parsed = [_parse_datapoint_id(nb.id) for nb in neighbors]
    note_ids = [p[0] for p in parsed if p[0]]
    try:
        texts = _fetch_texts(note_ids, hadm_id)
    except IsolationViolation as exc:
        _LOG.error("rag_search: %s", exc)
        return _error(hadm_id, "isolation_violation", str(exc))

    # Deterministically re-chunk each fetched note so a passage returns the
    # exact SECTION chunk the index matched, not the whole note.
    chunk_texts = {nid: _chunk_texts_for(nid, t) for nid, t in texts.items()}

    passages = []
    for nb, (note_id, section) in zip(neighbors, parsed):
        # An id that doesn't parse means the index holds a datapoint this
        # server doesn't understand (stale section vocabulary, foreign
        # datapoint). Error rather than emit a citation with no text.
        if note_id is None:
            return _error(
                hadm_id, "unparsed_datapoint",
                f"Index returned id {nb.id!r} that does not match "
                "'{note_id}_{section}_{ordinal}' for any indexed section",
            )
        text = texts.get(note_id)
        # A returned ID with no text in BigQuery must error, not silently drop
        # (a dropped passage looks like a retrieval gap and is hard to debug).
        if text is None:
            return _error(
                hadm_id, "missing_text",
                f"Index returned id {nb.id} but no note {note_id} in {DISCHARGE_TABLE}",
            )
        # Prefer the exact chunk. A miss means the serving chunker has drifted
        # from the index build; keep retrieval alive with the whole note, but
        # say so — in the log and on the passage itself — instead of silently
        # widening the citation.
        body = chunk_texts.get(note_id, {}).get(nb.id)
        passage = {
            "id": nb.id,
            "section": section,
            "text": body if body is not None else text,
            "score": round(float(nb.distance), 4),
        }
        if body is None:
            _LOG.warning(
                "rag_search: chunk %s not reproduced by the serving chunker; "
                "returning whole-note text (chunker drift — rebuild the index "
                "after chunker changes)", nb.id,
            )
            passage["granularity"] = "note"
        passages.append(passage)

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


# Discharge-note sections a summary should cite, in the order a summary
# should cite them. Queries are anchored to each section's ACTUAL text (see
# _search_sections): short free-text phrases embed far from the stored chunks
# for some patients (query-side drift, root-caused 2026-08-17 on hadm
# 23613002), so we query with the section body itself instead.
SUMMARY_SECTIONS = (
    "brief_hospital_course",
    "discharge_diagnosis",
    "discharge_medications",
    "discharge_instructions",
    "discharge_summary",
)


def _fetch_note(hadm_id: int) -> str | None:
    """The patient's discharge note text, for section anchoring.

    The index stores chunk embeddings, not raw text, so the section body is
    re-fetched from BigQuery and re-parsed with the same section parser the
    chunker uses. That keeps the anchored query aligned with what the index
    embeds (same chunk boundaries).
    """
    rows = _bigquery().query(
        f"SELECT text FROM `{DISCHARGE_TABLE}` WHERE hadm_id = @hadm_id "
        "ORDER BY note_id LIMIT 1",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("hadm_id", "INTEGER", hadm_id)
            ]
        ),
    ).result()
    for row in rows:
        return row["text"]
    return None


def _section_bodies(hadm_id: int) -> dict[str, str]:
    """section name -> body text, from the patient's discharge note."""
    text = _fetch_note(hadm_id)
    if not text:
        return {}
    parsed = parse_note(text)
    return {s.name: s.body for s in parsed.sections}


def _section_for_query(query: str) -> str | None:
    """Map a free-text query to the discharge-note section it clearly targets.

    Used as a retrieval fallback: when the raw query embedding misses, retry
    with the section's actual body text. Only matches unambiguous intent so the
    free-text path is unchanged for open-ended questions.
    """
    q = query.lower().strip()
    if "medication" in q or "discharged on" in q or "meds" in q:
        return "discharge_medications"
    if "discharge diagnosis" in q or "diagnoses" in q:
        return "discharge_diagnosis"
    if "instruction" in q:
        return "discharge_instructions"
    if "hospital course" in q or "admission course" in q:
        return "brief_hospital_course"
    return None


def _search_sections(hadm_id: int) -> dict[str, Any]:
    """One passage per major note section, merged in a fixed order.

    Deterministic section coverage for summary questions WITHOUT relying on the
    embedding index: re-parse the note and re-chunk it with the same
    deterministic chunker that built the index, so every section present in the
    note is returned with its exact chunk text. The agent cites ^[n] in this
    fixed order, so each section maps to a distinct citation number and recall
    is 100% by construction — no top-k luck, no whole-note leakage, no
    dropped sections.
    """
    if not valid_hadm_id(hadm_id):
        return _error(hadm_id, "bad_request", "hadm_id must be a positive integer")
    note_id, text = _fetch_note_row(hadm_id)
    if not text:
        return {"hadm_id": hadm_id, "query": "discharge notes",
                "returned": 0, "passages": [],
                "note": "no discharge note found for this admission"}
    chunks = _chunk_texts_for(note_id, text)
    by_section: dict[str, list[tuple[str, str]]] = {}
    for cid, ctext in chunks.items():
        m = _SECTION_RE.search(cid)
        if m:
            by_section.setdefault(m.group("section"), []).append((cid, ctext))

    merged: list[dict[str, Any]] = []
    for expected in SUMMARY_SECTIONS:
        picks = by_section.get(expected)
        if not picks:
            continue  # note lacks this section -> nothing to cite
        cid, ctext = picks[0]
        # These passages are re-parsed from the note, not retrieved by the
        # index, so an embedding score would be a fabrication. Mark them
        # honestly instead of the old hardcoded 1.0.
        merged.append({"id": cid, "section": expected, "text": ctext,
                       "retrieval": "deterministic"})

    if not merged:
        # A real note can lack all discharge-narrative summary sections (e.g. a
        # speech-therapy or narrative note). Honest empty rather than guessing.
        return {"hadm_id": hadm_id, "query": "discharge notes",
                "returned": 0, "passages": [],
                "note": "no discharge summary sections found in this note"}
    return {"hadm_id": hadm_id, "query": "discharge notes",
            "returned": len(merged), "passages": merged}


async def rag_search_sections(hadm_id: int) -> dict[str, Any]:
    """Retrieve one cited passage per major discharge-note section (hospital
    course, discharge diagnosis, discharge medications, discharge
    instructions), merged in that fixed order. Use for summarization questions
    so the answer can cite each section distinctly.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
    """
    return await asyncio.to_thread(_search_sections, hadm_id)
