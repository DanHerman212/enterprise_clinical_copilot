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
from typing import Annotated, Any

from google import genai
from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
)
from google.genai import types
from pydantic import Field

from services.mcp.retrieval.embed import (
    EMBEDDING_MODEL,
    OUTPUT_DIMENSIONALITY,
    RESTRICT_NAMESPACE,
)
from services.mcp.dependencies.features import (
    FeatureSource,
    get_feature_source,
)
from services.mcp.retrieval.sections import (
    ADMISSION_DIAGNOSIS,
    ASSESSMENT_AND_PLAN,
    BRIEF_HOSPITAL_COURSE,
    DISCHARGE_DIAGNOSIS,
    DISCHARGE_INSTRUCTIONS,
    DISCHARGE_MEDICATIONS,
    DISCHARGE_SUMMARY,
    HISTORY_OF_PRESENT_ILLNESS,
    MEDICATIONS_ON_ADMISSION,
    MEDICATIONS_ON_TRANSFER,
    parse_note,
    redaction_profile,
)
from services.mcp.retrieval.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_PACK_TO,
    INDEX_SECTIONS,
    chunk_note,
)

from ..config import (
    DEPLOYED_INDEX_ID,
    DISCHARGE_TABLE,
    INDEX_ENDPOINT_NAME,
    LOCATION,
    PROJECT,
)
from ..contracts import (
    RetrievalResult,
    ToolContractError,
    ToolError,
    tool_error,
    validate_retrieval_result,
)
from ._validation import valid_hadm_id

# The narrative sections we index — single-sourced from services.mcp.retrieval.chunking, the same
# tuple the ingest pipeline whitelists, so build and serving can never drift.
# A returned datapoint id is "{note_id}_{section}_{ordinal}" and section names
# contain underscores (brief_hospital_course), so we match the exact section
# token to recover both note_id and section instead of blindly splitting on '_'.
_KNOWN_SECTIONS = INDEX_SECTIONS
_SECTION_RE = re.compile(r"_(?P<section>(" + "|".join(_KNOWN_SECTIONS) + r"))_")

_LOG = logging.getLogger(__name__)

# The one parameter in this layer with a range. Defined once so the advertised schema and the
# check that enforces it cannot disagree: a schema steers a caller, it does not stop one.
TOP_K_MIN = 1
TOP_K_MAX = 20


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


def _error(
    hadm_id: int, code: str, message: str, *, detail: str | None = None
) -> dict[str, Any]:
    """A failure the caller can act on, with the detail left in the log.

    The message IS the tool result: it reaches the model, the caller and the
    browser, so it carries a sentence and nothing else. Ids, table names and
    exception text are diagnostics — an operator needs them and a prompt must not
    have them, and `str(exc)` on the isolation path names another patient.
    """
    if detail:
        _LOG.warning("%s refused for hadm %s: %s", code, hadm_id, detail)
    return tool_error(code, message, hadm_id=hadm_id)


@lru_cache(maxsize=1)
def _admission_source() -> FeatureSource:
    return get_feature_source()


def _admission_known(hadm_id: int) -> bool | None:
    """Whether this admission is one the site serves at all.

    Retrieval must not answer "no notes" for an admission that does not exist: the
    prompt tells the model an all-empty result is a real answer, so a typo becomes a
    claim about a patient rather than a statement that the identifier is unknown.
    Prediction already tells those apart, and this asks the same source it asks, so
    `unknown_patient` means one thing across the three tools.

    None means the lookup itself failed, and it is deliberately not False: a failed
    query must not become a claim that the patient does not exist. The caller keeps
    its empty answer and the failure goes to the log.
    """
    try:
        return _admission_source().exists(hadm_id)
    except Exception:
        _LOG.error("admission lookup failed for hadm %s", hadm_id, exc_info=True)
        return None


def _unknown_if_absent(hadm_id: int) -> dict[str, Any] | None:
    """The unknown-admission error, or None when the admission is one we serve."""
    if _admission_known(hadm_id) is False:
        return _error(
            hadm_id, "unknown_patient",
            "No such admission in the dataset.",
            detail="absent from the feature source (admission lookup returned no row)",
        )
    return None


def _nothing_retrieved(hadm_id: int, query: str) -> dict[str, Any]:
    """Nothing to cite for this admission — unless there is no such admission.

    An empty result is only a real answer when the admission is one we serve. The
    prompt tells the model to read it as one, so answering this way for an admission
    that does not exist turns a typo into a statement about a patient (gap 6).
    """
    unknown = _unknown_if_absent(hadm_id)
    if unknown:
        return unknown
    _LOG.info("rag_search: nothing retrieved for hadm %s", hadm_id)
    return {"hadm_id": hadm_id, "query": query, "returned": 0, "passages": [],
            "note": "no passages were retrieved for this admission"}


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
    # The schema carries this bound, so over MCP the SDK refuses an out-of-range argument
    # before this line runs. Kept as the guard for a caller that bypasses the schema.
    if not (TOP_K_MIN <= top_k <= TOP_K_MAX):
        return _error(
            hadm_id,
            "bad_request",
            f"top_k must be between {TOP_K_MIN} and {TOP_K_MAX}",
        )

    # Embed the query.
    try:
        resp = _embed_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(
                output_dimensionality=OUTPUT_DIMENSIONALITY,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        query_vec = [float(v) for v in resp.embeddings[0].values]
    except Exception as exc:  # network, auth, model unreachable
        # Detail stays server-side (ECC-21): exception text can embed project
        # ids, URLs and IAM detail, and the error message reaches the model.
        _LOG.error("rag_search: embed failed for hadm %s", hadm_id, exc_info=exc)
        return _error(hadm_id, "embed_failed",
                      "The embedding service could not be reached. Try again shortly.")

    # Query the index with the hadm_id restrict applied server-side.
    try:
        res = _index_endpoint().find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_vec],
            num_neighbors=top_k,
            filter=[Namespace(RESTRICT_NAMESPACE, [str(hadm_id)])],
        )
    except Exception as exc:
        _LOG.error("rag_search: index query failed for hadm %s", hadm_id, exc_info=exc)
        return _error(hadm_id, "search_failed",
                      "The retrieval index could not be reached. Try again shortly.")

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
        return _nothing_retrieved(hadm_id, query)

    # Resolve text by note_id in one batched query, re-checking that every
    # note belongs to THIS admission (second R1 layer — ECC-19).
    parsed = [_parse_datapoint_id(nb.id) for nb in neighbors]
    note_ids = [p[0] for p in parsed if p[0]]
    try:
        texts = _fetch_texts(note_ids, hadm_id)
    except IsolationViolation as exc:
        # The refusal may say a note did not belong to this admission; it must not
        # say which note. A note id is "{subject_id}-DS-{note}", so `str(exc)`
        # names another patient — and this message reaches the model, the caller
        # and the browser (ECC-21 / gap 4).
        return _error(
            hadm_id, "isolation_violation",
            "A retrieved note did not belong to the requested admission; "
            "no passage was served.",
            detail=str(exc),
        )

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
                "The index returned an identifier this server does not recognise.",
                detail=(
                    f"datapoint id {nb.id!r} does not match "
                    "'{note_id}_{section}_{ordinal}' for any indexed section"
                ),
            )
        text = texts.get(note_id)
        # A returned ID with no text in BigQuery must error, not silently drop
        # (a dropped passage looks like a retrieval gap and is hard to debug).
        if text is None:
            return _error(
                hadm_id, "missing_text",
                "The index returned a passage with no text in the source table.",
                detail=f"note {note_id} is not in {DISCHARGE_TABLE}",
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

    if not passages:
        return _nothing_retrieved(hadm_id, query)

    return {"hadm_id": hadm_id, "query": query, "returned": len(passages),
            "passages": passages}


async def rag_search(
    hadm_id: int,
    query: str,
    top_k: Annotated[
        int,
        Field(
            ge=TOP_K_MIN,
            le=TOP_K_MAX,
            description="max passages to return (1-20, default 5)",
        ),
    ] = 5,
) -> RetrievalResult | ToolError:
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
    payload = await asyncio.to_thread(_search, hadm_id, query, top_k)
    try:
        return validate_retrieval_result(payload)
    except ToolContractError:
        _LOG.error("rag_search: produced payload outside its tool contract")
        return _error(hadm_id, "invalid_tool_response", "Retrieval response was malformed.")


# The sections a summary cites come from the retrieval vocabulary rather than
# being declared here — `services.mcp.retrieval.chunking` owns both the indexed
# section set and this subset, because the agent's routing table reads the same
# tuple and a local copy could drift from it.


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


def _norm_body(text: str) -> str:
    """A section body folded to the form two sections are compared in.

    Whitespace collapsed and case removed, so a note that prints the same block under two
    headings with different line wrapping is recognised as one block. Nothing else is
    folded: two bodies that differ by a word are two statements, and consolidating them
    would be an editorial decision this layer is not entitled to make.
    """
    return " ".join(str(text or "").split()).lower()


def _keep_final_diagnosis(passage: dict[str, Any], incoming: str, note_id: str,
                          ordinals: dict[str, int]) -> None:
    """Retag a duplicated admission-diagnosis passage as the discharge diagnosis.

    Called when a body has been seen before. The common shape is a note that lists the
    same diagnoses under `ADMISSION DIAGNOSES` and again under `DISCHARGE DIAGNOSES`: one
    block, two headings. The passage is kept once, and the heading that survives is the
    final one — a summary asks what the patient left with, and a citation labelled
    "Admission diagnosis" pointing at the discharge list is a label the note contradicts.

    Only that pair is upgraded. Any other duplicate keeps the label it arrived with: this
    is a rule about which of two names for one block is the better name, not a rename pass
    over the whole note.
    """
    if passage.get("section") != ADMISSION_DIAGNOSIS or incoming != DISCHARGE_DIAGNOSIS:
        return
    # The admission ordinal goes back into the pool: the passage is no longer an admission
    # diagnosis, so a later, genuinely different admission block should be `_1` rather than
    # `_2` with nothing before it.
    if ordinals.get(ADMISSION_DIAGNOSIS):
        ordinals[ADMISSION_DIAGNOSIS] -= 1
    ordinals[DISCHARGE_DIAGNOSIS] = ordinals.get(DISCHARGE_DIAGNOSIS, 0) + 1
    passage["section"] = DISCHARGE_DIAGNOSIS
    passage["id"] = f"{note_id}_{DISCHARGE_DIAGNOSIS}_{ordinals[DISCHARGE_DIAGNOSIS]}"


def _search_sections(hadm_id: int) -> dict[str, Any]:
    """Every section of the discharge note, in the order the note states them.

    This is what `/ehr`'s narrative rests on, so it is lossless by construction: the note
    is parsed once with `parse_note` and each parsed section becomes one passage, with the
    text the parser found and the name it gave it. No chunker, no section whitelist, no
    per-section truncation to the first chunk.

    What it replaced, and why that was wrong: the previous version re-chunked the note with
    the index's chunker and returned the FIRST chunk of each of four `SUMMARY_SECTIONS`.
    Three separate losses followed from that. A note's sections outside those four —
    operative procedures, complications, consults, the assessment and plan — were never
    returned at all, so the narrative could not mention the operation the admission was
    for. A section longer than one chunk lost its tail, which for a hospital course is
    usually where the outcome is. And sections were re-ordered into the whitelist's
    order, so a summary that cites `^[3]` and `^[4]` pointed at passages that were not in
    the order the note reads in. Measured on the cohort 2026-11-03: fourteen of eighty-seven
    notes had content the old path could not reach.

    Order is the document's, which is also the order the formatting turn reads and the
    citation ledger numbers. Duplicate section names keep their own ordinals — a note that
    states a diagnosis on admission and again at discharge yields two passages, `_1` and
    `_2`, rather than one overwriting the other.

    Byte-identical bodies are collapsed (2026-11-03): a note that repeats one block under
    `ADMISSION DIAGNOSES` and `DISCHARGE DIAGNOSES` is stating one fact twice, and two
    passages for it give the prose two citations for one sentence. The surviving passage
    keeps the final diagnosis's name, because that is the heading a summary is answered by.

    The text before the first recognised heading is real note text and is emitted as the
    history of present illness, which is what an unheaded preamble of a discharge summary
    is. A note with no recognised headings at all is a narrative note; the whole of it is
    returned as one passage rather than as nothing.
    """
    if not valid_hadm_id(hadm_id):
        return _error(hadm_id, "bad_request", "hadm_id must be a positive integer")
    note_id, text = _fetch_note_row(hadm_id)
    if not text:
        # No note for this admission — but which case is it? A note that was never
        # ingested is an empty record; an admission that does not exist is not a
        # record at all, and saying "no discharge note" about it invents a patient
        # (gap 6).
        unknown = _unknown_if_absent(hadm_id)
        if unknown:
            return unknown
        return {"hadm_id": hadm_id, "query": "discharge notes",
                "returned": 0, "passages": [],
                "note": "no discharge note found for this admission"}

    # Folded exactly as `parse_note` folds it, because the offsets below are offsets into
    # the folded text and a note saved with CRLF would otherwise slice the preamble at the
    # wrong character.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    parsed = parse_note(text)

    passages: list[dict[str, Any]] = []
    ordinals: dict[str, int] = {}
    #: Normalised body -> the index of the passage that already carries it. A note that
    #: states the same block under two headings is stating one thing twice, and a passage
    #: per heading gives the model two citations for one fact — which is exactly what
    #: `/ehr` did on 90000069, where `^[1]` and `^[2]` were the same two diagnoses
    #: (2026-11-03).
    seen_bodies: dict[str, int] = {}

    def emit(name: str, body: str) -> None:
        key = _norm_body(body)
        existing = seen_bodies.get(key)
        if existing is not None:
            _keep_final_diagnosis(passages[existing], name, note_id, ordinals)
            return
        ordinals[name] = ordinals.get(name, 0) + 1
        passages.append({
            "id": f"{note_id}_{name}_{ordinals[name]}",
            "section": name,
            "text": body,
            "retrieval": "deterministic",
        })
        seen_bodies[key] = len(passages) - 1

    if parsed.sections:
        first = parsed.sections[0]
        preamble = normalised[:first.start].strip()
        if preamble and not redaction_profile(preamble).is_placeholder_only:
            emit("history_of_present_illness", preamble)
        for section in parsed.sections:
            body = section.body.strip()
            if not body or redaction_profile(body).is_placeholder_only:
                continue
            emit(section.name, body)
    else:
        # A headingless narrative note. One passage, labelled for what it is, rather than
        # an empty result: the absence of headings is not the absence of a note.
        body = normalised.strip()
        if body and not redaction_profile(body).is_placeholder_only:
            emit("discharge_summary", body)

    if not passages:
        # Either every section was redaction placeholders or the note is empty of prose.
        # Honest empty rather than a fabricated passage.
        return {"hadm_id": hadm_id, "query": "discharge notes",
                "returned": 0, "passages": [],
                "note": "no readable sections found in this note"}
    return {"hadm_id": hadm_id, "query": "discharge notes",
            "returned": len(passages), "passages": passages}


async def rag_search_sections(hadm_id: int) -> RetrievalResult | ToolError:
    """Retrieve every section of the admission's discharge note, in document order.

    Use for summarization questions. Each passage is one section as the note states it,
    with the heading's own name, so the answer can cite the hospital course and the
    operative procedures as distinct sources and a reader who opens a marker lands on the
    passage that supports the sentence they were reading.

    Every section is returned, not a chosen few, and each is whole. The sections a
    narration can draw on are the record's, not this tool's, and a tool that decided which
    parts of a note were worth reading would be deciding what the summary could say.

    `discharge_medications` is included here as one passage among the rest; call
    `discharge_medications` when the complete medication list is what is needed, since that
    path returns the discharge-instructions section beside it and the doses live there.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
    """
    payload = await asyncio.to_thread(_search_sections, hadm_id)
    try:
        return validate_retrieval_result(payload)
    except ToolContractError:
        _LOG.error("rag_search_sections: produced payload outside its tool contract")
        return _error(hadm_id, "invalid_tool_response", "Retrieval response was malformed.")


#: The sections that exist to carry a DISCHARGE medication list. A note that has either of
#: them has its discharge regimen where the document says it is.
#:
#: `medications_on_transfer` is deliberately absent (2026-11-03). In a transitional-care or
#: rehabilitation note that heading is the list the patient ARRIVED on — "On transfer,
#: Celebrex, Coumadin, Colace, …" — carried over from the acute hospital, and a drug on it
#: may have been stopped before discharge. Reading it beside the discharge list put two
#: drugs the patient left without into a `/meds` grid, both with no dose, because an
#: arrival list states names and not regimens.
_DEDICATED_MED_SECTIONS = (
    DISCHARGE_MEDICATIONS,
    DISCHARGE_INSTRUCTIONS,
)

#: Where a discharge prescription goes when the note has no discharge medication section.
#: Surgical notes write the prescriptions at the end of the course — "we will send her home
#: with a steroid taper and an opioid" — and a headingless narrative note holds the only
#: medication mention the record has. Measured 2026-11-03: `/meds` on 90000069 returned
#: nothing, because the old path asked the chunker for two sections the note does not have.
#:
#: The transfer list belongs here rather than above: with no discharge section it is the
#: only medication text the record has, and a note whose sole list is "on transfer" is
#: better read than reported as having nothing.
_FALLBACK_MED_SECTIONS = (
    ASSESSMENT_AND_PLAN,
    BRIEF_HOSPITAL_COURSE,
    MEDICATIONS_ON_ADMISSION,
    MEDICATIONS_ON_TRANSFER,
    HISTORY_OF_PRESENT_ILLNESS,
    DISCHARGE_SUMMARY,
)


def _ordered_sections(parsed, normalised: str) -> list[tuple[str, str]]:
    """Every readable section of the note, in document order, naming the preamble.

    `_search_sections` builds this list inline; it is repeated here rather than shared
    because that function's shape is fixed by the `/ehr` contract and this one needs the
    same reading of the document — including the text before the first heading, which is
    the history in a discharge summary, and a headingless note, which is one narrative.
    """
    found: list[tuple[str, str]] = []
    if parsed.sections:
        first = parsed.sections[0]
        preamble = normalised[:first.start].strip()
        if preamble and not redaction_profile(preamble).is_placeholder_only:
            found.append((HISTORY_OF_PRESENT_ILLNESS, preamble))
        for section in parsed.sections:
            body = section.body.strip()
            if not body or redaction_profile(body).is_placeholder_only:
                continue
            found.append((section.name, body))
    else:
        body = normalised.strip()
        if body and not redaction_profile(body).is_placeholder_only:
            found.append((DISCHARGE_SUMMARY, body))
    return found


def _no_medication_sections_payload(hadm_id: int, note_id: str | None) -> dict[str, Any]:
    """The honest empty: this note states no discharge medications to cite."""
    _LOG.info("discharge_medications: no medication section in note %s (hadm %s)",
              note_id, hadm_id)
    return {"hadm_id": hadm_id, "query": "discharge medications",
            "returned": 0, "passages": [],
            "note": "no discharge medication section found in this note"}


def _search_medication_sections(hadm_id: int) -> dict[str, Any]:
    """The note's medication text, whole, from wherever this note keeps it.

    One pass over the parsed document, and a fallback that is the point of the rewrite.
    The old version asked the chunker for two named sections and concatenated their chunks,
    so a note without a `discharge_medications` section returned nothing at all — which is
    most surgical notes, where the prescriptions sit at the end of the course, and any note
    that has no headings. `/meds` on 90000069 answered "no medications" for a record that
    names three (2026-11-03).

    So: if the note has a section that exists to hold a medication list, those sections are
    the answer and nothing else is. If it has none, the sections a discharge prescription is
    actually written in are the answer — the assessment and plan, the course, the admission
    list, the history, or the whole of a headingless note. Never both: a record with a
    dedicated list is read from that list, because the narrative's inpatient drugs are not
    the patient's discharge medications and mixing them is how a `/meds` grid lists IV
    morphine as something to take at home.

    Every chosen section is returned whole — the truncation to the first chunk is gone — and
    duplicate bodies are dropped, so a note that states the same list twice cites it once.
    """
    if not valid_hadm_id(hadm_id):
        return _error(hadm_id, "bad_request", "hadm_id must be a positive integer")
    note_id, text = _fetch_note_row(hadm_id)
    if not text:
        unknown = _unknown_if_absent(hadm_id)
        if unknown:
            return unknown
        return {"hadm_id": hadm_id, "query": "discharge medications",
                "returned": 0, "passages": [],
                "note": "no discharge note found for this admission"}

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    ordered = _ordered_sections(parse_note(text), normalised)

    dedicated = [entry for entry in ordered if entry[0] in _DEDICATED_MED_SECTIONS]
    chosen = dedicated or [
        entry for entry in ordered if entry[0] in _FALLBACK_MED_SECTIONS
    ]

    passages: list[dict[str, Any]] = []
    ordinals: dict[str, int] = {}
    seen_bodies: set[str] = set()
    for name, body in chosen:
        key = _norm_body(body)
        if key in seen_bodies:
            continue
        seen_bodies.add(key)
        ordinals[name] = ordinals.get(name, 0) + 1
        passages.append({
            "id": f"{note_id}_{name}_{ordinals[name]}",
            "section": name,
            "text": body,
            "retrieval": "deterministic",
        })

    if not passages:
        return _no_medication_sections_payload(hadm_id, note_id)
    return {"hadm_id": hadm_id, "query": "discharge medications",
            "returned": len(passages), "passages": passages}


async def discharge_medications(hadm_id: int) -> RetrievalResult | ToolError:
    """Retrieve the discharge medications section and the discharge instructions
    section, in that order. Use when the medication list must be complete.

    Two sections, because in these notes they carry different halves of one fact. The
    medications section is often a bare list of drug names — "On transfer, Celebrex,
    Coumadin, Colace, …" — while the doses and frequencies sit in the instructions
    ("Celebrex 200 mg daily … Toprol-XL 50 mg daily"). Reading only the first produced
    a grid of drug names with no doses, and the extraction could not have improved it:
    the numbers were never in its context.

    Both sections come back whole and labelled, so the extraction reconciles names
    with doses and `parse_medications` then verifies every value against this same
    text — a dose survives only if it appears in a passage here.

    Args:
        hadm_id: MIMIC-IV hospital admission id.
    """
    payload = await asyncio.to_thread(_search_medication_sections, hadm_id)
    try:
        return validate_retrieval_result(payload)
    except ToolContractError:
        _LOG.error("discharge_medications: produced payload outside its tool contract")
        return _error(hadm_id, "invalid_tool_response", "Retrieval response was malformed.")
