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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `services.mcp.tools.retrieval` is shadowed by the function of the same name
# (tools/__init__ re-exports it), so import the module explicitly.
rs = importlib.import_module("services.mcp.tools.retrieval")

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
            values = [0.1] * rs.OUTPUT_DIMENSIONALITY

        class _Resp:
            embeddings = [_Emb()]

        return _Resp()


class _FakeEmbedClient:
    def __init__(self):
        self.models = _FakeEmbeddings()


def _run(coro):
    return asyncio.run(coro)


class _FakeAdmissions:
    """The admission lookup, which is only consulted when nothing was found."""

    def __init__(self, known=True, blows_up=False):
        self._known = known
        self._blows_up = blows_up
        self.asked: list[int] = []

    def exists(self, hadm_id):
        self.asked.append(hadm_id)
        if self._blows_up:
            raise RuntimeError("warehouse unreachable")
        return self._known


def _admissions(known=True, blows_up=False):
    """Patch the lookup, and hand the fake back so a test can read `asked`.

    Patched rather than left real in every test: without credentials the real source
    raises, which lands on the same branch as a genuine failure, so a test that did
    not patch it would pass while exercising the wrong path.
    """
    fake = _FakeAdmissions(known=known, blows_up=blows_up)
    return patch.object(rs, "_admission_source", lambda: fake), fake


def _run_search(endpoint: _FakeEndpoint, note_texts: dict | None = None,
                known_admission: bool = True, **kwargs):
    """Run rag_search with all cloud touchpoints faked."""
    embed = _FakeEmbedClient()
    note_texts = note_texts or {}

    def fake_fetch(note_ids, hadm_id):
        # Only return texts that were explicitly provided — a missing note_id
        # stays missing so the missing_text error path can be exercised.
        return {nid: note_texts[nid] for nid in note_ids if nid in note_texts}

    admission_patch, _ = _admissions(known=known_admission)
    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: embed), \
         patch.object(rs, "_fetch_texts", fake_fetch), \
         admission_patch:
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
    """No neighbors -> empty passages, not an error.

    The result says which empty this is. An admission we serve that returned nothing
    is an empty record; the same shape for an admission that does not exist would be
    a claim about a patient (gap 6), so the note is part of the contract now.
    """
    endpoint = _FakeEndpoint([])
    result = _run_search(endpoint, hadm_id=HADM_A, query="nothing should match")
    assert result == {"hadm_id": HADM_A, "query": "nothing should match",
                      "returned": 0, "passages": [],
                      "note": "no passages were retrieved for this admission"}


def test_bad_hadm_id_is_structured_error():
    """Non-positive hadm_id is a structured error, not a crash."""
    result = _run_search(_FakeEndpoint([]), hadm_id=-5, query="sepsis")
    assert result["error"] == "bad_request"
    assert "hadm_id" in result["message"]


def test_empty_query_is_structured_error():
    result = _run_search(_FakeEndpoint([]), hadm_id=HADM_A, query="   ")
    assert result["error"] == "bad_request"


def test_top_k_out_of_range_is_structured_error():
    """The guard behind the declared bound.

    Over MCP the populated schema refuses this before the tool body runs, so this asserts the
    path a caller that bypasses the schema takes — an in-process call, or a transport that does
    not validate. Called directly on purpose: through the SDK it never reaches the check.
    """
    result = _run_search(_FakeEndpoint([]), hadm_id=HADM_A, query="sepsis", top_k=50)
    assert result["error"] == "bad_request"


def test_missing_bigquery_text_errors_not_drops(caplog):
    """A returned ID missing from BigQuery must error, not silently drop.

    The id names a note, so it identifies a patient: it goes to the log with the
    table name, and the message carries the sentence (gap 4).
    """
    endpoint = _FakeEndpoint([
        _FakeNeighbor("99999999-DS-1_brief_hospital_course_1", 0.25),
    ])
    with caplog.at_level("WARNING"):
        result = _run_search(endpoint, note_texts={}, hadm_id=HADM_A, query="sepsis")

    assert result["error"] == "missing_text"
    assert "99999999-DS-1" not in result["message"]
    assert "99999999-DS-1" in caplog.text


def test_unparsed_datapoint_id_is_structured_error(caplog):
    """An index id no section token matches (stale vocabulary, foreign
    datapoint) must be a structured error, never a citation with no text."""
    endpoint = _FakeEndpoint([
        _FakeNeighbor("12345-DS-9_unknown_section_1", 0.25),
    ])
    with caplog.at_level("WARNING"):
        result = _run_search(endpoint, note_texts={}, hadm_id=HADM_A, query="sepsis")

    assert result["error"] == "unparsed_datapoint"
    assert "12345-DS-9_unknown_section_1" not in result["message"]
    assert "12345-DS-9_unknown_section_1" in caplog.text


def test_whole_note_fallback_is_tagged_and_exact_chunk_is_not():
    """When the chunk id cannot be reproduced (chunker drift) the passage falls
    back to whole-note text and must SAY so via granularity="note"; a passage
    whose chunk did reproduce carries no such tag."""
    # "real note text" has no parseable sections, so the id can't reproduce.
    endpoint = _FakeEndpoint([
        _FakeNeighbor("13479418-DS-24_brief_hospital_course_2", 0.27),
    ])
    result = _run_search(endpoint, note_texts={"13479418-DS-24": "real note text"},
                         hadm_id=HADM_A, query="sepsis treatment")
    passage = result["passages"][0]
    assert passage["text"] == "real note text"
    assert passage["granularity"] == "note"

    note = "Brief Hospital Course:\nDiuresed and improved steadily.\n"
    endpoint = _FakeEndpoint([
        _FakeNeighbor("13479418-DS-24_brief_hospital_course_1", 0.27),
    ])
    result = _run_search(endpoint, note_texts={"13479418-DS-24": note},
                         hadm_id=HADM_A, query="sepsis treatment")
    passage = result["passages"][0]
    assert passage["text"] == "Diuresed and improved steadily."
    assert "granularity" not in passage


def test_sections_tool_marks_passages_deterministic_not_scored():
    """rag_search_sections passages are re-parsed, not retrieved: they must be
    marked deterministic instead of carrying a fabricated score of 1.0."""
    note = "Brief Hospital Course:\nDiuresed.\n\nDischarge Diagnosis:\nCHF.\n"
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("13479418-DS-24", note)):
        result = _run(rs.rag_search_sections(HADM_A))
    assert result["returned"] == 2
    assert [p["section"] for p in result["passages"]] == [
        "brief_hospital_course", "discharge_diagnosis"]
    for passage in result["passages"]:
        assert passage["retrieval"] == "deterministic"
        assert "score" not in passage


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

    def fake_fetch(note_ids, hadm_id):
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
    # Passage text is the exact SECTION chunk (re-chunked deterministically),
    # not the whole note — other sections must not leak into the citation.
    assert "Docusate Sodium" in result["passages"][0]["text"]
    assert "Some home meds" not in result["passages"][0]["text"]
    assert "Follow up with PCP" not in result["passages"][0]["text"]
    # The fallback embeds the section's body, not the original short phrase:
    # the last embed call's contents carry the section text.
    assert embed.models.last_contents is not None
    assert "Docusate Sodium" in embed.models.last_contents[0]
    # The response `query` stays the caller's original phrase (the UI renders it
    # as the source-card title) — the section body must NOT leak into it.
    assert result["query"] == "discharge medications"


def test_meds_query_retries_when_intended_section_not_rank_one():
    """A section-intent query whose top hit is a DIFFERENT section must retry
    with the section's text so the intended section ranks first (2026-08-24
    enhancement: recall@1 82% -> ~100%)."""
    course_neighbor = _FakeNeighbor("13219116-DS-18_brief_hospital_course_1", 0.25)
    meds_neighbor = _FakeNeighbor("13219116-DS-18_discharge_medications_1", 0.28)
    endpoint = _Recorder([course_neighbor], [meds_neighbor])
    embed = _FakeEmbedClient()

    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: embed), \
         patch.object(rs, "_fetch_texts", lambda note_ids, hadm_id: {nid: _NOTE for nid in note_ids}), \
         patch.object(rs, "_fetch_note", lambda hadm: _NOTE):
        result = _run(rs.rag_search(hadm_id=23613002, query="medications"))

    assert endpoint.calls == 2
    assert result["returned"] == 1
    assert result["passages"][0]["section"] == "discharge_medications"
    # The retry embedded the section's body text, not the short phrase.
    assert embed.models.last_contents is not None
    assert "Docusate Sodium" in embed.models.last_contents[0]
    assert result["query"] == "medications"


def test_anchored_retry_fires_at_most_once():
    """The retry is structurally single-shot (ECC-20): a section body that
    itself reads like a section-intent query must NOT re-anchor — every extra
    level is a billed embed + index query + BigQuery fetch."""
    course_neighbor = _FakeNeighbor("13219116-DS-18_brief_hospital_course_1", 0.25)
    # Both rounds rank the WRONG section first; the note's meds body contains
    # the phrase "discharge medications", so pre-fix the retry re-anchors and
    # recurses without bound.
    endpoint = _Recorder([course_neighbor], [course_neighbor])
    note = ("Brief Hospital Course:\nRecovered well.\n\n"
            "Discharge Medications:\nsee the discharge medications list.\n")

    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts", lambda note_ids, hadm_id: {nid: note for nid in note_ids}), \
         patch.object(rs, "_fetch_note", lambda hadm: note):
        result = _run(rs.rag_search(hadm_id=23613002, query="medications"))

    assert endpoint.calls == 2  # original + exactly one anchored retry
    assert result["returned"] == 1
    assert result["query"] == "medications"


def test_empty_retry_does_not_discard_real_hits():
    """A wrong-rank retry that comes back EMPTY must not replace the original
    non-empty result (ECC-24) — fall through to the real neighbors."""
    course_neighbor = _FakeNeighbor("13219116-DS-18_brief_hospital_course_1", 0.25)
    endpoint = _Recorder([course_neighbor], [])

    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts", lambda note_ids, hadm_id: {nid: _NOTE for nid in note_ids}), \
         patch.object(rs, "_fetch_note", lambda hadm: _NOTE):
        result = _run(rs.rag_search(hadm_id=23613002, query="medications"))

    assert endpoint.calls == 2
    assert result["returned"] == 1
    assert result["passages"][0]["section"] == "brief_hospital_course"


def test_non_section_query_still_returns_empty():
    """A query with no section intent and 0 hits stays empty (no fabrication)."""
    endpoint = _Recorder([], [])
    admission_patch, _ = _admissions(known=True)
    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts", lambda note_ids, hadm_id: {}), \
         admission_patch:
        result = _run(rs.rag_search(hadm_id=23613002, query="what did the nurse chart say"))

    assert result == {"hadm_id": 23613002, "query": "what did the nurse chart say",
                      "returned": 0, "passages": [],
                      "note": "no passages were retrieved for this admission"}
    assert endpoint.calls == 1


# --- Section-chunk text (2026-08-24) ----------------------------------------
#
# Root cause: passages returned the WHOLE note as text even though the index
# matched a section-level chunk, so citations leaked unrelated sections and the
# agent read the full note in every passage. Fix: deterministically re-chunk
# each fetched note and return the exact section chunk the id names.


_MEDS_NOTE = ("HOSPITAL COURSE: The patient recovered well.\n\n"
              "DISCHARGE DIAGNOSES: 1. Pneumonia.\n\n"
              "MEDICATIONS: Tylenol 650 mg q.6h., Lasix 80 mg daily.")


# Every section the summary is made of, so the fixed order and the exact-text
# property can both be asserted. The headings are deliberately the document's
# words rather than the storage names: the mapping between the two is part of
# what this pins.
_SUMMARY_NOTE = (
    "HISTORY OF PRESENT ILLNESS: The patient presented with a cough.\n\n"
    "HOSPITAL COURSE: The patient recovered well.\n\n"
    "DISCHARGE DIAGNOSES: 1. Pneumonia.\n\n"
    "DISCHARGE INSTRUCTIONS: Complete the antibiotic course."
)


def test_passage_text_is_the_section_chunk_not_the_whole_note():
    """rag_search returns the exact chunk text for the matched id."""
    endpoint = _FakeEndpoint([
        _FakeNeighbor("MT-1-DS_discharge_medications_1", 0.21),
    ])
    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts",
                      lambda note_ids, hadm_id: {nid: _MEDS_NOTE for nid in note_ids}):
        result = _run(rs.rag_search(hadm_id=90000015, query="medications"))

    assert result["returned"] == 1
    text = result["passages"][0]["text"]
    assert "Tylenol" in text
    assert "HOSPITAL COURSE" not in text
    assert "Pneumonia" not in text


def test_search_sections_returns_every_section_in_document_order():
    """`_search_sections` is lossless: one passage per parsed section, in note order.

    The contract changed on 2026-11-03. It used to re-chunk the note and return the first
    chunk of each of four whitelisted sections, which lost every other section of the note,
    lost the tail of any section longer than a chunk, and returned the four in the
    whitelist's order rather than the note's. A summary built on that could not mention the
    operation the admission was for, and its citation numbers pointed at passages in an
    order the note does not read in.

    So what is asserted here is coverage and order, not a chosen four: every section the
    parser recognises appears, carrying its own text and nothing else, in the sequence the
    document states them.
    """
    with patch.object(rs, "_fetch_note_row",
                      lambda hadm: ("MT-1-DS", _SUMMARY_NOTE)):
        result = rs._search_sections(90000015)

    sections = [p["section"] for p in result["passages"]]
    assert sections == [
        "history_of_present_illness", "brief_hospital_course",
        "discharge_diagnosis", "discharge_instructions"]
    assert result["returned"] == len(sections)
    # Each passage is that one section and nothing else: the hospital course passage must
    # not carry the diagnosis, and the diagnosis passage must not carry the instructions.
    by_section = {p["section"]: p["text"] for p in result["passages"]}
    assert "recovered well" in by_section["brief_hospital_course"]
    assert "Pneumonia" not in by_section["brief_hospital_course"]
    assert "Pneumonia" in by_section["discharge_diagnosis"]
    assert "antibiotic" not in by_section["discharge_diagnosis"]
    # Ids are stable and carry the section, which is what the citation layer reads.
    assert [p["id"] for p in result["passages"]] == [
        "MT-1-DS_history_of_present_illness_1",
        "MT-1-DS_brief_hospital_course_1",
        "MT-1-DS_discharge_diagnosis_1",
        "MT-1-DS_discharge_instructions_1",
    ]
    assert all(p["retrieval"] == "deterministic" for p in result["passages"])


_OPERATIVE_NOTE = (
    "ADMISSION DIAGNOSIS: Right hip fracture.\n\n"
    "OPERATIVE PROCEDURES: Hemiarthroplasty of the right hip.\n\n"
    "COMPLICATIONS: None.\n\n"
    "COURSE ON ADMISSION: She was admitted and taken to theatre the next morning.\n\n"
    "DISCHARGE DIAGNOSIS: Right hip fracture, status post hemiarthroplasty.\n\n"
    "DISCHARGE MEDICATIONS: Oxycodone 5 mg every six hours as needed.\n\n"
    "DISCHARGE INSTRUCTIONS: Weight bear as tolerated. Follow up in two weeks."
)


def test_search_sections_keeps_the_operative_and_complication_blocks():
    """The sections a summary could not previously reach are the ones it needed most.

    Operative procedures, complications and the admission course sit between the diagnosis
    and the discharge plan in this note. None of them was in the old whitelist, so all
    three were absent from the tool result and a summary could not say the patient had an
    operation or that it went without complication.
    """
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-3-DS", _OPERATIVE_NOTE)):
        result = rs._search_sections(90000018)

    sections = [p["section"] for p in result["passages"]]
    assert sections == [
        "admission_diagnosis",
        "major_procedure",
        "complications",
        "brief_hospital_course",
        "discharge_diagnosis",
        "discharge_medications",
        "discharge_instructions",
    ]
    by_section = {p["section"]: p["text"] for p in result["passages"]}
    assert "Hemiarthroplasty" in by_section["major_procedure"]
    # "COMPLICATIONS: None." is its own section, not the tail of the procedure list.
    assert by_section["complications"] == "None."
    assert "taken to theatre" in by_section["brief_hospital_course"]


def test_search_sections_keeps_two_diagnosis_blocks_apart():
    """Admission and discharge diagnoses are two statements, not one key.

    They shared a canonical name until 2026-11-03, so a note documenting both produced one
    section whose heading was whichever came first. The passage carry the ordinal, so a
    citation can name the one it quotes.
    """
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-3-DS", _OPERATIVE_NOTE)):
        result = rs._search_sections(90000018)

    by_section = {p["section"]: p["text"] for p in result["passages"]}
    assert by_section["admission_diagnosis"] == "Right hip fracture."
    assert "status post hemiarthroplasty" in by_section["discharge_diagnosis"]


def test_search_sections_numbers_repeated_sections_without_overwriting():
    """Two sections of the same name both survive, with incrementing ordinals."""
    note = (
        "DISCHARGE DIAGNOSIS: Pneumonia.\n\n"
        "DISCHARGE DIAGNOSIS: Acute kidney injury.\n\n"
        "HOSPITAL COURSE: Treated with antibiotics."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-4-DS", note)):
        result = rs._search_sections(90000019)

    assert [p["id"] for p in result["passages"]] == [
        "MT-4-DS_discharge_diagnosis_1",
        "MT-4-DS_discharge_diagnosis_2",
        "MT-4-DS_brief_hospital_course_1",
    ]
    assert result["passages"][0]["text"] == "Pneumonia."
    assert result["passages"][1]["text"] == "Acute kidney injury."


def test_search_sections_keeps_the_text_before_the_first_heading():
    """A discharge summary's opening lines are note text, not a preamble to discard."""
    note = (
        "The patient is a 71-year-old woman admitted with pneumonia.\n\n"
        "HOSPITAL COURSE: Treated with antibiotics."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-5-DS", note)):
        result = rs._search_sections(90000020)

    assert result["passages"][0]["section"] == "history_of_present_illness"
    assert result["passages"][0]["id"] == "MT-5-DS_history_of_present_illness_1"
    assert "71-year-old woman" in result["passages"][0]["text"]


def test_search_sections_returns_a_headingless_note_whole():
    """A narrative note with no recognised headings is one passage, not nothing.

    The old path returned an empty result for these, so three of the eighty-seven notes in
    the cohort reached the formatting turn as "no summary sections found" while holding a
    full narrative.
    """
    note = ("The patient presented with two days of pleuritic chest pain. She was admitted "
            "for observation and discharged home the following day.")
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-6-DS", note)):
        result = rs._search_sections(90000021)

    assert result["returned"] == 1
    passage = result["passages"][0]
    assert passage["section"] == "discharge_summary"
    assert passage["id"] == "MT-6-DS_discharge_summary_1"
    assert "pleuritic chest pain" in passage["text"]


def test_search_sections_skips_a_fully_redacted_section():
    """A section that is only de-identification placeholders is not a reading of anything."""
    note = (
        "HISTORY OF PRESENT ILLNESS: ____________\n\n"
        "HOSPITAL COURSE: Recovered well."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-7-DS", note)):
        result = rs._search_sections(90000022)

    assert [p["section"] for p in result["passages"]] == ["brief_hospital_course"]


_IDENTICAL_DIAGNOSES = (
    "ADMISSION DIAGNOSES: 1. Neck pain with right upper extremity radiculopathy.\n\n"
    "DISCHARGE DIAGNOSES: 1. Neck pain with right upper extremity radiculopathy.\n\n"
    "HOSPITAL COURSE: She was admitted and taken to theatre."
)

_DIFFERENT_DIAGNOSES = (
    "ADMISSION DIAGNOSES: 1. Neck pain with right upper extremity radiculopathy.\n\n"
    "DISCHARGE DIAGNOSES: 1. Cervical spondylosis with stenosis at C5-C6.\n\n"
    "HOSPITAL COURSE: She was admitted and taken to theatre."
)


def test_an_identical_diagnosis_block_is_returned_once_as_the_discharge_diagnosis():
    """One block under two headings is one passage, and the final heading is the one kept.

    Measured on 90000069 (2026-11-03): the note lists the same two diagnoses under
    `ADMISSION DIAGNOSES` and `DISCHARGE DIAGNOSES`, so the tool returned two identical
    passages and the summary cited both for one sentence — `^[1]` and `^[2]` naming the
    same two diagnoses. A summary is answered by the diagnosis the patient left with, so
    the surviving passage takes that name.
    """
    with patch.object(rs, "_fetch_note_row",
                      lambda hadm: ("MT-8-DS", _IDENTICAL_DIAGNOSES)):
        result = rs._search_sections(90000023)

    assert [p["section"] for p in result["passages"]] == [
        "discharge_diagnosis", "brief_hospital_course"]
    # The id is recomputed with the ordinal of the name that survived, so nothing
    # downstream sees a `discharge_diagnosis` passage numbered as an admission one.
    assert result["passages"][0]["id"] == "MT-8-DS_discharge_diagnosis_1"


def test_two_different_diagnosis_blocks_are_both_kept():
    """A deduplication, not a consolidation: different text is a different statement."""
    with patch.object(rs, "_fetch_note_row",
                      lambda hadm: ("MT-9-DS", _DIFFERENT_DIAGNOSES)):
        result = rs._search_sections(90000024)

    assert [p["section"] for p in result["passages"]] == [
        "admission_diagnosis", "discharge_diagnosis", "brief_hospital_course"]
    by_section = {p["section"]: p["text"] for p in result["passages"]}
    assert "radiculopathy" in by_section["admission_diagnosis"]
    assert "stenosis" in by_section["discharge_diagnosis"]


def test_a_duplicate_under_any_other_heading_keeps_its_first_label():
    """Only the admission/discharge diagnosis pair is renamed; nothing else is upgraded."""
    note = (
        "SOCIAL HISTORY: Lives alone.\n\n"
        "FAMILY HISTORY: Lives alone."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-10-DS", note)):
        result = rs._search_sections(90000025)

    assert [p["section"] for p in result["passages"]] == ["social_history"]


# --- /meds: the medication text, from wherever this note keeps it -----------------


def test_medication_sections_returns_every_dedicated_section_whole():
    """No first-chunk truncation, and the instructions come too.

    The old path asked the chunker for one chunk of each of two sections, so a medication
    list that ran onto a second chunk lost its tail — the failure mode a `/meds` grid shows
    as a patient on four drugs when the note lists nine.
    """
    note = (
        "DISCHARGE MEDICATIONS: On transfer, Celebrex, Coumadin, Colace, Synthroid.\n\n"
        "DISCHARGE INSTRUCTIONS: Celebrex 200 mg daily. Coumadin 5 mg daily. "
        "Colace 100 mg twice daily. Synthroid 75 mcg daily."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-11-DS", note)):
        result = rs._search_medication_sections(90000014)

    assert [p["section"] for p in result["passages"]] == [
        "discharge_medications", "discharge_instructions"]
    by_section = {p["section"]: p["text"] for p in result["passages"]}
    assert "Synthroid" in by_section["discharge_medications"]
    # Every dose survives, not just the first chunk's worth.
    for dose in ("200 mg", "5 mg", "100 mg", "75 mcg"):
        assert dose in by_section["discharge_instructions"]


def test_medication_sections_falls_back_to_the_narrative_when_there_is_no_med_section():
    """The 90000069 shape: a surgical note whose prescriptions sit at the end of the course."""
    note = (
        "ADMISSION DIAGNOSIS: Neck pain.\n\n"
        "HOSPITAL COURSE: She underwent anterior cervical discectomy. Postoperatively "
        "she was managed with IV morphine and treated with Decadron. We will send her "
        "home with Medrol, Darvocet and Flexeril."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-2113-DS", note)):
        result = rs._search_medication_sections(90000069)

    assert [p["section"] for p in result["passages"]] == ["brief_hospital_course"]
    assert "send her home with Medrol" in result["passages"][0]["text"]


def test_medication_sections_falls_back_to_a_headingless_note():
    note = "She was discharged home on warfarin 5 mg daily with follow-up in one week."
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-12-DS", note)):
        result = rs._search_medication_sections(90000013)

    assert [p["section"] for p in result["passages"]] == ["discharge_summary"]


def test_a_dedicated_medication_section_wins_over_the_narrative():
    """Both are present: the list is the answer, and the course is not mixed into it.

    Mixing them is how a `/meds` grid ends up listing IV morphine as something to take at
    home, which is the extraction rule this path exists to make possible.
    """
    note = (
        "DISCHARGE MEDICATIONS: Warfarin 5 mg daily.\n\n"
        "HOSPITAL COURSE: Managed with IV morphine overnight."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-13-DS", note)):
        result = rs._search_medication_sections(90000012)

    assert [p["section"] for p in result["passages"]] == ["discharge_medications"]


def test_a_transfer_list_is_not_read_beside_the_discharge_list():
    """An arrival list is what the patient came in on, so the dedicated pass excludes it.

    In a transitional-care note, `MEDICATIONS ON TRANSFER` is the list carried in from the
    acute hospital, and a drug on it may have been stopped before discharge — 90000015's
    Lovenox was discontinued the day after transfer, and niacin was dropped from the
    discharge regimen entirely. Read beside the discharge list, both reached the grid with
    no dose, because an arrival list states names and not regimens (2026-11-03).
    """
    note = (
        "MEDICATIONS ON TRANSFER: Celebrex, Coumadin, Colace, Lovenox, niacin.\n\n"
        "DISCHARGE MEDICATIONS: Coumadin 5 mg daily, Colace 100 mg twice daily.\n\n"
        "DISCHARGE INSTRUCTIONS: Continue Coumadin 5 mg daily and Colace 100 mg twice "
        "daily. Lovenox was discontinued."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-15-DS", note)):
        result = rs._search_medication_sections(90000015)

    assert [p["section"] for p in result["passages"]] == [
        "discharge_medications", "discharge_instructions"]
    assert "Lovenox" not in " ".join(p["text"] for p in result["passages"]
                                     if p["section"] != "discharge_instructions")


def test_a_note_whose_only_list_is_the_transfer_list_still_reports_it():
    """Fallback, not exclusion: with no discharge section it is the only text there is."""
    note = (
        "MEDICATIONS ON TRANSFER: Coumadin, Colace.\n\n"
        "HISTORY OF PRESENT ILLNESS: Admitted for rehabilitation."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-16-DS", note)):
        result = rs._search_medication_sections(90000010)

    assert [p["section"] for p in result["passages"]] == [
        "medications_on_transfer", "history_of_present_illness"]


def test_medication_sections_reports_an_honest_empty():
    """A note with no medication section and nothing a prescription is written in.

    The diagnosis and the complications block are the whole note: neither is a place a
    discharge prescription is stated, so there is nothing to cite and the tool says so
    rather than returning a passage the extraction would then mine for drug names.
    """
    note = (
        "DISCHARGE DIAGNOSIS: Symptomatic thyroid goiter.\n\n"
        "COMPLICATIONS: None."
    )
    with patch.object(rs, "_fetch_note_row", lambda hadm: ("MT-14-DS", note)):
        result = rs._search_medication_sections(90000011)

    assert result["returned"] == 0
    assert result["passages"] == []
    assert "no discharge medication section" in result["note"]


def test_meds_fallback_keeps_hadm_restrict():
    """The fallback re-query must still apply the hadm restrict server-side."""
    endpoint = _Recorder([], [_FakeNeighbor("13219116-DS-18_discharge_medications_1", 0.28)])
    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts", lambda note_ids, hadm_id: {nid: _NOTE for nid in note_ids}), \
         patch.object(rs, "_fetch_note", lambda hadm: _NOTE):
        _run(rs.rag_search(hadm_id=23613002, query="discharge medications"))
    assert endpoint.calls == 2
    assert endpoint.last_filter is not None
    assert endpoint.last_filter[0].allow_tokens == ["23613002"]


# --- Cluster F: cross-patient isolation & retrieval correctness -------------


class _FakeBQ:
    """Records the SQL and returns canned rows from .query().result()."""

    def __init__(self, rows):
        self.rows = rows
        self.last_query = None

    def query(self, sql, job_config=None):
        self.last_query = sql

        class _Job:
            def __init__(self, rows):
                self._rows = rows

            def result(self):
                return self._rows

        return _Job(self.rows)


def test_fetch_texts_rechecks_hadm_and_raises_on_foreign_note():
    """ECC-19: a resolved note belonging to ANOTHER admission is a hard
    IsolationViolation at the BigQuery layer, never a served passage."""
    bq = _FakeBQ([
        {"note_id": "MT-1-DS", "hadm_id": 90000015, "text": "mine"},
        {"note_id": "MT-9-DS", "hadm_id": 90000099, "text": "someone else's"},
    ])
    with patch.object(rs, "_bigquery", lambda: bq):
        with pytest.raises(rs.IsolationViolation) as exc:
            rs._fetch_texts(["MT-1-DS", "MT-9-DS"], 90000015)
    assert "MT-9-DS" in str(exc.value)
    assert "90000015" in str(exc.value)


def test_fetch_texts_passes_when_all_rows_match():
    bq = _FakeBQ([{"note_id": "MT-1-DS", "hadm_id": 90000015, "text": "mine"}])
    with patch.object(rs, "_bigquery", lambda: bq):
        texts = rs._fetch_texts(["MT-1-DS"], 90000015)
    assert texts == {"MT-1-DS": "mine"}
    assert "hadm_id" in bq.last_query  # the re-check column is fetched


def test_isolation_violation_is_a_structured_error(caplog):
    """End to end: an isolation breach surfaces as a structured error the
    agent can read, not an exception.

    The exception names the foreign note, and a note id is "{subject_id}-DS-{note}"
    — so the one place in this layer that handles another patient's data is the one
    place the message must say the least. The id is in the log; the message is not
    (gap 4).
    """
    endpoint = _FakeEndpoint([
        _FakeNeighbor("MT-9-DS_brief_hospital_course_1", 0.3),
    ])

    def raising_fetch(note_ids, hadm_id):
        raise rs.IsolationViolation("note MT-9-DS belongs to another admission")

    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: _FakeEmbedClient()), \
         patch.object(rs, "_fetch_texts", raising_fetch), \
         caplog.at_level("WARNING"):
        result = _run(rs.rag_search(hadm_id=90000015, query="course"))

    assert result["error"] == "isolation_violation"
    assert "MT-9-DS" not in result["message"]
    assert "another admission" not in result["message"]
    # It still says what a caller needs: which admission, and that nothing served.
    assert result["hadm_id"] == 90000015
    assert "did not belong" in result["message"]
    assert "MT-9-DS" in caplog.text


def test_note_row_queries_are_deterministic():
    """ECC-27: both single-note queries order by note_id so a multi-note
    admission cites the same note on every call."""
    bq = _FakeBQ([])
    with patch.object(rs, "_bigquery", lambda: bq):
        rs._fetch_note_row(90000015)
        assert "ORDER BY note_id" in bq.last_query
        rs._fetch_note(90000015)
        assert "ORDER BY note_id" in bq.last_query


def test_bool_hadm_id_rejected_everywhere():
    """ECC-28: bool is an int subclass and must be rejected by every entry
    point through the shared validator."""
    from services.mcp.tools import prediction as pr
    from services.mcp.tools._validation import valid_hadm_id

    assert not valid_hadm_id(True)
    assert not valid_hadm_id(False)
    assert not valid_hadm_id(0)
    assert not valid_hadm_id(-1)
    assert not valid_hadm_id("90000015")
    assert valid_hadm_id(90000015)

    # rag_search
    result = _run_search(_FakeEndpoint([]), hadm_id=True, query="sepsis")
    assert result["error"] == "bad_request"

    # rag_search_sections (previously had NO validation at all)
    for bad in (True, 0, -5, "x"):
        result = rs._search_sections(bad)
        assert result["error"] == "bad_request"

    # predict_readmission (previously had NO validation at all)
    for bad in (True, 0, -5):
        result = pr._predict(bad)
        assert result["error"] == "bad_request"


# --- unknown admission vs empty record (gap 6) ---------------------------------
#
# The prompt tells the model an all-empty result is a real answer, so an empty result
# for an admission that does not exist turns a typo into a statement about a patient.
# Prediction already answers `unknown_patient`; these pin the same meaning onto both
# retrieval tools, and pin the borrow of its source rather than a second opinion.

def test_an_unknown_admission_is_an_error_not_an_empty_record():
    result = _run_search(_FakeEndpoint([]), hadm_id=90000001, query="sepsis",
                         known_admission=False)

    assert result["error"] == "unknown_patient"
    assert "passages" not in result, "an error is not a record with nothing in it"


def test_an_unknown_admission_is_an_error_for_sections_too():
    admission_patch, _ = _admissions(known=False)
    with patch.object(rs, "_fetch_note_row", lambda hadm: (None, None)), admission_patch:
        result = _run(rs.rag_search_sections(90000001))

    assert result["error"] == "unknown_patient"


def test_an_admission_that_exists_without_a_note_is_still_an_empty_success():
    """The other side of the distinction: no note is an empty record, not a typo."""
    admission_patch, fake = _admissions(known=True)
    with patch.object(rs, "_fetch_note_row", lambda hadm: (None, None)), admission_patch:
        result = _run(rs.rag_search_sections(90000015))

    assert "error" not in result
    assert result["returned"] == 0
    assert "no discharge note" in result["note"]
    assert fake.asked == [90000015], "the lookup is what told the two cases apart"


def test_an_empty_retrieval_says_so_without_claiming_an_empty_record():
    result = _run_search(_FakeEndpoint([]), hadm_id=HADM_A, query="nothing matches")

    assert "error" not in result
    assert result["returned"] == 0
    assert result["passages"] == []
    assert "no passages were retrieved" in result["note"]


def test_a_failed_lookup_does_not_invent_an_unknown_patient(caplog):
    """A failed lookup is not an answer.

    None is deliberately not False: if the admission lookup itself breaks, the tool
    keeps the empty answer and logs, rather than telling the model the patient does
    not exist — the same failure-must-not-become-a-fact rule as gap 4.
    """
    admission_patch, _ = _admissions(blows_up=True)
    with patch.object(rs, "_fetch_note_row", lambda hadm: (None, None)), \
         admission_patch, caplog.at_level("ERROR"):
        result = _run(rs.rag_search_sections(90000015))

    assert "error" not in result
    assert result["returned"] == 0
    assert "warehouse unreachable" in caplog.text


def test_the_lookup_is_not_consulted_when_there_is_something_to_return():
    """It is a fallback, not a tax: a normal retrieval must not pay for it."""
    endpoint = _FakeEndpoint([
        _FakeNeighbor("13479418-DS-24_brief_hospital_course_2", 0.27),
    ])
    admission_patch, fake = _admissions(known=True)
    embed = _FakeEmbedClient()
    note = "Brief Hospital Course: Diuresed with furosemide."

    with patch.object(rs, "_index_endpoint", lambda: endpoint), \
         patch.object(rs, "_embed_client", lambda: embed), \
         patch.object(rs, "_fetch_texts", lambda ids, hadm: {ids[0]: note}), \
         admission_patch:
        result = _run(rs.rag_search(hadm_id=HADM_A, query="sepsis"))

    assert result["returned"] == 1
    assert fake.asked == [], "a successful retrieval must not call the lookup at all"
