"""Zero-unit tool results: the annotation the model reads, and the count it does not move.

`holdout_t2_02` cited an absence. Turn 2 called `discharge_medications` for a note with no
such section, which returns `{"returned": 0, "passages": [], "note": "the note has no
discharge_medications section"}`, and wrote:

    The discharge medication section is not documented in the record ^[13]

The rule forbidding it was already in `SYSTEM_PROMPT` ("never attach a citation marker to
an absence"), and `enforce_citations` could not remove the marker because 13 was in range
— it resolved, to the brief hospital course. What the model lacked was any local signal
that the result in front of it owned no number at all, so it reached for the next one it
could see.

`render_numbered` now says so on the payload itself. These tests hold the two halves of
that change apart: the annotation is present and descriptive, and the count is untouched —
zero-unit payloads still consume zero, so no marker anywhere in the conversation shifts.

The annotation is also confined to payloads that carry a citation array. A prediction and
a literature search are answers rather than evidence, and a literature turn's records are
cited by `^[PMID: ...]` — marking them `citable: false` would deny a marker that is valid.
"""

from __future__ import annotations

from services.agent.postprocess import citations

#: The response a note retrieval gives when the note has no such section
#: (`services/mcp/tools/retrieval.py`, `_retrieve_section`).
EMPTY_MEDS = {
    "hadm_id": 90000101,
    "query": "discharge_medications",
    "returned": 0,
    "passages": [],
    "note": "the note has no discharge_medications section",
}

#: One note passage, the unit a follow-up's retrieval adds.
RAG = {
    "query": "discharge medications",
    "returned": 2,
    "passages": [
        {"id": "MT-2616-DS_discharge_medications_1",
         "section": "discharge_medications",
         "text": "1. Aspirin 81 mg daily."},
        {"id": "MT-2616-DS_discharge_medications_2",
         "section": "discharge_medications",
         "text": "2. Metoprolol succinate 25 mg daily."},
    ],
}

#: A result that is an answer rather than evidence: it has no citations array at all.
PREDICTION = {
    "hadm_id": 90000101,
    "probability": 0.112,
    "threshold": 0.11,
    "decision": "above_threshold",
    "drivers": [{"label": "recent ED visits", "contribution": 0.04}],
}

#: A literature search. Its records are cited by `^[PMID: ...]`, so it must never be
#: marked uncitable — the marker is valid on the turn that produced it.
LITERATURE = {
    "query": "readmission risk stratification",
    "articles": [
        {"pmid": "35186616", "title": "Readmission risk models in COPD",
         "journal": "Chest", "year": 2021},
    ],
}


def test_an_absent_section_says_it_publishes_no_citable_unit():
    rendered = citations.render_numbered(EMPTY_MEDS)

    assert rendered["cite"] is None
    assert rendered["citable"] is False
    assert rendered["cite_note"] == citations.ZERO_UNIT_NOTE
    # The tool's own statement of the absence survives; the annotation is additive.
    assert rendered["note"] == EMPTY_MEDS["note"]
    assert rendered["returned"] == 0


def test_a_zero_unit_payload_still_consumes_no_citation_number():
    rendered = citations.render_numbered(EMPTY_MEDS)

    assert citations.passage_total(EMPTY_MEDS) == 0
    assert citations.passage_total(rendered) == 0
    # The annotation must not be mistaken for a unit by the counter that bounds markers:
    # `citable_passages` reads `passages`/`trends`, never `cite`.
    assert citations.citable_passages(rendered) == []


def test_an_annotated_payload_does_not_shift_the_numbering_after_it():
    """The defect's arithmetic: 13 was in range because nothing between was miscounted."""
    conversations = [{"name": "discharge_medications", "response": EMPTY_MEDS},
                     {"name": "rag_search", "response": RAG}]

    assert citations.citation_bound(conversations) == 2
    assert citations.citation_bound(
        [{"name": "discharge_medications",
          "response": citations.render_numbered(EMPTY_MEDS)},
         {"name": "rag_search", "response": RAG}]
    ) == 2


def test_the_payload_the_tool_returned_is_not_mutated():
    citations.render_numbered(EMPTY_MEDS)

    assert EMPTY_MEDS == {
        "hadm_id": 90000101,
        "query": "discharge_medications",
        "returned": 0,
        "passages": [],
        "note": "the note has no discharge_medications section",
    }


def test_a_unit_bearing_payload_is_numbered_exactly_as_before():
    """The regression guard: non-empty results keep their numbers and gain no annotation."""
    rendered = citations.render_numbered(RAG, offset=9)

    assert [passage["cite"] for passage in rendered["passages"]] == [10, 11]
    assert all(passage["cite_header"] == f"[CITATION ^[{passage['cite']}] | "
               f"CLINICAL NOTE SECTION — discharge_medications]"
               for passage in rendered["passages"])
    assert "citable" not in rendered
    assert "cite_note" not in rendered
    assert citations.passage_total(rendered) == 2


def test_a_laboratory_panel_is_numbered_exactly_as_before():
    panel = {
        "hadm_id": 90000101,
        "returned": 2,
        "trends": [
            {"key": "rdw", "label": "RDW", "unit": "%", "decimals": 1,
             "reference_low": 11.5, "reference_high": 14.5,
             "series": [{"day": 0.0, "value": 16.6, "flag": "high"}]},
            {"key": "rbc", "label": "Red blood cells", "unit": "x10^6/uL", "decimals": 2,
             "reference_low": 4.5, "reference_high": 5.9,
             "series": [{"day": 1.4, "value": 3.06, "flag": "low"}]},
        ],
    }

    rendered = citations.render_numbered(panel, offset=0)

    assert [trend["cite"] for trend in rendered["trends"]] == [1, 2]
    assert "citable" not in rendered
    assert citations.passage_total(rendered) == 2


def test_a_result_with_no_citations_array_is_returned_unchanged():
    """A prediction is an answer, not evidence: it has no citation surface to annotate.

    Stamping `citable: false` on every zero-unit payload was the first cut of this
    change, and it reached too far: it also denied a marker that a literature turn is
    entitled to write.
    """
    assert citations.render_numbered(PREDICTION) is PREDICTION
    assert citations.render_numbered(LITERATURE) is LITERATURE
    assert citations.passage_total(PREDICTION) == 0
    assert citations.passage_total(LITERATURE) == 0


def test_a_literature_payload_keeps_its_pmid_markers_resolvable():
    """The payload is untouched, and the PMID it carries still resolves."""
    rendered = citations.render_numbered(LITERATURE, offset=0)
    pmids = citations.literature_pmids([{"name": "search_literature",
                                          "response": rendered}])

    assert pmids == frozenset({"35186616"})
    answer, flags = citations.enforce_citations(
        "Risk models stratify COPD readmission ^[PMID: 35186616].", slots=0, pmids=pmids)
    assert flags == []
    assert "PMID: 35186616" in answer


def test_an_empty_trends_array_is_annotated_like_an_empty_passage_array():
    panel = {"hadm_id": 90000101, "returned": 0, "trends": []}

    rendered = citations.render_numbered(panel)

    assert rendered["cite"] is None
    assert rendered["citable"] is False
    assert rendered["cite_note"] == citations.ZERO_UNIT_NOTE
    assert citations.passage_total(rendered) == 0


def test_a_non_dict_payload_is_still_returned_as_it_arrived():
    assert citations.render_numbered("a raw string") == "a raw string"
