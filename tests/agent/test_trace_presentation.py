"""The trace must show what the reader was shown, not only what the turn asked for.

Stage Gate 2 closed the boundary from the other side. `record_turn_tools` has recorded the
calls since 2026-10-17, and every stage-gate report since has had to reconstruct the
rendered surface by hand from a live run, because the tool calls cannot answer it: a turn
that retrieved twenty-five candidate cohorts and drew a table with no rows in it produces
exactly the tool calls of a turn that drew two. The sentence explaining the empty table is
the only place that difference appears, and it existed nowhere but the payload.

`record_presentation` writes the composed surface: the component names, the comparative
matrix exactly as the renderer received it, and the numbered channel. These tests pin the
three properties that make it worth having — that it carries the matrix's rows AND its
note, that an empty table is recorded as an empty table rather than as an absent record,
and that a process with no span records nothing and fails nothing (rule 1).
"""

from __future__ import annotations

from services.agent.presentation.a2ui import compose_presentation
from services.agent.telemetry import observability

PREDICT_CALL = {
    "name": "predict_readmission",
    "args": {"hadm_id": 90000015},
    "response": {
        "hadm_id": 90000015, "probability": 0.2, "threshold": 0.11, "decision": 1,
        "model_version": "test-v1",
        "top_factors": [{"feature": "prior_inpatient_days", "contribution": 0.2,
                         "direction": "increases"}],
    },
}

SECTIONS_CALL = {
    "name": "rag_search_sections",
    "args": {"hadm_id": 90000015},
    "response": {"query": "sections", "returned": 1, "passages": [{
        "id": "p1", "section": "discharge_diagnosis",
        "text": "Discharge Diagnosis: Osteoarthritis of the right knee.",
    }]},
}

COMPARATOR = {
    "pmid": "34793857",
    "title": "Pre-operative healthcare utilization and readmission after total joint "
             "arthroplasty.",
    "authors": "Creager AE, Kleven AD, et al.",
    "journal": "The Journal of Arthroplasty",
    "pub_date": "2022 Mar 12",
    "reported_rate": "3.4%",
    "rate_kind": "point",
    "cohort_summary": "One institution, 2015-2019, primary total joint arthroplasty.",
    "evidence_tier": "Retrospective Cohort",
    "rate_sentence": "The 30-day readmission rate was 3.4%.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/34793857/",
}


class _Span:
    """A span that keeps what it was told, in the order it was told."""

    def __init__(self) -> None:
        self.metadata: list[dict] = []

    def update(self, **kwargs):
        self.metadata.append(kwargs)


def _presented(comparators: list[dict]) -> dict:
    return compose_presentation(
        "/ehr", "Knee osteoarthritis.^[1]", [SECTIONS_CALL, PREDICT_CALL],
        None, "summarize", is_outpatient=False, comparators=comparators,
    )


def test_the_trace_records_the_table_the_reader_was_shown():
    span = _Span()

    observability.record_presentation(_presented([COMPARATOR]), span=span)

    (update,) = span.metadata
    recorded = update["metadata"]["presentation"]
    assert "ComparativeRiskMatrix" in recorded["components"]
    matrix = recorded["matrix"]
    # The rows, with the figure and the sentence it came from: what was claimed, and the
    # text a reviewer checks the claim against.
    (row,) = matrix["rows"]
    assert row["pmid"] == "34793857"
    assert row["reported_rate"] == "3.4%"
    assert row["rate_kind"] == "point"
    assert row["excerpt"] == "The 30-day readmission rate was 3.4%."
    assert matrix["baseline"]["probability"] == 0.2
    assert matrix["baseline_note"] == ""


def test_an_empty_table_is_recorded_as_empty_with_its_reason():
    """The case two live runs could not be told apart without this.

    Verification can reject every candidate the retrieval returned. What the reader then
    sees is the estimate alone plus a sentence saying no cohort was substituted for it —
    and that sentence is the record of the rejection. A trace that carried only the tool
    calls would show a successful search and no explanation.
    """
    span = _Span()

    observability.record_presentation(_presented([]), span=span)

    recorded = span.metadata[0]["metadata"]["presentation"]
    assert recorded["matrix"]["rows"] == []
    assert "No cohort has been substituted" in recorded["matrix"]["baseline_note"]


def test_the_numbered_channel_travels_with_the_surface():
    """A marker in the trace's output has to be resolvable from the trace.

    The output is the served prose, so `^[1]` appears there; without the channel beside
    it, an operator reading an exported span cannot say what `^[1]` addressed.
    """
    span = _Span()
    presentation = _presented([COMPARATOR])

    observability.record_presentation(presentation, span=span)

    recorded = span.metadata[0]["metadata"]["presentation"]
    assert [source["cite"] for source in recorded["sources"]] == [1, 2]
    assert recorded["sources"][-1]["kind"] == "Literature"
    assert [row["index"] for row in recorded["references"]] == [1, 2]
    assert recorded["references"][-1]["pmid"] == "34793857"


def test_a_turn_with_no_span_records_nothing_and_fails_nothing(monkeypatch):
    """Rule 1: tracing is a sink, never a dependency.

    Reaching the end of this test is the assertion. An unconfigured process is a supported
    state, and the only unacceptable outcome here would be an exception taking the answer
    with it.
    """
    monkeypatch.setattr(observability, "current_turn_span", lambda: None)

    observability.record_presentation(_presented([COMPARATOR]))


def test_a_presentation_that_is_not_a_mapping_is_ignored():
    span = _Span()

    observability.record_presentation(None, span=span)

    assert span.metadata == []
