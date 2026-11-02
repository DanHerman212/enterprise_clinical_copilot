"""The comparison table's published cohort is evidence, so the turn has to carry it.

`ComparativeRiskMatrix` draws published cohorts beside this patient's estimate, and it is
built outside the prose citation pipeline: the rows are the comparators the engine verified
against the abstracts it retrieved, not passages the answer cites. So the table names a
journal, a year and a PMID on screen — and without this registration the turn holds no
record of any of them: the references drawer lists `1..4` (the note sections) and the cohort
the whole comparison rests on cannot be opened from the turn that showed it.

The registration is done in the composer rather than in the canvas because this is the layer
that owns what a turn publishes. What the canvas owns is whether the table DRAWS at all,
which is why the sources are read from the composed surface instead of from the comparator
list directly: a turn that drew no table owes no source.

Before Stage Gate 2 this file asserted the same contract against a curated corpus selected
by the admission's phenotype. The corpus is gone; the contract is not. What changes is that
the rows are now whatever verification returned, so the fixtures below are comparators in
the shape the engine emits them (`verify_comparator` output), not corpus entries.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.presentation.a2ui import compose_presentation  # noqa: E402

COMPARATOR_PMID = "34793857"
#: What the table calls the row: the journal and the year. This stays the pill's label.
COMPARATOR_CITATION = "The Journal of Arthroplasty · 2022"
#: The article, which is what the record IS.
COMPARATOR_TITLE = (
    "The Impact of Pre-Operative Healthcare Utilization on Complications, "
    "Readmissions, and Post-Operative Healthcare Utilization Following "
    "Total Joint Arthroplasty."
)
COMPARATOR_AUTHORS = "Creager AE, Kleven AD, Kesimoglu ZN, et al."
COMPARATOR_SUBTITLE = (
    f"The Journal of Arthroplasty. 2022. {COMPARATOR_AUTHORS} · PMID {COMPARATOR_PMID}"
)

#: One comparator as `verify_comparator` leaves it: every field read out of the abstract it
#: came from, and the sentence the rate was taken from kept verbatim.
COMPARATOR = {
    "pmid": COMPARATOR_PMID,
    "title": COMPARATOR_TITLE,
    "authors": COMPARATOR_AUTHORS,
    "journal": "The Journal of Arthroplasty",
    "pub_date": "2022 Mar 12",
    "year": "2022",
    "reported_rate": "3.4%",
    "rate_kind": "point",
    "cohort_summary": "One institution, 2015-2019, primary total joint arthroplasty.",
    "evidence_tier": "Retrospective Cohort",
    "rate_sentence": "The 30-day readmission rate was 3.4%.",
    "url": f"https://pubmed.ncbi.nlm.nih.gov/{COMPARATOR_PMID}/",
}

SECTIONS = [
    ("history_of_present_illness", "History of Present Illness: Knee pain for two years."),
    ("brief_hospital_course", "Brief Hospital Course: Uneventful stay."),
    ("discharge_diagnosis", "Discharge Diagnosis: Osteoarthritis."),
    ("discharge_instructions", "Discharge Instructions: Follow up in two weeks."),
]

#: The prose the model wrote, one marker per section, in the order it read them.
ANSWER = " ".join(
    f"{text.split(':')[0]} was recorded ^[{position}]."
    for position, (_section, text) in enumerate(SECTIONS, start=1)
)

RAG = {
    "query": "summary",
    "passages": [
        {"cite": position, "section": section, "text": text}
        for position, (section, text) in enumerate(SECTIONS, start=1)
    ],
}

PREDICT = {
    "hadm_id": 90000015,
    "probability": 0.2,
    "threshold": 0.11,
    "decision": 1,
    "model_version": "test-v1",
}


def _calls(*, predict: bool = True) -> list[dict]:
    calls = [{"name": "rag_search_sections", "args": {"hadm_id": 90000015},
              "response": RAG}]
    if predict:
        calls.append({"name": "predict_readmission", "args": {"hadm_id": 90000015},
                      "response": PREDICT})
    return calls


def _compose(command: str | None = "summarize", *, predict: bool = True,
             comparators: list[dict] | None = None) -> dict:
    return compose_presentation(
        "/ehr", ANSWER, _calls(predict=predict), None, command,
        comparators=[COMPARATOR] if comparators is None else comparators,
    )


def _matrix_rows(payload: dict) -> list[dict]:
    """The comparison table's rows, out of the envelope's own protocol messages."""
    for message in payload["a2ui"]["messages"]:
        update = message.get("updateComponents")
        if not update:
            continue
        for component in update["components"]:
            if component.get("component") == "ComparativeRiskMatrix":
                return component["rows"]
    return []


def test_the_comparator_is_the_fifth_source_with_everything_a_reader_opens():
    payload = _compose()

    assert [source["cite"] for source in payload["sources"]] == [1, 2, 3, 4, 5]
    comparator = payload["sources"][-1]
    assert comparator["kind"] == "Literature"
    assert comparator["badge"] == "PubMed"
    assert comparator["pmid"] == COMPARATOR_PMID
    # The article, not its journal. A source titled with its journal gave the drawer
    # `5. The Journal of Arthroplasty · 2022` and nothing else — no paper, no authors.
    assert comparator["title"] == COMPARATOR_TITLE
    assert comparator["authors"] == COMPARATOR_AUTHORS
    assert comparator["label"] == f"5 · {COMPARATOR_CITATION}"
    assert comparator["journal_citation"] == (
        f"The Journal of Arthroplasty. 2022. {COMPARATOR_AUTHORS}"
    )
    assert comparator["subtitle"] == COMPARATOR_SUBTITLE
    assert comparator["tier"] == "Retrospective Cohort"
    assert comparator["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{COMPARATOR_PMID}/"
    # The passage is what the popover shows, so it has to be the cohort and the sentence
    # the rate came from rather than a label the reader already has.
    assert "arthroplasty" in comparator["text"].lower()
    assert "readmission rate" in comparator["text"].lower()


def test_the_drawer_gets_the_comparator_as_its_own_row():
    payload = _compose()

    assert [row["index"] for row in payload["references"]] == [1, 2, 3, 4, 5]
    row = payload["references"][-1]
    assert row["source"] == "PubMed"
    assert row["pmid"] == COMPARATOR_PMID
    assert row["title"] == COMPARATOR_TITLE
    assert row["authors"] == COMPARATOR_AUTHORS
    # The drawer renders the title, then this line, then the passage. A row that carried
    # only `5 · The Journal of Arthroplasty · 2022` was the naked row a reader reported.
    assert row["subtitle"] == COMPARATOR_SUBTITLE
    assert row["excerpt"] == payload["sources"][-1]["text"]


def test_the_row_carries_the_number_the_pill_will_show():
    """Phase C draws the pill from the row, so the row has to say which source it is."""
    payload = _compose()

    rows = _matrix_rows(payload)
    assert rows, "the comparison table did not draw"
    assert rows[0]["source_index"] == 5
    assert rows[0]["source_label"] == f"5 · {COMPARATOR_CITATION}"


def test_the_prose_citations_and_the_badge_are_untouched():
    """The registration is additive: it must not renumber what the answer already cites.

    `^[1]..^[4]` address the note passages, and `verified` says the answer rests on this
    patient's record. A published cohort about somebody else's patients must not be able
    to flip either of them by being appended to the list they are decided from.
    """
    payload = _compose()

    assert payload["answer"] == ANSWER
    assert [source["section"] for source in payload["sources"][:4]] == [
        section for section, _text in SECTIONS
    ]
    assert payload["verified"] is True
    assert payload["scope"] == "patient_chart"
    assert payload["scope_suppressed"] is False


def test_a_turn_that_draws_no_table_owes_no_source():
    payload = _compose(command=None)

    assert _matrix_rows(payload) == []
    assert all(source.get("kind") != "Literature" for source in payload["sources"])


def test_a_turn_that_verified_no_cohort_owes_no_source():
    """The empty case is a real outcome now: verification can legitimately return nothing.

    The table still draws — the patient's own estimate is still the answer to the question
    that was asked — and no cohort is substituted for the one that was not found.
    """
    payload = _compose(comparators=[])

    assert _matrix_rows(payload) == []
    assert all(source.get("kind") != "Literature" for source in payload["sources"])


def test_the_comparator_is_registered_once_however_the_surface_repeats_it():
    """Two rows citing the same record are one source, not two numbers.

    Verification de-duplicates by PMID within a turn, and this numbering pass must not
    depend on that staying true: a second number for one PMID would put two pills on the
    same record and shift every source after it.
    """
    from services.agent.presentation.a2ui import register_comparator_sources

    row = {
        "pmid": COMPARATOR_PMID,
        "source": COMPARATOR_CITATION,
        "cohort_summary": "One institution, 2015-2019.",
        "excerpt": "The 30-day readmission rate was 3.4%.",
        "evidence_tier": "Retrospective Cohort",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{COMPARATOR_PMID}/",
    }
    canvas = {"messages": [{"updateComponents": {"components": [{
        "id": "comparison",
        "component": "ComparativeRiskMatrix",
        "rows": [dict(row), dict(row)],
    }]}}]}

    sources, references = register_comparator_sources(canvas, start=5)

    assert [source["pmid"] for source in sources] == [COMPARATOR_PMID]
    assert [source["cite"] for source in sources] == [5]
    assert [reference["index"] for reference in references] == [5]
