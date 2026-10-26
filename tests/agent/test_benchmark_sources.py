"""The comparison table's published cohort is evidence, so the turn has to carry it.

`ComparativeRiskMatrix` draws curated PubMed cohorts beside this patient's estimate, and
it is built outside the prose citation pipeline: the rows come from `literature_benchmarks`,
selected by the admission's phenotype rather than by anything a search returned. So the
table named a journal, a year and a PMID on screen while the turn held no record of any of
them — the references drawer listed `1..4` (the note sections) and the benchmark the whole
comparison rests on could not be opened from the turn that showed it.

The registration is done in the composer rather than in the canvas because this is the
layer that owns what a turn publishes. What the canvas owns is whether the table DRAWS at
all, which is why the sources are read from the composed surface instead of from
`benchmarks_for_phenotype` directly: a condition that was retrieved and not compared owes
no source.

Measured on `90000015` (orthopedic arthroplasty → one curated retrospective cohort).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.a2ui import compose_presentation  # noqa: E402

BENCHMARK_PMID = "34793857"
#: What the table calls the row: the journal and the year. This stays the pill's label.
BENCHMARK_CITATION = "The Journal of Arthroplasty · 2022"
#: The article, which is what the record IS.
BENCHMARK_TITLE = (
    "The Impact of Pre-Operative Healthcare Utilization on Complications, "
    "Readmissions, and Post-Operative Healthcare Utilization Following "
    "Total Joint Arthroplasty."
)
BENCHMARK_AUTHORS = "Creager AE, Kleven AD, Kesimoglu ZN, et al."
BENCHMARK_SUBTITLE = (
    f"The Journal of Arthroplasty. 2022. {BENCHMARK_AUTHORS} · PMID {BENCHMARK_PMID}"
)

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


def _compose(command: str | None = "summarize", *, predict: bool = True) -> dict:
    return compose_presentation("/ehr", ANSWER, _calls(predict=predict),
                                None, command)


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


def test_the_benchmark_is_the_fifth_source_with_everything_a_reader_opens():
    payload = _compose()

    assert [source["cite"] for source in payload["sources"]] == [1, 2, 3, 4, 5]
    benchmark = payload["sources"][-1]
    assert benchmark["kind"] == "Literature"
    assert benchmark["badge"] == "PubMed"
    assert benchmark["pmid"] == BENCHMARK_PMID
    # The article, not its journal. A source titled with its journal gave the drawer
    # `5. The Journal of Arthroplasty · 2022` and nothing else — no paper, no authors.
    assert benchmark["title"] == BENCHMARK_TITLE
    assert benchmark["authors"] == BENCHMARK_AUTHORS
    assert benchmark["label"] == f"5 · {BENCHMARK_CITATION}"
    assert benchmark["journal_citation"] == (
        f"The Journal of Arthroplasty. 2022. {BENCHMARK_AUTHORS}"
    )
    assert benchmark["subtitle"] == BENCHMARK_SUBTITLE
    assert benchmark["tier"] == "Retrospective Cohort"
    assert benchmark["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{BENCHMARK_PMID}/"
    # The passage is what the popover shows, so it has to be the cohort and the sentence
    # the rate came from rather than a label the reader already has.
    assert "arthroplasty" in benchmark["text"].lower()
    assert "readmission rate" in benchmark["text"].lower()


def test_the_drawer_gets_the_benchmark_as_its_own_row():
    payload = _compose()

    assert [row["index"] for row in payload["references"]] == [1, 2, 3, 4, 5]
    row = payload["references"][-1]
    assert row["source"] == "PubMed"
    assert row["pmid"] == BENCHMARK_PMID
    assert row["title"] == BENCHMARK_TITLE
    assert row["authors"] == BENCHMARK_AUTHORS
    # The drawer renders the title, then this line, then the passage. A row that carried
    # only `5 · The Journal of Arthroplasty · 2022` was the naked row a reader reported.
    assert row["subtitle"] == BENCHMARK_SUBTITLE
    assert row["excerpt"] == payload["sources"][-1]["text"]


def test_the_row_carries_the_number_the_pill_will_show():
    """Phase C draws the pill from the row, so the row has to say which source it is."""
    payload = _compose()

    rows = _matrix_rows(payload)
    assert rows, "the comparison table did not draw"
    assert rows[0]["source_index"] == 5
    assert rows[0]["source_label"] == f"5 · {BENCHMARK_CITATION}"


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


def test_a_turn_that_draws_no_table_owes_no_benchmark():
    payload = _compose(command=None)

    assert _matrix_rows(payload) == []
    assert all(source.get("kind") != "Literature" for source in payload["sources"])


def test_the_benchmark_is_registered_once_however_the_surface_repeats_it():
    """Two rows citing the same record are one source, not two numbers.

    A curated corpus cannot list the same cohort twice today. The numbering pass must not
    depend on that staying true: a second number for one PMID would put two pills on the
    same record and shift every source after it.
    """
    from services.agent.note_presentation import register_benchmark_sources

    row = {
        "pmid": BENCHMARK_PMID,
        "source": BENCHMARK_CITATION,
        "cohort_summary": "One institution, 2015-2019.",
        "excerpt": "The 30-day readmission rate was 3.4%.",
        "evidence_tier": "Retrospective Cohort",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{BENCHMARK_PMID}/",
    }
    canvas = {"messages": [{"updateComponents": {"components": [{
        "id": "comparison",
        "component": "ComparativeRiskMatrix",
        "rows": [dict(row), dict(row)],
    }]}}]}

    sources, references = register_benchmark_sources(canvas, start=5)

    assert [source["pmid"] for source in sources] == [BENCHMARK_PMID]
    assert [source["cite"] for source in sources] == [5]
    assert [reference["index"] for reference in references] == [5]


def test_every_curated_benchmark_can_be_cited_as_a_paper():
    """Every row in the corpus, not just the one this admission happens to select.

    The corpus is what the comparison table draws from, and a phenotype that is reached
    tomorrow reaches a row that was never opened. `title` and `authors` fall back to the
    journal line when absent, so a missing one degrades silently: the drawer would show a
    naked journal line again, for that cohort only, and nothing would fail.
    """
    from services.agent.literature_benchmarks import BENCHMARKS, benchmarks_for_phenotype

    assert BENCHMARKS, "the curated corpus is empty"
    for pmid, benchmark in BENCHMARKS.items():
        assert benchmark.title.strip(), f"{pmid} has no article title"
        assert benchmark.authors.strip(), f"{pmid} has no authors"
        assert benchmark.journal.strip(), f"{pmid} has no journal"
        assert benchmark.year.strip(), f"{pmid} has no year"
        # The fields the citations are built from have to survive the trip through the
        # surface, not just exist on the dataclass.
        (row,) = [
            candidate
            for phenotype in {benchmark.phenotype}
            for candidate in benchmarks_for_phenotype(phenotype)
            if candidate["pmid"] == pmid
        ]
        assert row["title"] == benchmark.title
        assert row["authors"] == benchmark.authors
        assert row["journal"] == benchmark.journal
        assert row["year"] == benchmark.year
