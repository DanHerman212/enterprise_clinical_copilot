"""The judge's evidence carries the citation numbers the answer writes.

`_evidence_for` used to hand the judge a flat list of tool payloads with no numbering, so
a marker could only be graded by counting array positions — and with several tool results
in one turn, "the third passage" is not a thing the judge could know. `00e` (2026-10-28)
is the record: `holdout_t5_02` cited `^[3]`, `^[4]` and `^[6]`, the numbers were rewritten
downstream, and the judge resolved the result against the wrong units.

Now every passage and every analyte carries the marker the answer must write for it, and
the counter runs across the whole conversation in the order the tools ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "evaluation" / "agent") not in sys.path:
    sys.path.insert(0, str(REPO / "evaluation" / "agent"))

import judge  # noqa: E402


def _panel(n: int) -> dict:
    return {"hadm_id": 90000101, "returned": n, "trends": [
        {"key": f"a{index}", "label": f"Analyte {index}",
         "series": [{"day": 0.0, "value": 1.0, "flag": "normal"}]}
        for index in range(1, n + 1)
    ]}


def _rag(*sections: str) -> dict:
    return {"query": "q", "passages": [
        {"id": f"id-{index}", "section": section, "text": f"{section} body"}
        for index, section in enumerate(sections, start=1)
    ]}


def test_a_lab_panel_is_numbered_one_to_n():
    rec = {"tool_calls": [{"name": "get_lab_trends", "response": _panel(9)}]}

    evidence = judge._evidence_for(rec)
    labs = evidence[0]["labs"]

    assert [trend["cite"] for trend in labs] == [f"^[{n}]" for n in range(1, 10)]


def test_the_counter_continues_into_the_next_tool_result():
    """Nine analytes, then `rag_search` starts at `^[10]` — not at `^[1]` again.

    This is the numbering the agent renders (`graph.tool_node` → `cite_offset`), so the
    judge and the model agree on what `^[10]` means.
    """
    rec = {"tool_calls": [
        {"name": "get_lab_trends", "response": _panel(9)},
        {"name": "rag_search", "response": _rag("brief_hospital_course",
                                                "discharge_diagnosis")},
    ]}

    evidence = judge._evidence_for(rec)

    assert evidence[0]["labs"][-1]["cite"] == "^[9]"
    assert [p["cite"] for p in evidence[1]["passages"]] == ["^[10]", "^[11]"]


def test_a_second_retrieval_continues_the_same_counter():
    rec = {"tool_calls": [
        {"name": "rag_search", "response": _rag("a", "b")},
        {"name": "rag_search_sections", "response": _rag("c")},
    ]}

    evidence = judge._evidence_for(rec)

    assert [p["cite"] for p in evidence[0]["passages"]] == ["^[1]", "^[2]"]
    assert [p["cite"] for p in evidence[1]["passages"]] == ["^[3]"]


def test_a_tool_that_is_not_evidence_consumes_no_number():
    """An estimate and a medication list are answers; they are not citable passages."""
    rec = {"tool_calls": [
        {"name": "predict_readmission", "response": {"probability": 0.1}},
        {"name": "discharge_medications", "response": {"medications": [{"drug": "A"}]}},
        {"name": "rag_search", "response": _rag("brief_hospital_course")},
    ]}

    evidence = judge._evidence_for(rec)

    assert evidence[2]["passages"][0]["cite"] == "^[1]"


def test_the_raw_payload_is_not_mutated_by_stamping():
    """The recorded response is the execution record; the judge gets a copy."""
    panel = _panel(3)
    rec = {"tool_calls": [{"name": "get_lab_trends", "response": panel}]}

    judge._evidence_for(rec)

    assert all("cite" not in trend for trend in panel["trends"])


def test_a_literature_record_keeps_its_identifier():
    """`^[PMID: …]` is not a passage number, and the record is the only way to check it."""
    rec = {"tool_calls": [{"name": "search_literature", "response": {
        "query": "sepsis", "articles": [
            {"title": "A trial", "pmid": "35186616", "abstract": "Results.",
             "journal": "NEJM", "year": 2021},
        ]}}]}

    article = judge._evidence_for(rec)[0]["articles"][0]

    assert article["pmid"] == "35186616"
    assert article["title"] == "A trial"
    assert article["abstract"] == "Results."


def test_the_manifest_is_first_and_takes_no_number():
    rec = {"tool_calls": [{"name": "rag_search", "response": _rag("a")}],
           "admission_profile": {"age": "73"}}

    evidence = judge._evidence_for(rec)

    assert evidence[0]["tool"] == "admission_manifest"
    assert evidence[1]["passages"][0]["cite"] == "^[1]"
