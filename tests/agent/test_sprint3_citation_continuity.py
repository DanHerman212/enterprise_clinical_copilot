"""Sprint 3: citation numbering holds across a mixed-modality thread.

The trajectory under test is the one that produced the original defect shape:
Turn 1 `/labs` registers laboratory units and publishes no markers at all, Turn 2 is a
note question whose answer is compacted to `1..K`, and Turn 3 is a follow-up with no new
tool call. The only thing standing between Turn 3 and a note claim bound to Turn 1's lab
trajectory is arithmetic: the turn-local map plus the position the replay numbered that
turn from.

Zero LLM, zero network.
"""

from __future__ import annotations

from services.agent.postprocess.citations import (
    cited_numbers,
    compact_display_numbers,
    passage_total,
    restore_global_markers,
    rewrite_pmid_markers,
)
from services.agent.postprocess.evidence_state import SessionEvidenceState

HADM = 90000012


def _labs_payload() -> dict:
    return {
        "hadm_id": HADM,
        "synthetic": True,
        "trends": [
            {
                "label": "Hemoglobin", "unit": "g/dL", "decimals": 1,
                "reference_low": 12.0, "reference_high": 15.5,
                "series": [{"day": 1.0, "value": 11.9, "flag": "low"},
                           {"day": 3.0, "value": 10.0, "flag": "low"}],
            },
            {
                "label": "Creatinine", "unit": "mg/dL", "decimals": 2,
                "reference_low": 0.5, "reference_high": 1.1,
                "series": [{"day": 1.0, "value": 0.9, "flag": "normal"},
                           {"day": 3.0, "value": 1.06, "flag": "normal"}],
            },
        ],
    }


def _note_payload(*sections: str) -> dict:
    return {
        "hadm_id": HADM,
        "query": "discharge",
        "returned": len(sections),
        "passages": [
            {"id": f"MT-1-DS_{name}_1", "section": name,
             "text": f"The {name.replace('_', ' ')} states a fact."}
            for name in sections
        ],
    }


def _thread() -> SessionEvidenceState:
    """The record the replay builds: two lab units, then two note passages."""
    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("get_lab_trends", {"hadm_id": HADM}, _labs_payload())
    record.ingest_tool_output(
        "rag_search_sections", {"hadm_id": HADM},
        _note_payload("brief_hospital_course", "discharge_instructions"))
    return record


def test_a_lab_turn_advances_positions_without_publishing_a_marker():
    """`/labs` publishes `shown_sources: []` and `citation_map: {}` — and still moves the array.

    This is the asymmetry the numbering has to survive: the reader sees no markers on Turn
    1, so the turn ships an empty map, while the record has already handed out positions
    1..N. A replay that read the empty map as "this turn contributed nothing" would number
    Turn 2's first passage 1 instead of N+1.
    """
    record = SessionEvidenceState(hadm_id=HADM)
    result = record.ingest_tool_output(
        "get_lab_trends", {"hadm_id": HADM}, _labs_payload())

    assert result.assigned, "a lab panel registers citable units"
    assert record.position_count() == len(result.assigned)
    # And the published page carries nothing to resolve, which is why the map is empty.
    assert compact_display_numbers("Hemoglobin fell to 10.0 at discharge.") == (
        "Hemoglobin fell to 10.0 at discharge.", {})


def test_a_turn_two_map_restores_onto_the_positions_the_record_assigned():
    record = _thread()
    turn_offset = 2  # the two lab units Turn 1 registered
    turn_answer = "The course states a fact ^[1] and the instructions another ^[2]."

    restored = restore_global_markers(turn_answer, {"1": 1, "2": 2}, turn_offset)

    numbers = cited_numbers(restored)
    assert numbers == [3, 4], "Turn 2's own units start after Turn 1's"
    assert [record.items[n].section for n in numbers] == [
        "brief_hospital_course", "discharge_instructions"]
    assert all(record.items[n].source_tool != "get_lab_trends" for n in numbers)


def test_an_empty_map_on_a_later_turn_drops_rather_than_rebinds():
    """A turn that shipped no map had its markers deleted, never renumbered.

    `restore_global_markers` cannot translate a number it has no entry for, and the safe
    failure is removal: a marker that resolves to the wrong passage is a false statement
    about where a claim came from, which is the failure the whole mechanism exists to
    prevent.
    """
    restored = restore_global_markers("A claim ^[1] and another ^[2].", {}, 2)

    assert "^[" not in restored
    assert "A claim and another." == restored.strip()


def test_turn_three_citing_the_thread_lands_on_note_passages():
    """The end-to-end invariant, expressed over the numbers Turn 3 is shown.

    Turn 2's answer, replayed, carries global `^[3]` and `^[4]`. A follow-up that copies a
    marker out of that answer — or that numbers its own prose from 1 — has to land back on
    the record's positions, and both of those positions are note passages.
    """
    record = _thread()
    replayed = restore_global_markers(
        "The course states a fact ^[1] and the instructions another ^[2].",
        {"1": 1, "2": 2}, 2)
    assert replayed.count("^[3]") == 1 and replayed.count("^[4]") == 1

    # The compacted page a reader still sees numbers them 1..2, and the map says which
    # absolute positions those are.
    page, display = compact_display_numbers(replayed)
    assert page.count("^[1]") == 1 and page.count("^[2]") == 1
    resolved = {int(shown): absolute for shown, absolute in display.items()}
    assert sorted(resolved.values()) == [3, 4]
    assert all(record.items[absolute].source_tool == "rag_search_sections"
               for absolute in resolved.values())


def test_replaying_each_turn_of_the_thread_keeps_the_offsets_monotonic():
    """Turn 1 contributes zero markers and two positions; the replay must not lose one."""
    record = SessionEvidenceState(hadm_id=HADM)
    offsets = []
    for name, payload in (
        ("get_lab_trends", _labs_payload()),
        ("rag_search_sections", _note_payload("brief_hospital_course")),
    ):
        offsets.append(record.position_count())
        record.ingest_tool_output(name, {"hadm_id": HADM}, payload)

    assert offsets == [0, 2]
    assert record.position_count() == 3


def test_a_medication_passage_carries_its_provenance_into_session_state():
    """Sprint 2's `medication_source` has to survive ingestion, or a follow-up cannot see it.

    `dedicated` is the record's own discharge list; `narrative` is the course or the plan,
    where a named drug may be one the ward administered. The header is the only place the
    model reads it.
    """
    record = SessionEvidenceState(hadm_id=HADM)
    record.ingest_tool_output("discharge_medications", {"hadm_id": HADM}, {
        "hadm_id": HADM, "query": "discharge medications", "returned": 1,
        "dedicated_section_present": False,
        "passages": [{"id": "MT-1-DS_brief_hospital_course_1",
                      "section": "brief_hospital_course",
                      "text": "Treated with intravenous antibiotics.",
                      "retrieval": "deterministic",
                      "medication_source": "narrative"}],
    })

    rendered = record.render_for_prompt()
    assert "medication_source=narrative" in rendered
    assert record.items[1].metadata["medication_source"] == "narrative"


def test_pubmed_badges_map_by_identifier_not_by_order():
    """Three records, cited out of order: the identifier decides, the sentence does not."""
    answer = ("The cohort rate was 21% ^[PMID: 29876543], a second reported 18% "
              "^[PMID: 35186616], and a third 24% ^[PMID: 11112222].")
    published = {"35186616": 1, "29876543": 3, "11112222": 2}

    rewritten = rewrite_pmid_markers(answer, published)

    assert "PMID" not in rewritten
    assert "21% ^[3]" in rewritten, "the first sentence's record is published third"
    assert "18% ^[1]" in rewritten
    assert "24% ^[2]" in rewritten


def test_the_harness_numbers_no_evidence_of_its_own():
    """`passage_total` is the single counter, and the harness must not keep a second one.

    The evaluation task used to stamp a cumulative `cite` on every evidence unit beside a
    per-turn `sources` list numbered from 1 — two schemes for one passage, which is how a
    correct citation was once scored as pointing at the wrong section.
    """
    import inspect

    from evaluation.platform import task as harness

    source = inspect.getsource(harness)
    assert "passage_total" not in source, (
        "the harness must not re-derive the count the owner module defines")
    assert "_numbered" not in source
    assert passage_total(_note_payload("a", "b")) == 2
    assert passage_total(_labs_payload()) == 2
