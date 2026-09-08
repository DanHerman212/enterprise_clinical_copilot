"""Tests for the A2UI canvas composer.

These matter more than their size suggests. Every mistake this module can make
fails *silently in a browser*: a v0.8 component shape, a wrong catalog id, or a
dangling child reference all produce a console warning and an empty box, not an
exception. Catching those here turns an invisible rendering bug into a red test.

The citation-semantics tests (renumbering, remapping, intent resolution,
section extraction) live in tests/agent/test_citations.py; this file pins the
composed envelope and the end-to-end presentation contract.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.agent.a2ui import (  # noqa: E402
    AUDIENCE,
    CATALOG_ID,
    SURFACE_ID,
    compose_presentation,
    compose_risk_canvas,
    predict_payload,
    rag_payload,
)

# The pinned fixture from §14 — the same admission used everywhere else in the
# suite, so a change in its numbers shows up as one obvious diff.
GOOD_PAYLOAD = {
    "hadm_id": 20924467,
    "probability": 0.131398,
    "threshold": 0.12,
    "decision": 1,
    "base_value": -1.33862,
    "top_factors": [
        {"feature": "n_prior_admissions", "contribution": 0.4213, "direction": "increases"},
        {"feature": "age", "contribution": -0.1021, "direction": "decreases"},
    ],
    "model_version": "readmission-final-20260723172647",
    "feature_source": "bigquery",
}


GOOD_RAG = {
    "hadm_id": 20924467,
    "query": "discharge notes",
    "returned": 1,
    "passages": [
        {"id": "n_meds_1", "section": "discharge_medications",
         "text": "Discharge Medications:\nwarfarin 4 mg QD", "score": 0.2},
    ],
}


def _by_id(canvas):
    components = canvas["messages"][1]["updateComponents"]["components"]
    return {c["id"]: c for c in components}


class TestMessageEnvelope:
    def test_message_order_is_create_then_components(self):
        """The renderer applies messages in order; the surface must exist first."""
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        keys = [next(k for k in m if k != "version") for m in canvas["messages"]]
        assert keys == ["createSurface", "updateComponents"]

    def test_every_message_declares_v0_9(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        assert all(m["version"] == "v0.9" for m in canvas["messages"])

    def test_catalog_id_matches_the_registered_custom_catalog(self):
        """A mismatch here resolves no components and raises nothing."""
        create = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)["messages"][0]["createSurface"]
        assert create["catalogId"] == CATALOG_ID
        assert create["surfaceId"] == SURFACE_ID

    def test_component_is_a_string_not_a_nested_object(self):
        """The v0.8 shape `{component: {Text: {...}}}` renders an empty box."""
        for component in _by_id(compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)).values():
            assert isinstance(component["component"], str), component

    def test_every_child_reference_resolves(self):
        """A dangling id renders nothing, silently."""
        components = _by_id(compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG))
        for component in components.values():
            for ref in [component.get("child")] + list(component.get("children") or []):
                if ref is not None:
                    assert ref in components, f"dangling reference {ref!r}"


class TestRiskComposition:
    def test_emits_the_custom_components(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        types = {c["component"] for c in _by_id(canvas).values()}
        assert {"RiskBar", "FactorBars", "SourceCard"} <= types

    def test_factor_bars_carry_human_labels_not_raw_model_keys(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        factors = _by_id(canvas)["factors"]["factors"]
        # Not in the curated map -> Title Case fallback.
        assert factors[0]["label"] == "N Prior Admissions"
        # "age" is a curated FEATURE_LABELS entry.
        assert factors[1]["label"] == "age"

    def test_risk_bar_carries_probability_threshold_and_band(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        risk = _by_id(canvas)["risk"]
        assert risk["probability"] == pytest.approx(0.131398)
        assert risk["threshold"] == pytest.approx(0.12)
        # 0.1314 >= 0.12 and < 0.20 -> borderline.
        assert risk["band"] == "borderline"

    def test_provenance_names_model_and_feature_source(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        assert "readmission-final-20260723172647" in _by_id(canvas)["prov"]["text"]

    def test_fallback_text_carries_the_number_and_band(self):
        """R8 — if the renderer never boots the user still gets the answer."""
        fallback = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)["fallback_text"]
        assert "13.1%" in fallback
        assert "borderline" in fallback

    def test_audience_hides_the_payload_from_the_model(self):
        """R7 — stops the model re-reading its own UI as evidence."""
        assert compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)["audience"] == AUDIENCE == ["user"]


class TestNoEstimatePaths:
    def test_without_predict_composes_a_source_only_canvas(self):
        """A non-risk question (meds / summarize) still composes a canvas."""
        canvas = compose_risk_canvas(None, GOOD_RAG)
        types = {c["component"] for c in _by_id(canvas).values()}
        assert "SourceCard" in types
        assert "RiskBar" not in types
        assert canvas["fallback_text"].strip()
        prov = _by_id(canvas)["prov"]
        assert "no readmission estimate was requested" in prov["text"]

    def test_errored_predict_is_an_honest_note_never_a_500(self):
        canvas = compose_risk_canvas(
            {"error": "upstream 503", "status": "failed"}, GOOD_RAG)
        note = _by_id(canvas)["note"]
        assert "did not return a usable estimate" in note["text"]
        assert "did not return a usable estimate" in canvas["fallback_text"]

    def test_unusable_predict_missing_keys_is_not_an_estimate(self):
        """A partial payload must not produce something that looks like a result."""
        canvas = compose_risk_canvas({"hadm_id": 1, "probability": 0.4}, None)
        assert "did not return a usable estimate" in canvas["fallback_text"]

    def test_no_passages_is_an_honest_empty_source_card(self):
        canvas = compose_risk_canvas(None, None)
        source = _by_id(canvas)["source"]
        assert source["section"] == "not found"
        assert "No supporting note passage" in source["text"]


class TestSourceCardResolution:
    def test_resolves_the_cited_passage_by_section_not_number(self):
        """A meds answer cites ^[1] while the meds passage sits at index 2 —
        the canvas must still show discharge_medications."""
        rag = {"passages": [
            {"id": "n_bhc_1", "section": "brief_hospital_course",
             "text": "Hospital Course: recovered.", "score": 0.3},
            {"id": "n_dx_1", "section": "discharge_diagnosis",
             "text": "Discharge Diagnoses: TKA.", "score": 0.2},
            {"id": "n_meds_1", "section": "discharge_medications",
             "text": "Discharge Medications: Celebrex 200 mg daily.", "score": 0.1},
        ], "query": "discharge notes"}
        canvas = compose_risk_canvas(
            None, rag, cite=1, sections=("discharge_medications",))
        source = _by_id(canvas)["source"]
        assert source["section"] == "discharge_medications"
        assert "Celebrex" in source["text"]
        # Badge mirrors the thread's citation number, not the array position.
        assert source["cite"] == 1

        # Without a section hint the number mapping is unchanged.
        canvas = compose_risk_canvas(None, rag, cite=1)
        assert _by_id(canvas)["source"]["section"] == "brief_hospital_course"

        # A section hint the note does not have is a deterministic
        # "not available" card — never a fallback to the wrong section.
        canvas = compose_risk_canvas(
            None, rag, cite=2, sections=("discharge_instructions",))
        source = _by_id(canvas)["source"]
        assert source["section"] == "not available"
        assert "instruction" in source["text"]

    def test_extracts_the_intent_section_from_whole_note_chunks(self):
        """The index stores whole-note chunks, so the intent-labeled passage
        can miss the returned list entirely. The canvas must then pull the
        intent section's body OUT of any returned chunk's text."""
        whole_note = (
            "CHIEF COMPLAINT: Knee pain.\n\n"
            "HOSPITAL COURSE: She underwent a right TKA and recovered.\n\n"
            "DISCHARGE DIAGNOSES: 1. S/p right TKA.\n\n"
            "MEDICATIONS: Celebrex 200 mg daily."
        )
        rag = {"passages": [
            {"id": "n_bhc_1", "section": "brief_hospital_course",
             "text": whole_note, "score": 0.3},
            {"id": "n_dx_1", "section": "discharge_diagnosis",
             "text": whole_note, "score": 0.2},
        ], "query": "discharge notes"}
        canvas = compose_risk_canvas(
            None, rag, cite=1, sections=("discharge_medications",))
        source = _by_id(canvas)["source"]
        assert source["section"] == "discharge_medications"
        assert "Celebrex" in source["text"]
        assert "HOSPITAL COURSE" not in source["text"]
        assert source["cite"] == 1

    def test_resolves_meds_to_instructions_when_no_meds_section(self):
        """The meds claim's supporting text can live in DISCHARGE INSTRUCTIONS
        — the intent set must resolve there, never to brief_hospital_course."""
        note = (
            "DISCHARGE DIAGNOSES: Cellulitis.\n\n"
            "DISCHARGE INSTRUCTIONS: The patient would be discharged on his "
            "usual Valium 10-20 mg at bedtime for spasticity, Flomax 0.4 mg "
            "daily, cefazolin 500 mg q.i.d., and Lotrimin cream between toes.\n\n"
            "HOSPITAL COURSE: The patient was admitted to the General Medical "
            "floor and treated with intravenous ceftriaxone and topical "
            "Lotrimin."
        )
        rag = {"passages": [
            {"id": "n_bhc_1", "section": "brief_hospital_course",
             "text": note, "score": 0.3},
            {"id": "n_ins_1", "section": "discharge_instructions",
             "text": note, "score": 0.1},
        ], "query": "discharge notes"}
        canvas = compose_risk_canvas(
            None, rag, cite=1,
            sections=("discharge_medications", "discharge_instructions"))
        source = _by_id(canvas)["source"]
        assert source["section"] == "discharge_instructions"
        assert "Valium" in source["text"]
        assert "HOSPITAL COURSE" not in source["text"]

    def test_unavailable_when_the_note_lacks_all_target_sections(self):
        """The note mentions meds only inside the hospital course — the
        deterministic answer is 'not available', never a meds sentence mined
        from the hospital course narrative."""
        note = (
            "ADMISSION DIAGNOSIS: Symptomatic thyroid goiter.\n\n"
            "HOSPITAL COURSE: The patient underwent total thyroidectomy on "
            "09/22/08, which she tolerated very well. She was given "
            "prescription for Vicodin for pain and Synthroid thyroid hormone."
        )
        rag = {"passages": [
            {"id": "n_bhc_1", "section": "brief_hospital_course",
             "text": note, "score": 0.3},
            {"id": "n_dx_1", "section": "discharge_diagnosis",
             "text": note, "score": 0.2},
        ], "query": "discharge notes"}
        canvas = compose_risk_canvas(
            None, rag, cite=1,
            sections=("discharge_medications", "discharge_instructions"))
        source = _by_id(canvas)["source"]
        assert source["section"] == "not available"
        assert "No discharge medication information" in source["text"]
        assert "Vicodin" not in source["text"]


class TestToolCallExtraction:
    def test_predict_payload_uses_the_last_prediction(self):
        other = {**GOOD_PAYLOAD, "hadm_id": 111}
        calls = [
            {"name": "predict_readmission", "response": GOOD_PAYLOAD},
            {"name": "predict_readmission", "response": other},
        ]
        assert predict_payload(calls)["hadm_id"] == 111

    def test_predict_payload_is_none_without_a_prediction(self):
        """Answering without a predict call is legitimate (meds / summarize)."""
        assert predict_payload([]) is None
        assert predict_payload([{"name": "other", "response": {}}]) is None

    def test_rag_payload_prefers_free_text_search(self):
        calls = [
            {"name": "rag_search_sections", "response": {"query": "sections"}},
            {"name": "rag_search", "response": {"query": "free"}},
        ]
        assert rag_payload(calls)["query"] == "free"

    def test_rag_payload_falls_back_to_sections(self):
        calls = [{"name": "rag_search_sections", "response": {"query": "sections"}}]
        assert rag_payload(calls)["query"] == "sections"

    def test_rag_payload_is_none_without_retrieval(self):
        assert rag_payload([]) is None


class TestPresentationContract:
    def test_compose_presentation_returns_the_full_browser_contract(self):
        calls = [
            {"name": "predict_readmission", "response": GOOD_PAYLOAD},
            {"name": "rag_search", "response": GOOD_RAG},
        ]
        out = compose_presentation(
            "What medications was this patient discharged on? "
            "For admission 20924467.",
            "The patient was discharged on warfarin^[1].",
            calls,
        )
        assert out["answer"] == "The patient was discharged on warfarin^[1]."
        assert out["citation_map"] == {"1": 1}
        assert out["intent_sections"] == [
            "discharge_medications", "discharge_instructions"]
        assert out["a2ui"]["surface_id"] == "risk-canvas"
        # A predict ran, so the canvas carries the risk widgets too.
        types = {c["component"] for c in _by_id(out["a2ui"]).values()}
        assert {"RiskBar", "SourceCard"} <= types

    def test_compose_presentation_renumbers_out_of_order_citations(self):
        out = compose_presentation(
            "Summarize the recent discharge notes for this patient.",
            "A^[2] and B^[1]",
            [{"name": "rag_search", "response": GOOD_RAG}],
        )
        assert out["answer"] == "A^[1] and B^[2]"
        assert out["citation_map"] == {"1": 2, "2": 1}
        assert out["intent_sections"] == []
