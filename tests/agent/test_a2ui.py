"""Tests for the A2UI canvas composer.

These matter more than their size suggests. Every mistake this module can make
fails *silently in a browser*: a v0.8 component shape, a wrong catalog id, or a
dangling child reference all produce a console warning and an empty box, not an
exception. Catching those here turns an invisible rendering bug into a red test.

The citation-semantics tests (renumbering, remapping, intent resolution,
section extraction) live in tests/agent/test_citations.py; this file pins the
composed envelope and the end-to-end presentation contract.
"""

import json
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
    resolve_source,
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

        assert "RiskInfluencers" in types
        # No source card: the passage travels as the narrative's citation now — one
        # marker per sentence, opened by the reader, rather than one card detached
        # from all of them.
        assert "SourceCard" not in types

    def test_influencers_carry_human_labels_not_raw_model_keys(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        factors = _by_id(canvas)["risk"]["factors"]
        # Not in the curated map -> Title Case fallback.
        assert factors[0]["label"] == "N Prior Admissions"
        # "age" is a curated FEATURE_LABELS entry.
        assert factors[1]["label"] == "age"

    def test_the_widget_carries_the_unrounded_probability_and_the_decision(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        risk = _by_id(canvas)["risk"]
        # Unrounded on the wire: the widget prints one decimal place, and rounding
        # here to the prose's two decimals would erase the digit that separates a
        # score from its threshold. The served value is 0.131398.
        assert risk["probability"] == pytest.approx(0.131398)
        assert risk["threshold"] == pytest.approx(0.12)
        # The model's own call travels with the number, so the widget's `>=`
        # comparison can never overrule it.
        assert risk["decision"] == 1
        assert risk["factors"][0]["contribution"] == pytest.approx(0.4213)
        assert risk["factors"][0]["direction"] == "increases"

    def test_provenance_names_model_and_feature_source(self):
        canvas = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)
        assert "readmission-final-20260723172647" in _by_id(canvas)["prov"]["text"]

    def test_fallback_text_carries_the_number_and_band(self):
        """R8 — if the renderer never boots the user still gets the answer."""
        fallback = compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)["fallback_text"]
        # Whole percent, which is the same precision as two decimals of a
        # probability: 0.13 is 13%, not 13.1%.
        assert "13%" in fallback
        assert "0.13" in fallback
        assert "13.1%" not in fallback
        assert "borderline" in fallback

    def test_audience_hides_the_payload_from_the_model(self):
        """R7 — stops the model re-reading its own UI as evidence."""
        assert compose_risk_canvas(GOOD_PAYLOAD, GOOD_RAG)["audience"] == AUDIENCE == ["user"]


class TestMedicationGrid:
    """The grid is drawn from rows the command already verified, never recomposed."""

    ROWS = {
        "medications": [
            {"medication": "warfarin", "dosage": "4 mg", "frequency": "QD"},
        ],
        "dropped": [],
        "returned_passages": 1,
    }

    def test_the_grid_draws_the_verified_rows(self):
        canvas = compose_risk_canvas(None, None, medications=self.ROWS)
        grid = _by_id(canvas)["medications"]

        assert grid["component"] == "MedicationGrid"
        assert grid["medications"] == [
            {"medication": "warfarin", "dosage": "4 mg", "frequency": "QD"}
        ]
        assert grid["dropped_count"] == 0

    def test_no_rows_composes_no_grid_at_all(self):
        """An empty grid reads as "this patient takes nothing" — a clinical claim.

        The narrative states the absence instead, once.
        """
        canvas = compose_risk_canvas(
            None, None,
            medications={"medications": [], "dropped": [], "returned_passages": 0},
        )

        assert "medications" not in _by_id(canvas)

    def test_the_unverified_rows_are_counted_and_not_drawn(self):
        rows = {
            **self.ROWS,
            "dropped": [{"medication": "gabapentin", "reason": "no_dosage"}],
        }
        grid = _by_id(compose_risk_canvas(None, None, medications=rows))["medications"]

        assert grid["dropped_count"] == 1
        assert "gabapentin" not in {row["medication"] for row in grid["medications"]}

    def test_a_missing_field_arrives_as_an_empty_string(self):
        """The component's schema is strict, so a null would take the grid down."""
        rows = {
            "medications": [{"medication": "aspirin"}],
            "dropped": [],
            "returned_passages": 1,
        }
        grid = _by_id(compose_risk_canvas(None, None, medications=rows))["medications"]

        assert grid["medications"][0] == {
            "medication": "aspirin", "dosage": "", "frequency": ""
        }


class TestNoEstimatePaths:
    def test_without_predict_composes_a_canvas_with_no_estimate(self):
        """A non-risk question (meds / summarize) still composes a canvas — and one
        that states no provenance, because there is none to state. The caption that
        used to stand in for one ("no readmission estimate was requested") answered
        a question the clinician had not asked, in the voice of a bug report
        (removed 2026-10-12)."""
        canvas = compose_risk_canvas(None, GOOD_RAG)
        types = {c["component"] for c in _by_id(canvas).values()}

        assert "RiskInfluencers" not in types
        assert canvas["fallback_text"].strip()
        assert "prov" not in _by_id(canvas)

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

    def test_no_passages_composes_no_source_card_at_all(self):
        """A card captioned "not found" over a sentence about the agent not
        fabricating passages was a note to a reviewer, printed in the clinician's
        canvas on every turn that cited no note — including /risk, where retrieval
        is forbidden by design (removed 2026-10-12). An empty citation slot is not
        evidence of a fault."""
        canvas = compose_risk_canvas(None, None)

        assert "source" not in _by_id(canvas)
        assert "not found" not in json.dumps(canvas)


class TestSourceResolution:
    """Which passage a citation resolves to, and how a section hint picks it.

    Tested against `resolve_source` rather than through the canvas: the canvas no
    longer draws an evidence card, and this resolution is what the narrative's
    citations are built from, so this is now the layer that matters.
    """

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
        resolved = resolve_source(rag["passages"], 1, ("discharge_medications",))
        assert resolved["section"] == "discharge_medications"
        assert "Celebrex" in resolved["text"]

        # Without a section hint the number mapping is unchanged.
        assert resolve_source(rag["passages"], 1, ())["section"] == "brief_hospital_course"

        # A section hint the note does not have is a deterministic
        # "not available" — never a fallback to the wrong section.
        resolved = resolve_source(rag["passages"], 2, ("discharge_instructions",))
        assert resolved["section"] == "not available"
        assert "instruction" in resolved["text"]

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
        resolved = resolve_source(rag["passages"], 1, ("discharge_medications",))
        assert resolved["section"] == "discharge_medications"
        assert "Celebrex" in resolved["text"]
        assert "HOSPITAL COURSE" not in resolved["text"]

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
        resolved = resolve_source(
            rag["passages"], 1, ("discharge_medications", "discharge_instructions"))
        assert resolved["section"] == "discharge_instructions"
        assert "Valium" in resolved["text"]
        assert "HOSPITAL COURSE" not in resolved["text"]

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
        resolved = resolve_source(
            rag["passages"], 1, ("discharge_medications", "discharge_instructions"))
        assert resolved["section"] == "not available"
        assert "No discharge medication information" in resolved["text"]
        assert "Vicodin" not in resolved["text"]


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

    def test_rag_payload_merges_every_retrieval_call_in_order(self):
        """Numbering is global across calls, so the array is the concatenation.

        This used to return only the `rag_search` payload when both kinds were present,
        which is what the citation markers do NOT index: the prompt tells the model the
        second call's passages continue after the first's, so a marker past the first
        call resolved to the wrong passage (`/ehr`'s sectioned call plus its record-wide
        fallback).
        """
        calls = [
            {"name": "rag_search_sections",
             "response": {"query": "sections", "passages": [{"text": "a"}]}},
            {"name": "rag_search",
             "response": {"query": "free", "passages": [{"text": "b"}]}},
        ]
        merged = rag_payload(calls)
        assert merged["query"] == "sections", "held to call order, like the numbering"
        assert [p["text"] for p in merged["passages"]] == ["a", "b"]

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
        assert {"RiskInfluencers", "CitedNarrative"} <= types
        # The surface draws the prose and the payload says so, which is what stops
        # the console writing the same answer into its own text element as well.
        assert out["a2ui"]["renders_prose"] is True
        narrative = _by_id(out["a2ui"])["narrative"]
        assert narrative["text"] == out["answer"]
        assert [c["index"] for c in narrative["citations"]] == [1]
        assert narrative["streaming"] is False

    def test_compose_presentation_renumbers_out_of_order_citations(self):
        out = compose_presentation(
            "Summarize the recent discharge notes for this patient.",
            "A^[2] and B^[1]",
            [{"name": "rag_search", "response": GOOD_RAG}],
        )
        assert out["answer"] == "A^[1] and B^[2]"
        assert out["citation_map"] == {"1": 2, "2": 1}
        assert out["intent_sections"] == []

    def test_a_chart_turn_without_a_resolved_citation_keeps_no_scope(self):
        """A chart read that resolved nothing cannot claim chart synthesis.

        The scope is a statement about what the answer rests on, and an answer with
        no note citation has nothing to show — so it is withheld, and the
        suppression is reported so the BFF can carry it as a flag rather than lose
        it at this boundary.
        """
        out = compose_presentation(
            "Summarize the recent discharge notes for this admission.",
            "The patient recovered uneventfully and was discharged home.",
            [{"name": "rag_search", "args": {"hadm_id": 20924467},
              "response": {"query": "discharge notes", "passages": []}}],
        )
        assert out["scope"] is None
        assert out["scope_suppressed"] is True

    def test_a_chart_turn_with_a_resolved_citation_keeps_the_scope(self):
        out = compose_presentation(
            "Summarize the recent discharge notes for this admission.",
            "The patient was discharged on warfarin^[1].",
            [{"name": "rag_search", "args": {"hadm_id": 20924467},
              "response": GOOD_RAG}],
        )
        assert out["scope"] == "patient_chart"
        assert out["scope_suppressed"] is False

    def test_a_general_turn_without_citations_keeps_its_scope(self):
        """A general-medical answer legitimately cites nothing, so the rule must
        not reach it — which is why it waited on the classifier rather than
        shipping alongside the citation guard."""
        out = compose_presentation(
            "What does the literature say about early mobilisation after surgery?",
            "Early mobilisation shortens length of stay.",
            [{"name": "search_literature", "response": {"articles": []}}],
        )
        assert out["scope"] == "general_evidence"
        assert out["scope_suppressed"] is False

    def test_a_comparative_turn_is_general_evidence_not_chart_synthesis(self):
        """The patient's estimate does not turn a literature turn into a chart read.

        `
        `/pubmed` fetches both (Phase 2, 2026-10-16): the records, and this patient's own
        estimate as the baseline the matrix pins above them. Reading only as far as "a
        patient-scoped tool ran" made every comparative turn a chart read with no note
        citations — so it dropped its scope pill AND raised a visible safety swap over an
        answer nothing had been removed from.

        The admission is one the reviewed map carries (Phase 4B), because the matrix is
        selected by the patient's own phenotype now: a comparison needs an admission whose
        condition is known, and an id that resolves to nothing draws no table. That is
        asserted separately; here it is only the vehicle.
        """
        out = compose_presentation(
            "Compare this patient's readmission risk against peer-reviewed clinical "
            "cohorts.",
            "His 9.0% sits below the 13.2% pooled for hospitalised heart failure^[1].",
            [
                {"name": "search_literature", "args": {"query": "heart failure"},
                 "response": {"query": "heart failure", "returned": 1, "articles": [{
                     "pmid": "37495280",
                     "title": "Global Comparison of Readmission Rates.",
                     "journal": "JACC",
                     "pub_date": "2023 Aug 01",
                     "abstract": "13.2% were readmitted within 30 days.",
                     "url": "https://pubmed.ncbi.nlm.nih.gov/37495280/"}]}},
                {"name": "predict_readmission", "args": {"hadm_id": 90000023},
                 "response": GOOD_PAYLOAD},
            ],
            # A `/pubmed` turn, which is one of the two commands whose question the
            # comparison table answers. Without the command the composer cannot know
            # which surface it is composing for, and draws no table.
            command="literature",
        )

        assert out["scope"] == "general_evidence"
        assert out["scope_suppressed"] is False
        # And the table is drawn, so the one claim about this patient has somewhere to
        # sit beside the rate it is being compared with.
        types = [c["component"] for c in
                 out["a2ui"]["messages"][1]["updateComponents"]["components"]]
        assert "ComparativeRiskMatrix" in types


class TestUnifiedEhrRiskTurn:
    """`/ehr` is the one composite command: one turn, a narrative and a risk
    estimate (§3, 2026-10-15).

    The pairing is the feature. A clinician who has to ask twice to see the score
    and the note together is being asked to hold the first answer while they wait
    for the second, and both are about the same admission.
    """

    #: A turn that made both calls, in the order the routing table declares them.
    BOTH_CALLS = [
        {"name": "rag_search_sections",
         "args": {"hadm_id": 20924467},
         "response": GOOD_RAG},
        {"name": "predict_readmission",
         "args": {"hadm_id": 20924467},
         "response": GOOD_PAYLOAD},
    ]

    def _composed(self):
        out = compose_presentation(
            "Summarize the discharge notes for this patient.",
            "The patient was discharged on warfarin.^[1]",
            self.BOTH_CALLS,
        )
        return out, _by_id(out["a2ui"])

    def test_one_turn_carries_the_narrative_and_the_risk_widget(self):
        """Both halves, composed from the two calls the command made. The canvas
        needed no change to do this — it derives the estimate from the recorded
        calls — which is why the unified view is a routing decision, not a
        rendering one."""
        _, by_id = self._composed()
        assert "narrative" in by_id, "the summary is missing from the unified turn"
        assert "risk" in by_id, "the risk widget is missing from the unified turn"
        assert by_id["risk"]["component"] == "RiskInfluencers"

    def test_the_driver_bars_come_with_the_chip(self):
        """The chip is the number and the bars are why. A score drawn without its
        drivers is the half of the pairing a clinician cannot act on."""
        _, by_id = self._composed()
        labels = [f["label"] for f in by_id["risk"]["factors"]]
        assert labels, "the unified turn drew a score with no drivers"

    def test_the_turn_is_still_a_chart_turn(self):
        assert self._composed()[0]["scope"] == "patient_chart"

    def test_the_fallback_is_the_summary_not_the_score(self):
        """`fallback_text` is what a client shows when the renderer is unavailable.
        For `/ehr` that has to be the discharge summary: a bare risk percentage in
        its place reads as an answer, and it is not the answer that was asked for.
        """
        out, _ = self._composed()
        fallback = out["a2ui"]["fallback_text"]
        assert fallback == out["answer"]
        assert "warfarin" in fallback

    def test_the_widget_sits_immediately_below_the_narrative(self):
        """The widget is the evidence under the fourth section, not a neighbour of it.

        On this path the integration IS the order, and the order is decided here rather
        than in the browser: the composer appends the narrative first and the estimate
        second, so a reader meets the explanation and then the numbers it explains.

        Asserted as adjacency rather than as 'below', because 'below' is what produced a
        score floating beside a summary with nothing joining the two — the thing the
        2026-10-15 remediation exists to undo.
        """
        out, _ = self._composed()
        components = out["a2ui"]["messages"][1]["updateComponents"]["components"]
        ids = [c["id"] for c in components]

        assert "narrative" in ids and "risk" in ids, ids
        assert ids.index("narrative") + 1 == ids.index("risk"), ids


class TestResolvedSources:
    """`sources` is what the browser renders on a footnote click. It must be
    fully resolved here so the client holds no vocabulary and no heuristics."""

    THREE = {"query": "discharge notes", "passages": [
        {"id": "n_bhc_1", "section": "brief_hospital_course",
         "text": "Hospital Course: recovered.", "score": 0.3},
        {"id": "n_dx_1", "section": "discharge_diagnosis",
         "text": "Discharge Diagnoses: TKA.", "score": 0.2},
        {"id": "n_meds_1", "section": "discharge_medications",
         "text": "Discharge Medications: Celebrex 200 mg daily.", "score": 0.1},
    ]}

    def test_one_entry_per_citation_keyed_by_renumbered_cite(self):
        out = compose_presentation(
            "Summarize the recent discharge notes.",
            "A^[2] and B^[1]",
            [{"name": "rag_search", "response": self.THREE}],
        )
        assert [s["cite"] for s in out["sources"]] == [1, 2]
        # Renumbered ^[1] was the model's ^[2] -> the diagnosis passage.
        assert out["sources"][0]["section"] == "discharge_diagnosis"
        assert out["sources"][1]["section"] == "brief_hospital_course"
        assert all(s["query"] == "discharge notes" for s in out["sources"])

    def test_single_citation_resolves_by_section_intent(self):
        """A meds answer cites ^[1] but the meds passage is third — the
        resolved source is discharge_medications, not the first passage."""
        out = compose_presentation(
            "What medications was the patient discharged on?",
            "Celebrex^[1].",
            [{"name": "rag_search", "response": self.THREE}],
        )
        assert len(out["sources"]) == 1
        src = out["sources"][0]
        assert (src["cite"], src["section"], src["query"]) == (
            1, "discharge_medications", "discharge notes")
        assert "Celebrex" in src["text"]
        assert "Hospital Course" not in src["text"]

    def test_the_narrative_citation_is_sources_zero(self):
        """The marker the reader opens and the resolved list can never disagree.

        This was the source card's job. The card is gone and the invariant it
        protected is not, so it moves to the citation the reader actually clicks.
        """
        out = compose_presentation(
            "What medications was the patient discharged on?",
            "Celebrex^[1].",
            [{"name": "rag_search", "response": self.THREE}],
        )
        citation = _by_id(out["a2ui"])["narrative"]["citations"][0]
        first = out["sources"][0]

        assert citation["index"] == first["cite"]
        assert citation["excerpt"] == first["text"]
        assert citation["title"] == "Discharge medications"
        assert citation["source"] == "Discharge note"

    def test_unavailable_section_is_a_deterministic_entry(self):
        rag = {"query": "discharge notes", "passages": [
            {"id": "n_bhc_1", "section": "brief_hospital_course",
             "text": "Hospital Course: given Vicodin.", "score": 0.3}]}
        out = compose_presentation(
            "What medications was the patient discharged on?",
            "None listed^[1].",
            [{"name": "rag_search", "response": rag}],
        )
        assert out["sources"][0]["section"] == "not available"
        assert "Vicodin" not in out["sources"][0]["text"]

    def test_no_passages_or_no_citations_is_an_empty_list(self):
        assert compose_presentation("Q", "No cites.", []) ["sources"] == []
        out = compose_presentation(
            "Q", "No cites.", [{"name": "rag_search", "response": self.THREE}])
        assert out["sources"] == []
