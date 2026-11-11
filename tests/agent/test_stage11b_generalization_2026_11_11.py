"""Stage 11B synthetic tests: seven generalizable fixes, no eval data in them.

Each block is written against the CLASS of defect the corresponding holdout failure
exposed, with synthetic inputs only. Nothing here reads an evaluation dataset, and the
last test asserts that no name or figure from the held-out suite appears anywhere in
`services/`.
"""

from __future__ import annotations

import pathlib

import pytest

from services.agent.engine import deterministic, prompts
from services.agent.postprocess import answer as answer_postprocess
from services.agent.postprocess import answer_hygiene, followups
from services.agent.presentation import a2ui
from services.agent.routing import routing

SERVICES = pathlib.Path(__file__).resolve().parents[2] / "services"


# --------------------------------------------------------------------------- fixtures


def _trend(label: str, *, low: float, high: float, values: list[float],
           flags: list[str] | None = None, unit: str = "mg/dL",
           decimals: int = 1, key: str | None = None) -> dict:
    flags = flags or ["normal"] * len(values)
    return {
        "key": key or label.lower().replace(" ", "_"),
        "label": label,
        "unit": unit,
        "reference_low": low,
        "reference_high": high,
        "decimals": decimals,
        "anchored": [],
        "series": [
            {"day": index, "value": value, "flag": flag}
            for index, (value, flag) in enumerate(zip(values, flags))
        ],
    }


def _lab_calls(*trends: dict) -> list[dict]:
    return [{"name": "get_lab_trends", "response": {"returned": len(trends),
                                                    "trends": list(trends)}}]


def _section_calls(*passages: dict) -> list[dict]:
    return [{"name": "rag_search_sections",
             "response": {"returned": len(passages), "passages": list(passages)}}]


# ------------------------------------------------------- Fix 1: colloquial routing


class TestReadmissionConceptRouting:
    """A readmission question is asked in plain English as often as in the code's words."""

    @pytest.mark.parametrize("question", [
        "What is her readmission risk?",
        "Could she be readmitted?",
        "How likely is she to bounce back within a month, and what's driving that?",
        "What is the chance she returns to hospital within 30 days?",
        "What are the odds she comes back in 30 days?",
        "Which factors drive her rehospitalization risk?",
    ])
    def test_the_concept_is_recognised(self, question):
        assert routing.mentions_readmission(question) is True

    @pytest.mark.parametrize("question", [
        "Why was she started on metoprolol?",
        "What explains the drop in hemoglobin?",
        "What dose of warfarin was she discharged on?",
        "What symptoms brought this patient in?",
    ])
    def test_an_unrelated_question_is_not(self, question):
        assert routing.mentions_readmission(question) is False

    def test_a_compound_question_binds_the_predictor_through_the_driver_predicate(self):
        """Both halves of one question, answered from one call.

        The score predicate excludes a driver question on purpose: two predicates
        admitting the same call for different reasons would make the binding
        unauditable. The compound question still binds, through the driver predicate.
        """
        question = "How likely is she to bounce back within a month, and what's driving that?"
        assert routing.asks_why_risk_driver(question) is True
        assert routing.asks_for_risk_score(question) is False
        assert routing.wants_risk_driver(question, None) is True

    @pytest.mark.parametrize("question", [
        "What is her readmission risk score?",
        "What is his 30-day readmission risk score and is it above threshold?",
        "What is the chance she returns to hospital within 30 days?",
        "How likely is a readmission?",
    ])
    def test_an_estimate_question_is_a_score_question(self, question):
        assert routing.asks_for_risk_score(question) is True

    @pytest.mark.parametrize("question", [
        "What in this patient's chart explains oncology history as a readmission risk driver?",
        "What rate of readmission did that cohort report?",
        "Why was she started on metoprolol?",
    ])
    def test_the_predicates_stay_apart(self, question):
        """A driver question is not a score question, and a rate question is neither."""
        assert routing.asks_for_risk_score(question) is False

    def test_a_plain_request_for_the_score_is_a_score_question(self):
        assert routing.asks_for_risk_score("Show me the risk score for this admission")
        assert routing.asks_why_risk_driver("Show me the risk score for this admission") is False

    def test_a_consultation_question_is_still_not_a_readmission_question(self):
        assert routing.asks_why_risk_driver("Why was she started on metoprolol?") is False

    def test_the_spoke_states_the_compound_answer(self):
        text = " ".join(prompts.RISK_SPOKE.split())
        assert "how likely a return is AND what drives it" in text
        assert "the estimate against its threshold first, then the drivers" in text


class TestEstimateEvidencedSuppression:
    """A free-text turn resting on this turn's estimate is chart-grounded."""

    PREDICTION = {
        "hadm_id": 90000042,
        "probability": 0.176,
        "probability_display": "17.6%",
        "threshold": 0.11,
        "threshold_display": "11.0%",
        "decision": 1,
        "base_value": -1.2,
        "top_factors": [
            {"feature": "prior_inpatient_days", "contribution": 0.21,
             "direction": "increases", "value": 3},
        ],
        "model_version": "readmission-final-20260723172647",
        "feature_source": "bigquery",
    }

    def test_an_estimate_only_turn_keeps_its_scope_and_its_badge(self):
        out = a2ui.compose_presentation(
            "How likely is she to come back within 30 days, and what drives that?",
            "Her estimate is 17.6% against an 11.0% threshold, above it.",
            [{"name": "predict_readmission", "args": {"hadm_id": 90000042},
              "response": self.PREDICTION}],
        )
        assert out["scope"] == "patient_chart"
        assert out["scope_suppressed"] is False
        assert out["verified"] is True

    def test_a_chart_synthesis_with_nothing_behind_it_is_still_suppressed(self):
        out = a2ui.compose_presentation(
            "Summarize the recent discharge notes for this admission.",
            "The patient recovered uneventfully and was discharged home.",
            [{"name": "rag_search", "args": {"hadm_id": 90000042},
              "response": {"query": "discharge notes", "passages": []}}],
        )
        assert out["scope"] is None
        assert out["scope_suppressed"] is True


# ------------------------------------------------ Fix 2: chip ranking by trajectory


class TestAbnormalChipRanking:
    def test_a_critical_reading_outranks_a_wider_excursion(self):
        wide = _trend("Sodium", low=135.0, high=145.0, values=[140.0, 150.0],
                      flags=["normal", "high"])
        critical = _trend("Potassium", low=3.5, high=5.0, values=[4.0, 5.6],
                          flags=["normal", "critical_high"])
        assert followups._rank_abnormal([wide, critical])[0]["label"] == "Potassium"

    def test_the_worst_reading_of_the_stay_outranks_a_barely_abnormal_discharge_value(self):
        """A series that peaked far outside its interval is the one to ask about."""
        margin = _trend("Sodium", low=135.0, high=145.0, values=[140.0, 145.4],
                        flags=["normal", "high"])
        peaked = _trend("Creatinine", low=0.5, high=1.1, values=[3.4, 1.3],
                        flags=["critical_high", "high"])
        assert followups._rank_abnormal([margin, peaked])[0]["label"] == "Creatinine"

    def test_an_analyte_normal_at_discharge_is_never_named_as_an_abnormal_discharge_value(self):
        """The filter stays on the discharge value, so the chip's wording stays true."""
        settled = _trend("Creatinine", low=0.5, high=1.1, values=[3.4, 0.9],
                         flags=["critical_high", "normal"])
        assert followups._rank_abnormal([settled]) == []
        assert followups._max_series_excursion(settled) > 0

    def test_the_discharge_value_still_breaks_a_tie(self):
        first = _trend("RDW", low=11.5, high=14.5, values=[13.0, 15.0],
                       flags=["normal", "high"])
        second = _trend("RBC", low=4.2, high=5.4, values=[5.0, 3.0],
                        flags=["normal", "low"], unit="x10^6/uL", decimals=2)
        ranked = followups._rank_abnormal([first, second])
        assert ranked[0]["label"] == "RBC"


# ------------------------------------- Fix 3: equipment is not a medication


class TestNonPharmacologicExclusion:
    REPLY = """{"medications": [
      {"medication": "Home oxygen", "dosage": "2 liters", "frequency": "continuous"},
      {"medication": "Atrovent", "dosage": "2 puffs", "frequency": "q.i.d."},
      {"medication": "Walker", "dosage": "", "frequency": ""},
      {"medication": "Coumadin", "dosage": "7 mg", "frequency": "daily"}
    ]}"""

    LIST = ("1. Home oxygen 2 liters continuous.\n"
            "2. Atrovent 2 puffs q.i.d.\n"
            "3. Walker.\n"
            "4. Coumadin 7 mg daily.")

    def test_equipment_is_dropped_and_drugs_survive(self):
        verified, dropped = deterministic.parse_medications(self.REPLY, self.LIST)
        assert [row["medication"] for row in verified] == ["Atrovent", "Coumadin"]
        assert [row["medication"] for row in dropped] == ["Home oxygen", "Walker"]
        assert {row["reason"] for row in dropped} == {
            "durable medical equipment or non-drug supply"
        }

    @pytest.mark.parametrize("name", [
        "Home oxygen", "O2", "CPAP", "BiPAP machine", "Incentive spirometer",
        "Glucometer", "Lancets", "Test strips", "Wheelchair", "Cane", "Crutches",
        "Commode", "Hospital bed", "Compression stockings", "TED hose",
    ])
    def test_the_equipment_vocabulary(self, name):
        assert deterministic._non_pharmacologic(name) is True

    @pytest.mark.parametrize("name", [
        "Albuterol nebulizer solution", "Spiriva", "Advair", "DuoNeb",
        "Prednisone", "Coumadin", "Dulcolax", "Insulin glargine",
    ])
    def test_a_drug_is_never_mistaken_for_equipment(self, name):
        assert deterministic._non_pharmacologic(name) is False

    def test_the_extraction_instruction_states_the_rule(self):
        text = " ".join(prompts.MEDS_FORMATTING.split())
        assert "Exclude equipment, gases and supplies" in text
        assert "inhaled or nebulised bronchodilator" in text


# ------------------------------------------- Fix 4: the chart's acronyms survive


class TestChartAcronyms:
    PASSAGE = ("62-year-old man with congestive heart failure (CHF) admitted with "
               "decompensated heart failure.")

    def test_the_definition_is_restored_on_first_mention(self):
        text, flags = answer_hygiene.restore_chart_acronyms(
            "He has congestive heart failure and was diuresed ^[1].",
            [{"section": "history_of_present_illness", "text": self.PASSAGE}])
        assert text.startswith("He has congestive heart failure (CHF) and was diuresed")
        assert flags == ("chart_acronym_restored:CHF",)

    def test_an_answer_that_kept_the_acronym_is_left_alone(self):
        prose = "He has CHF and was diuresed ^[1]."
        text, flags = answer_hygiene.restore_chart_acronyms(
            prose, [{"section": "history_of_present_illness", "text": self.PASSAGE}])
        assert text == prose
        assert flags == ()

    def test_an_acronym_the_initials_do_not_spell_is_ignored(self):
        text, _ = answer_hygiene.restore_chart_acronyms(
            "He has chronic kidney disease.",
            [{"section": "brief_hospital_course",
              "text": "History of chronic kidney disease (CHF)."}])
        assert "(" not in text

    def test_the_rule_is_stated_for_the_summary_command(self):
        text = " ".join(prompts.EHR_FORMATTING.split())
        assert "KEEP THE CHART'S OWN CLINICAL ABBREVIATIONS" in text
        assert "congestive heart failure (CHF)" in text

    def test_settlement_applies_it_on_a_summary_turn(self):
        settled = answer_postprocess.settle_answer(
            prose="He has congestive heart failure ^[1].",
            deterministic=None,
            tool_calls=_section_calls(
                {"section": "history_of_present_illness", "text": self.PASSAGE}),
            replayed_tool_calls=[],
            command="summarize",
        )
        assert "congestive heart failure (CHF)" in settled.prose
        assert "chart_acronym_restored:CHF" in settled.flags


# ------------------------------- Fix 5: an absence claim carries no stray marker


class TestAbsenceClaims:
    PROSE = ("The record contains no dedicated discharge medication section ^[1] ^[2].")

    def test_markers_into_other_sections_are_removed(self):
        settled = answer_postprocess.settle_answer(
            prose=self.PROSE,
            deterministic=None,
            tool_calls=_section_calls(
                {"section": "brief_hospital_course",
                 "text": "Given IV furosemide on the ward."},
                {"section": "discharge_instructions",
                 "text": "Follow up with cardiology in two weeks."}),
            replayed_tool_calls=[],
        )
        assert "^[" not in settled.prose
        assert "no dedicated discharge medication section" in settled.prose
        assert "absence_claim_citation_stripped:^1" in settled.flags
        assert "absence_claim_citation_stripped:^2" in settled.flags

    def test_the_section_the_sentence_names_keeps_its_marker(self):
        settled = answer_postprocess.settle_answer(
            prose="The record contains no dedicated discharge medication section ^[1].",
            deterministic=None,
            tool_calls=_section_calls(
                {"section": "discharge_medications",
                 "text": "Discharge Medications: none."}),
            replayed_tool_calls=[],
        )
        assert "^[1]" in settled.prose
        assert settled.flags == ()

    def test_a_sentence_citing_the_section_it_reports_is_untouched(self):
        """The Dev non-regression: `dev_t4_02` cites the section it is talking about."""
        settled = answer_postprocess.settle_answer(
            prose="The discharge instructions document no medications ^[1].",
            deterministic=None,
            tool_calls=_section_calls(
                {"section": "discharge_instructions",
                 "text": "Discharge Instructions: no medications were prescribed."}),
            replayed_tool_calls=[],
        )
        assert "^[1]" in settled.prose

    def test_the_two_sentence_rule_is_stated(self):
        text = " ".join(prompts.MEDS_SPOKE.split())
        assert "Write it in TWO SENTENCES" in text
        assert "with NO marker, that the record contains no dedicated" in text
        assert "carrying its own `^[k]`" in text


# ----------------------------------- Fix 6: percentages compared at their precision


class TestPercentagePrecision:
    def test_the_rule_is_stated_with_generic_figures(self):
        text = " ".join(prompts.LITERATURE_SPOKE.split())
        assert "COMPARE PERCENTAGES AT THE PRECISION THEY ARE WRITTEN" in text
        assert "15.2% is slightly higher than 15%" in text
        assert "never \"identical\" to it" in text
        assert "(15.0% and 15%)" in text


# --------------------------------- Fix 7: no panel-wide run of lab markers


class TestBlanketLabStatements:
    PANEL = [
        _trend("Sodium", low=135.0, high=145.0, values=[139.0, 141.0]),
        _trend("Potassium", low=3.5, high=5.0, values=[4.0, 4.2]),
        _trend("Creatinine", low=0.5, high=1.1, values=[0.9, 0.8]),
        _trend("Hemoglobin", low=13.0, high=17.0, values=[14.2, 13.6]),
        _trend("Hematocrit", low=38.0, high=50.0, values=[41.0, 40.0]),
    ]

    def test_the_panel_run_comes_off_a_sentence_that_reads_nothing(self):
        prose = ("Every laboratory value measured during this admission sat inside its "
                 "reference range ^[1]^[2]^[3]^[4]^[5]. Sodium drifted from 139 to 141 "
                 "mmol/L ^[1].")
        settled = answer_postprocess.settle_answer(
            prose=prose, deterministic=None, tool_calls=_lab_calls(*self.PANEL),
            replayed_tool_calls=[])
        first_sentence = settled.prose.split(".")[0]
        assert "^[" not in first_sentence
        assert "^[1]" in settled.prose.split(".")[1]
        assert any(flag.startswith("blanket_lab_marker_run_stripped")
                   for flag in settled.flags)

    def test_a_sentence_that_quotes_a_value_keeps_its_markers(self):
        prose = ("Sodium 139, potassium 4.0, creatinine 0.9 and hemoglobin 14.2 were all "
                 "in range ^[1]^[2]^[3]^[4].")
        settled = answer_postprocess.settle_answer(
            prose=prose, deterministic=None, tool_calls=_lab_calls(*self.PANEL),
            replayed_tool_calls=[])
        assert "^[4]" in settled.prose
        assert settled.flags == ()

    def test_the_only_citation_in_the_answer_is_never_removed(self):
        prose = "Every laboratory value sat inside its reference range ^[1]^[2]^[3]^[4]."
        _text, flags = answer_hygiene.strip_blanket_lab_marker_runs(
            prose, tool_calls=_lab_calls(*self.PANEL))
        assert flags == ()

    def test_the_two_part_rule_is_stated(self):
        text = " ".join(prompts.LABS_SPOKE.split())
        assert "carries NO `^[n]` marker" in text
        assert "two notable within-range movements" in text
        assert "Never hang a run of markers on a panel-wide statement" in text


# ----------------------------------------------------------------- prompt budgets


def test_the_prompt_budgets_hold():
    assert len(prompts.CORE_HUB_PROMPT.split()) <= 1500
    for name in ("CHART_SPOKE", "MEDS_SPOKE", "LABS_SPOKE", "RISK_SPOKE",
                 "LITERATURE_SPOKE"):
        assert len(getattr(prompts, name).split()) <= 650, name


# --------------------------------------------------------- the anti-leakage guard


#: The forms that could only have come from the held-out suite: its patient's condition
#: name, the trial drugs, and its figures as percentages. Named here only so this test can
#: prove none of them reached the code; nothing here reads a dataset.
HOLDOUT_LITERALS = ("Ewing", "ifosfamide", "mesna", "23.2%", "16.6%", "3.06", "11.2%")

#: Clinical vocabulary that exists in this tree for reasons of its own and predates the
#: held-out suite. It is not a leak, and asserting its absence would be false. The two
#: modules this stage added must not reach for it either, which is what the second half
#: of the test pins.
STAGE_11B_MODULES = ("agent/postprocess/answer_hygiene.py",)


def test_no_holdout_literal_reached_the_services_tree():
    offenders: list[str] = []
    for path in sorted(SERVICES.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        source = path.read_text(errors="ignore")
        for literal in HOLDOUT_LITERALS:
            if literal in source:
                offenders.append(f"{path.relative_to(SERVICES)}: {literal}")
    assert offenders == [], "holdout literals in services/: " + ", ".join(offenders)


def test_this_stage_introduced_no_clinical_literals_of_its_own():
    """The new module stays generic: no diagnosis and no cohort figure of its own."""
    for relative in STAGE_11B_MODULES:
        source = (SERVICES / relative).read_text(errors="ignore")
        assert "COPD" not in source, relative
        assert "Ewing" not in source, relative
