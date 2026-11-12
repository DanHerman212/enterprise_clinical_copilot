"""Stage 12C: the hacks are gone, and the clean rules hold.

Two kinds of test. The first asserts absence: the two postprocess modules this stage
deleted are not in the tree, and nothing imports them. The rest assert the behaviour that
replaced them, at the layer that owns it — the citation owner for a panel-wide run of
laboratory markers, the presentation layer for the abstract a citation card carries, and
the medication narrative for a numbered list that holds a supply beside the drugs.

No evaluation dataset is read here. The held-out literals are named only so the sweep can
prove none of them reached `services/`.
"""

from __future__ import annotations

import pathlib

import pytest

from services.agent.engine import deterministic, prompts
from services.agent.postprocess import answer as answer_text
from services.agent.postprocess import citations
from services.agent.presentation import a2ui
from shared import labels

REPO = pathlib.Path(__file__).resolve().parents[2]
SERVICES = REPO / "services"


# ---------------------------------------------------------------- the hacks are gone


@pytest.mark.parametrize("relative", [
    "services/agent/postprocess/answer_hygiene.py",
    "services/agent/postprocess/risk_lab_citation.py",
])
def test_the_postprocess_hack_module_is_deleted(relative):
    assert not (REPO / relative).exists(), relative


def test_no_module_imports_a_deleted_hack():
    offenders = []
    for path in sorted(SERVICES.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        source = path.read_text(errors="ignore")
        for name in ("answer_hygiene", "risk_lab_citation"):
            if name in source:
                offenders.append(f"{path.relative_to(SERVICES)}: {name}")
    assert offenders == [], ", ".join(offenders)


def test_the_extraction_verifies_every_numbered_line():
    """A supply is an ORDER, not a row to refuse.

    The count a reader can check is the numbered list on the page. A module that drops a
    line and reports the remainder gives a number the page contradicts, and the oxygen
    order disappears from a list the record wrote out in full.
    """
    reply = ('{"medications": ['
             '{"medication": "Home oxygen", "dosage": "2 liters", "frequency": "continuous"},'
             '{"medication": "Atrovent", "dosage": "2 puffs", "frequency": "q.i.d."},'
             '{"medication": "Coumadin", "dosage": "7 mg", "frequency": "daily"}]}')
    note = ("1. Home oxygen 2 liters continuous.\n"
            "2. Atrovent 2 puffs q.i.d.\n"
            "3. Coumadin 7 mg daily.")
    verified, dropped = deterministic.parse_medications(reply, note)
    assert [row["medication"] for row in verified] == ["Home oxygen", "Atrovent", "Coumadin"]
    assert dropped == []


# ------------------------------------------- a panel-wide statement carries no markers


def _lab_panel() -> list[dict]:
    trends = []
    for key, label, unit, decimals, low, high in (
        ("rdw", "RDW", "%", 1, 11.5, 14.5),
        ("rbc", "Red blood cells", "x10^6/uL", 2, 4.2, 5.4),
        ("hemoglobin", "Hemoglobin", "g/dL", 1, 12.0, 15.5),
        ("sodium", "Sodium", "mmol/L", 0, 135.0, 145.0),
        ("monocytes", "Monocytes", "x10^3/uL", 1, 0.2, 0.8),
        ("creatinine", "Creatinine", "mg/dL", 2, 0.5, 1.1),
        ("egfr", "eGFR", "mL/min/1.73m2", 0, 90.0, 120.0),
        ("potassium", "Potassium", "mmol/L", 1, 3.5, 5.0),
        ("wbc", "White blood cells", "x10^3/uL", 1, 4.0, 11.0),
    ):
        trends.append({
            "key": key, "label": label, "unit": unit, "decimals": decimals,
            "reference_low": low, "reference_high": high, "anchored": [],
            "series": [{"day": 0.0, "value": (low + high) / 2, "flag": "normal"},
                       {"day": 2.0, "value": (low + high) / 2, "flag": "normal"}],
        })
    return [{"name": "get_lab_trends", "args": {"hadm_id": 90000042},
             "response": {"hadm_id": 90000042, "synthetic": True, "returned": 9,
                          "trends": trends}}]


BLANKET_ANSWER = (
    "All laboratory values measured at discharge were within their respective reference "
    "ranges ^[1] ^[2] ^[3] ^[4] ^[5] ^[6] ^[7] ^[8] ^[9]."
)


def test_a_single_sentence_panel_statement_settles_to_zero_markers():
    """Nine badges on a claim no single series makes, on an answer that is one sentence.

    A `/labs` answer is grounded by the drawn table, not by prose markers, so the strip is
    unconditional: keeping nine badges to avoid an empty `sources` list keeps the defect
    the badges caused.
    """
    settled = answer_text.settle_answer(
        prose=BLANKET_ANSWER, deterministic=None, tool_calls=_lab_panel(),
        replayed_tool_calls=[], evidence_state=None, command=None,
    )
    assert "^[" not in (settled.prose or "")
    assert citations.PANEL_WIDE_LAB_RUN_FLAG in settled.flags


def test_a_named_analyte_keeps_its_marker():
    """The rule is about a PANEL statement, not about laboratory markers as such.

    The marker survives the strip; the page's numbering then compacts it to `^[1]` and
    publishes the map back to position 8, which is the potassium series.
    """
    settled = answer_text.settle_answer(
        prose="Potassium was 4.1 mmol/L at discharge ^[8].", deterministic=None,
        tool_calls=_lab_panel(), replayed_tool_calls=[], evidence_state=None,
    )
    assert len(citations.cited_numbers(settled.prose or "")) == 1
    assert settled.display == {"1": 8}
    assert citations.PANEL_WIDE_LAB_RUN_FLAG not in settled.flags


def test_a_named_list_of_five_analytes_loses_its_run_of_five_markers():
    """Naming the values does not earn a badge per value.

    Five normal analytes and five markers is a table written as prose, which is the same
    statement the drawn panel makes without a badge on every row. A sentence citing one to
    three series is a sentence about those values and keeps its markers.
    """
    prose = ("RDW 12.8%, red cells 4.3 x10^6/uL, sodium 136 mmol/L, creatinine 0.9 mg/dL "
             "and eGFR 101 mL/min/1.73m2 were all within their reference ranges "
             "^[5] ^[6] ^[7] ^[8] ^[9].")
    settled = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=_lab_panel(),
        replayed_tool_calls=[], evidence_state=None,
    )
    assert "^[" not in (settled.prose or "")
    assert citations.PANEL_WIDE_LAB_RUN_FLAG in settled.flags


def test_a_sentence_citing_three_series_keeps_all_three():
    prose = ("RDW, red cells and hemoglobin were all within their reference ranges "
             "^[1] ^[2] ^[3].")
    settled = answer_text.settle_answer(
        prose=prose, deterministic=None, tool_calls=_lab_panel(),
        replayed_tool_calls=[], evidence_state=None,
    )
    assert len(citations.cited_numbers(settled.prose or "")) == 3
    assert citations.PANEL_WIDE_LAB_RUN_FLAG not in settled.flags


def test_a_free_text_panel_question_draws_the_table_and_stays_verified():
    out = a2ui.compose_presentation(
        "Were any of his discharge labs abnormal?",
        "All laboratory values measured at discharge were within their respective "
        "reference ranges.",
        _lab_panel(),
    )
    assert out["verified"] is True
    assert out["sources"] == []
    types = [component["component"]
             for component in out["a2ui"]["messages"][1]["updateComponents"]["components"]]
    assert "LabTrendCard" in types


def test_a_question_about_neither_analyte_nor_panel_draws_nothing():
    out = a2ui.compose_presentation(
        "Which issue should be treated first?",
        "The record does not state a priority.",
        _lab_panel(),
    )
    types = [component["component"]
             for component in out["a2ui"]["messages"][1]["updateComponents"]["components"]]
    assert "LabTrendCard" not in types


# ------------------------------------------------------- the abstract reaches the card


def _structured_abstract() -> str:
    """A structured abstract of the shape PubMed serves, with its figures in RESULTS.

    Synthetic throughout: a heart-failure admission cohort with two arms. The shape is what
    this test needs — a `RESULTS` label past the old 400-character bound, carrying the two
    figures the answer would quote — and the content belongs to no record in any suite.
    """
    body = (
        "OBJECTIVES/HYPOTHESIS: Heart failure is the leading cause of hospital admission in "
        "older adults, and the burden of a concurrent renal injury on length of stay, cost, "
        "and readmission is incompletely described. We hypothesized that patients with an "
        "acute kidney injury during the admission would have longer stays and higher "
        "readmission than those without. STUDY DESIGN: Retrospective parallel cohort study. "
        "METHODS: A study of patients admitted to the hospital with decompensated heart "
        "failure was completed. Two groups were identified and compared: those with a coded "
        "diagnosis of acute kidney injury (n = 210) in addition to heart failure, and a "
        "group with heart failure alone (n = 1,940). Number of inpatient days, cost per "
        "patient, diagnosis of pneumonia, 30-day readmission, and mortality rates were "
        "compared. RESULTS: For those in the renal-injury group, mean age was 81 years "
        "compared to 76 years in the group without it (P < .05); length of stay was 9 days, "
        "more than twice that of the group without it (4 days) (P < .05). Mortality within "
        "30 days of admission was higher (12% vs. 3%, respectively) and the 30-day "
        "readmission rate differed as well (6% vs. 9%, respectively). Rate of inpatient "
        "dialysis was 11 times greater in the renal-injury group (4.4%) compared with the "
        "group without it (0.4%). CONCLUSIONS: Acute kidney injury during a heart-failure "
        "admission increases length of stay and cost. It is associated with higher "
        "readmission and greater mortality. LEVEL OF EVIDENCE: 2b."
    )
    assert len(body) > 1200
    return body


def _synthetic_article(abstract: str) -> dict:
    return {"pmid": "99999999", "title": "Renal injury during heart-failure admission",
            "journal": "Journal of Hospital Medicine", "pub_date": "2021",
            "abstract": abstract,
            "url": "https://pubmed.ncbi.nlm.nih.gov/99999999/"}


def test_the_citation_card_carries_the_abstract_to_its_results():
    abstract = _structured_abstract()
    citation = a2ui.literature_citations([_synthetic_article(abstract)])[0]
    assert citation["excerpt"] == abstract
    assert "RESULTS" in citation["excerpt"]
    assert "6% vs. 9%" in citation["excerpt"]


def test_the_sources_entry_carries_it_too():
    abstract = _structured_abstract()
    sources = a2ui.literature_sources([_synthetic_article(abstract)])
    assert sources[0]["text"] == abstract


def test_the_abstract_bound_is_the_only_thing_that_limits_it():
    assert a2ui._CITE_EXCERPT_CHARS == 2400
    overlong = "BACKGROUND: " + ("x" * 3000)
    excerpt = a2ui._excerpt(overlong)
    assert len(excerpt) <= 2401


# --------------------------------------------- the numbered list and the drug count


SIXTEEN_ITEM_LIST = (
    "1. Home oxygen 1 to 2 liters to maintain O2 saturations at 89 to 91% at all times.\n"
    "2. Ativan 1 mg p.o. t.i.d.\n"
    "3. Metformin 1000 mg p.o. b.i.d.\n"
    "4. Glucotrol 5 mg p.o. daily.\n"
    "5. Spiriva 1 puff b.i.d.\n"
    "6. Lantus 25 units subcu q.a.m.\n"
    "7. Cardizem CD 180 mg p.o. q.a.m.\n"
    "8. Advair 250/50 mcg, 1 puff b.i.d.\n"
    "9. Iron 325 mg p.o. b.i.d.\n"
    "10. Aspirin 325 mg p.o. daily.\n"
    "11. Lipitor 10 mg p.o. bedtime.\n"
    "12. Digoxin 0.25 mg p.o. daily.\n"
    "13. Lisinopril 20 mg p.o. q.a.m.\n"
    "14. DuoNeb every 4 hours for the next several weeks.\n"
    "15. Prednisone 40 mg p.o. q.a.m. x3 days, then 30 mg p.o. q.a.m. x3 days.\n"
    "16. Clindamycin 300 mg p.o. q.i.d. x2 weeks."
)

SIXTEEN_ITEM_REPLY = """{"medications": [
  {"medication": "Home oxygen", "dosage": "1 to 2 liters", "frequency": "continuous"},
  {"medication": "Ativan", "dosage": "1 mg", "frequency": "t.i.d."},
  {"medication": "Metformin", "dosage": "1000 mg", "frequency": "b.i.d."},
  {"medication": "Glucotrol", "dosage": "5 mg", "frequency": "daily"},
  {"medication": "Spiriva", "dosage": "", "frequency": "1 puff b.i.d."},
  {"medication": "Lantus", "dosage": "25 units", "frequency": "subcu q.a.m."},
  {"medication": "Cardizem CD", "dosage": "180 mg", "frequency": "q.a.m."},
  {"medication": "Advair", "dosage": "250/50 mcg", "frequency": "1 puff b.i.d."},
  {"medication": "Iron", "dosage": "325 mg", "frequency": "b.i.d."},
  {"medication": "Aspirin", "dosage": "325 mg", "frequency": "daily"},
  {"medication": "Lipitor", "dosage": "10 mg", "frequency": "bedtime"},
  {"medication": "Digoxin", "dosage": "0.25 mg", "frequency": "daily"},
  {"medication": "Lisinopril", "dosage": "20 mg", "frequency": "q.a.m."},
  {"medication": "DuoNeb", "dosage": "", "frequency": "every 4 hours"},
  {"medication": "Prednisone", "dosage": "40 mg", "frequency": "q.a.m. x3 days"},
  {"medication": "Clindamycin", "dosage": "300 mg", "frequency": "q.i.d. x2 weeks"}
]}"""


def _sixteen_rows() -> tuple[list[dict], list[dict]]:
    return deterministic.parse_medications(SIXTEEN_ITEM_REPLY, SIXTEEN_ITEM_LIST)


def test_the_whole_numbered_list_is_verified():
    verified, dropped = _sixteen_rows()
    assert len(verified) == 16
    assert dropped == []


def test_the_narrative_counts_the_drugs_and_names_the_supply():
    """Neither the drug count nor the list a reader can count is contradicted."""
    verified, _ = _sixteen_rows()
    narrative = answer_text.medication_narrative(
        verified, 3, sections=("discharge_medications",))
    assert narrative.startswith(
        "15 discharge medications (plus Home oxygen ordered as equipment/supply, "
        "16 listed items in total), as recorded in the discharge note:")
    assert "Clindamycin" in narrative
    assert "Home oxygen" not in narrative.split("as recorded in the discharge note:")[1]


def test_a_list_with_no_supply_reads_exactly_as_it_did():
    rows = [{"medication": "Coumadin", "dosage": "7 mg", "frequency": "daily"},
            {"medication": "Aspirin", "dosage": "325 mg", "frequency": "daily"}]
    narrative = answer_text.medication_narrative(rows, 1, sections=("discharge_medications",))
    assert narrative.startswith(
        "2 discharge medications, as recorded in the discharge note: Coumadin, Aspirin")


def test_the_dosing_chip_never_asks_about_a_supply():
    from services.agent.postprocess import followups  # noqa: PLC0415

    state = {"tool_calls": [{"name": "discharge_medications", "response": {
        "returned": 1, "passages": [{"section": "discharge_medications",
                                     "text": "SIXTEEN"}]}}],
        "command_payload": {"medications": [
            {"medication": "Home oxygen", "dosage": "2 liters", "frequency": "continuous"},
            {"medication": "Oxycodone SR", "dosage": "20 mg", "frequency": "q.12.h."},
        ]}}
    rows = followups._medication_rows(state)
    assert rows == ["Oxycodone SR"]


@pytest.mark.parametrize("name", [
    "Home oxygen", "O2", "CPAP", "BiPAP machine", "Incentive spirometer", "Glucometer",
    "Lancets", "Test strips", "Wheelchair", "Cane", "Crutches", "Commode", "Hospital bed",
    "Compression stockings", "TED hose",
])
def test_the_equipment_classifier_recognises_equipment(name):
    assert labels.is_equipment_or_supply(name) is True


@pytest.mark.parametrize("name", [
    "Albuterol nebulizer solution", "Spiriva", "Advair", "DuoNeb", "Prednisone",
    "Coumadin", "Dulcolax", "Insulin glargine",
])
def test_the_equipment_classifier_never_catches_a_drug(name):
    assert labels.is_equipment_or_supply(name) is False


# ---------------------------------------------------------------- the prompt contract


def test_the_prompt_budgets_hold():
    assert len(prompts.CORE_HUB_PROMPT.split()) <= 1500
    for name in ("CHART_SPOKE", "MEDS_SPOKE", "LABS_SPOKE", "RISK_SPOKE",
                 "LITERATURE_SPOKE"):
        assert len(getattr(prompts, name).split()) <= 650, name


def test_the_hub_forbids_a_marker_on_the_models_own_numbers():
    text = " ".join(prompts.CORE_HUB_PROMPT.split())
    assert "including one restating an estimate the thread established" not in text
    assert "An estimate, driver, contribution or model `value` carries none" in text
    assert "nor does a statement that the whole panel was normal" in text


def test_the_risk_spoke_keeps_an_attribution_out_of_a_cited_sentence():
    text = " ".join(prompts.RISK_SPOKE.split())
    assert "carries NO marker" in text
    assert "and any diagnosis or series supporting it gets its own `^[n]`-cited sentence" in text


def test_the_lab_spoke_scopes_the_analyte_rule_and_the_yes_no_answer():
    text = " ".join(prompts.LABS_SPOKE.split())
    assert "A sentence naming two or three specific analytes cites each analyte's series" in text
    assert "answer that in one sentence carrying NO `^[n]` marker" in text
    assert "do not append unrequested within-range movements" in text


def test_the_literature_spoke_maps_each_arm_to_its_group():
    text = " ".join(prompts.LITERATURE_SPOKE.split())
    assert "A TWO-ARM RESULT KEEPS ITS ARMS" in text
    assert "never collapse the pair into a range" in text
    assert "15.2% is slightly higher than 15%" in text


def test_the_ehr_formatter_reports_the_attribution_without_editorialising():
    text = " ".join(prompts.EHR_FORMATTING.split())
    assert "REPORT THE MODEL'S ATTRIBUTION AND STOP" in text
    for editorial in (
        "what the named drivers mean clinically",
        "Stop after saying what this patient's drivers mean for the follow-up",
        "what the drivers mean for the follow-up plan",
    ):
        assert editorial not in text, editorial
    assert 'Do NOT call\nsuch a factor "favorable"' not in text  # newlines were joined
    assert '"favorable"' in text
    assert "never \"favorable\"" not in text
    assert "a 6% to 9% range" in text
    assert "KEEP THE CHART'S OWN CLINICAL ABBREVIATIONS" in text


# --------------------------------------------------------- the anti-leakage guard


HOLDOUT_LITERALS = ("Ewing", "ifosfamide", "mesna", "23.2%", "16.6%", "3.06", "11.2%",
                    "31314145", "Laryngoscope")


def test_no_holdout_literal_reached_the_services_tree():
    offenders = []
    for path in sorted(SERVICES.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        source = path.read_text(errors="ignore")
        for literal in HOLDOUT_LITERALS:
            if literal in source:
                offenders.append(f"{path.relative_to(SERVICES)}: {literal}")
    assert offenders == [], "holdout literals in services/: " + ", ".join(offenders)
