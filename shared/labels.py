"""Human labels for risk-model feature keys.

The one vocabulary that has to agree across three layers: the widget draws a driver's
bar with `humanize`, the prose names the same factor with the same string, and the med
and lab guards read keys by the label they were shown. It lived in
`presentation/domain.py`, which meant `engine/mcp_client.py` and
`postprocess/followups.py` imported the presentation layer to name a feature — the two
reverse edges that stopped those packages being testable on their own.

Nothing here imports anything from `services.agent`: it is a leaf, so every layer may
depend on it and the dependency graph stays one-way.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: The curated map. A key absent from it is title-cased by `humanize`, and reported
#: unchanged by `label_for` — two different fallbacks, both deliberate: display wants
#: something readable, and a guard that prints a key wants the key.
FEATURE_LABELS: dict[str, str] = {
    "age": "age",
    "gender": "sex",
    "has_procedure": "having a procedure",
    "procedure_count": "procedures",
    "oncology_flag": "oncology history",
    "medication_count": "medication count",
    "medication_order_count": "medication order count",
    "prior_inpatient_days": "prior inpatient days",
    "prior_admission_count": "prior admissions",
    "index_los_days": "length of stay",
    "recent_ed_visits": "recent ED visits",
    "hemoglobin_min": "lowest hemoglobin",
    "sodium_min": "lowest sodium",
    "sodium_max": "highest sodium",
    "sodium_last": "recent sodium",
    "rbc_min": "lowest red blood cell count",
    "rbc_last": "recent red blood cell count",
    "rdw_max": "red cell distribution width (RDW)",
    "monocytes_min": "lowest monocyte count",
    "race": "race",
    "admission_type": "admission type",
    "discharge_location": "discharge destination",
    "insurance": "insurance type",
}


def label_for(feature: str) -> str:
    """Human label for a feature key; falls back to the raw key if unknown."""
    return FEATURE_LABELS.get(feature, feature)


def humanize(feature: str) -> str:
    """Best human label for a feature key, for visual consumption.

    1. Known parent feature -> curated label (FEATURE_LABELS).
    2. Otherwise normalize the raw key to Title Case, handling both camelCase and
       snake_case: medicationOrderCount and medication_order_count both become
       "Medication Order Count". This is the safety net for any feature not in the
       curated map.
    """
    known = FEATURE_LABELS.get(feature)
    if known is not None:
        return known
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", feature)
    words = [word for word in re.split(r"[\s_\-]+", spaced) if word]
    return " ".join(word.capitalize() for word in words) if words else feature


# --- The question both driver affordances ask ------------------------------
#
# A ranked driver can be asked about from two controls: the follow-up chip under an
# answer, and a bar inside the risk card. They used to ask different things — the chip
# "Why does X increase this patient's readmission risk?", the bar "What in this patient's
# chart explains X as a readmission risk driver?" — and the same evidence produced two
# different shapes of answer, because the chip asked for the model's reasoning, which the
# prompt forbids explaining, and the bar asked for the chart's support, which the record
# can supply (measured 2026-11-20).
#
# One question now, and one that a record can answer: what does the factor DO to the
# estimate, and what stands behind it.
#
# The verb agrees with the label, which is why the labels matter here as words rather
# than as keys: `humanize` renders four features as plurals, and "How does prior inpatient
# days affect" is not a sentence anyone wrote. The set is listed rather than derived from
# a trailing "s", because a rule that guessed would be a rule the next label could
# falsify — and `static/js/a2ui/risk_influencers.js` reads the same four strings, since
# the widget is given the label and never the feature key.
PLURAL_LABELS: frozenset[str] = frozenset({
    "procedures",
    "prior inpatient days",
    "prior admissions",
    "recent ED visits",
})


def label_is_plural(label: str) -> bool:
    """Whether a rendered feature label names several things."""
    return str(label or "").strip() in PLURAL_LABELS


def driver_question(label: str) -> str:
    """The question a driver chip and a driver bar both ask, or "" without a label."""
    text = str(label or "").strip()
    if not text:
        return ""
    verb = "do" if label_is_plural(text) else "does"
    return f"How {verb} {text} affect this patient's readmission risk?"


# --- The analyte vocabulary ------------------------------------------------
#
# The names a clinician uses for the values a laboratory series can report. One list,
# because two layers ask the same question of it: `routing` decides whether
# `get_lab_trends` may be bound, and the presentation layer decides whether a turn that
# read the series should draw it. It lived in `routing/`, which presentation may not
# import (Charter 13 §5) — the second caller would have had to keep a copy, and a
# duplicated vocabulary drifts exactly the way duplicated tool registries do.
#
# The words are the model's feature language and the panel's labels, which is why most
# carry five-letter singular stems. `\b` after a singular stem has no boundary before
# the plural "s", so every word is compiled with an optional plural: on 2026-10-25 a
# `/labs` follow-up reading "Red blood cells (3.47 x10^6/uL)" was not recognised, and
# eight of the panel's ten labels could not satisfy a singular-only pattern.
LAB_ANALYTE_WORDS = (
    # The five the model scores on, whose driver drill-downs are the other caller.
    "hemoglobin", "haemoglobin", "hgb", "hb",
    "sodium",
    "red blood cell", "rbc", "erythrocyte",
    "rdw", "red cell distribution width",
    "monocyte",
    # The clinical panel: carried by the series, never model inputs.
    "creatinine",
    "egfr", "gfr",
    "potassium",
    "white blood cell", "wbc", "leukocyte",
    "nt-probnp", "bnp",
    # The collective and the panel. A clinician asks to compare "his discharge blood
    # counts" without naming hemoglobin, and a question with no analyte name in it lost the
    # series: the only domain cue left was the word "stay", which is a chart cue, and one
    # domain narrows the tool set — so `get_lab_trends` was dropped from a question whose
    # whole subject was laboratory values (`dev_d5_fup_02`, 2026-11-23). These name the
    # panel rather than one column, so they bind the series and canonicalise to no single
    # analyte.
    "blood count", "blood work", "lab value", "lab values",
)

_LAB_ANALYTE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) + r"s?" for word in LAB_ANALYTE_WORDS) + r")\b",
    re.IGNORECASE,
)


def names_an_analyte(text: str) -> bool:
    """Whether this text names a laboratory measurement rather than a count.

    Negative for anything that names no analyte, which is the safe direction in both
    callers: a tool is withheld rather than bound, and a card is not drawn rather than
    drawn unasked.
    """
    return bool(_LAB_ANALYTE.search(text or ""))


#: The PANEL key each vocabulary word answers to.
#:
#: The words above name a measurement as a clinician writes it — "lowest monocyte count",
#: "red cell distribution width" — and the panel names the column it draws: `monocytes`,
#: `rdw`. One is not the other, and the difference reached a live tool call: a driver
#: question about the lowest monocyte count asked `get_lab_trends` for `monocytes_min`, the
#: feature the model was scored on, and was refused (measured 2026-11-20, 90000015).
#:
#: Two words can name one column: "hgb" and "haemoglobin" are the same measurement, and a
#: question that says both reads once.
_PANEL_FOR_WORD: dict[str, str] = {
    "hemoglobin": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "sodium": "sodium",
    "red blood cell": "rbc",
    "rbc": "rbc",
    "erythrocyte": "rbc",
    "rdw": "rdw",
    "red cell distribution width": "rdw",
    "monocyte": "monocytes",
    "creatinine": "creatinine",
    "egfr": "egfr",
    "gfr": "egfr",
    "potassium": "potassium",
    "white blood cell": "wbc",
    "wbc": "wbc",
    "leukocyte": "wbc",
    "nt-probnp": "nt_probnp",
    "bnp": "nt_probnp",
}


def _panel_key(word: str) -> str | None:
    """The panel key for one vocabulary word, singular or plural.

    `_LAB_ANALYTE` matches a trailing `s`, so it returns "monocytes" where this table is
    keyed on "monocyte". Stripping the plural here rather than storing both spellings is
    what keeps one entry per measurement.
    """
    lowered = str(word or "").strip().lower()
    if lowered in _PANEL_FOR_WORD:
        return _PANEL_FOR_WORD[lowered]
    if lowered.endswith("s") and lowered[:-1] in _PANEL_FOR_WORD:
        return _PANEL_FOR_WORD[lowered[:-1]]
    return None


def canonical_analyte(text: str) -> str | None:
    """The panel key for a name of a laboratory measurement, or None.

    Accepts either a whole phrase ("lowest monocyte count") or a vocabulary word ("rdw"),
    because both reach it: a question is a sentence and a chip label is a phrase.
    """
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return None
    direct = _panel_key(lowered)
    if direct:
        return direct
    match = _LAB_ANALYTE.search(lowered)
    if match is None:
        return None
    return _panel_key(match.group(0))


def analytes_canonical(text: str) -> tuple[str, ...]:
    """Every panel analyte this text names, in the order it names them, once each.

    The list a caller uses when it has to act PER analyte — one series to read, one card
    to draw — rather than decide a yes/no. "The lowest monocyte count and the RDW" is two
    analytes and two reads.
    """
    found: list[str] = []
    for word in analytes_named(text):
        key = _panel_key(str(word))
        if key and key not in found:
            found.append(key)
    return tuple(found)


def analytes_named(text: str) -> tuple[str, ...]:
    """Every analyte this text names, as the vocabulary spells it, once each, in order.

    The list form of `names_an_analyte`, for the caller that has to do something PER
    analyte rather than decide a yes/no: a question asking about hemoglobin and creatinine
    needs both read, and a turn that read one of them and then reported the other as
    undocumented has stated an absence it never looked for (measured 2026-11-06 on
    `90000069`: `get_lab_trends(analyte="hemoglobin")`, and the served answer said "the
    record does not document a creatinine trend").

    The vocabulary is the panel's, so a bundle naming the same value twice ("hb",
    "hemoglobin") is one analyte here when the canonical words match. Synonyms that spell
    the same measure differently are separate entries on purpose: this returns what the
    text NAMED, and deciding that "hgb" and "hemoglobin" are one measurement is the
    retrieval layer's judgement, not this function's.
    """
    found: list[str] = []
    seen: set[str] = set()
    for match in _LAB_ANALYTE.finditer(text or ""):
        word = match.group(0).lower()
        if word in seen:
            continue
        seen.add(word)
        found.append(word)
    return tuple(found)


# --- The laboratory-question vocabulary -------------------------------------
#
# A question can be about the laboratory panel without naming an analyte: "were any of
# his discharge labs abnormal?" and "what did the blood work show?" are panel questions,
# and a surface that only recognised analyte names drew nothing for either. The vocabulary
# is here because `presentation` and `postprocess` both need it and neither may import the
# other.

#: The nouns a clinician uses for the panel as a whole.
LAB_PANEL_NOUNS = re.compile(
    r"\b(?:labs?|laboratory|bloodwork|blood\s+work|panel)\b",
    re.IGNORECASE,
)


def asks_about_labs(text: str) -> bool:
    """Whether this text asks about the laboratory panel rather than one measurement.

    Deliberately narrow: the nouns above, and nothing else. A question about a treatment,
    an issue or a diagnosis is not a laboratory question however clinical it reads, and a
    false positive draws a trajectory table under an answer that never mentioned one.
    """
    return bool(LAB_PANEL_NOUNS.search(text or ""))


# --- The equipment / supply vocabulary -------------------------------------
#
# A numbered discharge list also numbers what the patient takes home but does not swallow.
# One list, asked by three callers, because the alternative is three copies that drift:
# `/meds` reports which numbered items are supplies rather than prescriptions, the
# follow-up chips never offer a dosing question about oxygen, and the extraction question
# is phrased with this vocabulary in mind.

#: Equipment and supplies whose meaning spans two words, so a whole-word test misses them.
EQUIPMENT_OR_SUPPLY_PHRASES: tuple[str, ...] = (
    "home oxygen", "supplemental oxygen", "portable oxygen", "oxygen therapy",
    "oxygen concentrator", "nasal cannula", "incentive spirometer", "nebulizer machine",
    "hospital bed", "compression stockings", "compression stocking", "ted hose",
    "test strips", "glucose strips", "blood pressure cuff", "suction machine",
    "feeding pump", "shower chair", "grab bar",
)

#: Equipment named by one word. A whole-word test, so "o2" matches and "oxygenation" does
#: not.
#:
#: `nebulizer` and `stockings` are deliberately absent as single words: "albuterol
#: nebulizer solution" names a drug and "compression stockings" names a supply, and only
#: the second is in the phrase list above. Classifying a bronchodilator as equipment
#: because its route is spelled out would keep a real prescription out of a dosing answer.
EQUIPMENT_OR_SUPPLY_WORDS: frozenset[str] = frozenset({
    "oxygen", "o2", "cpap", "bipap", "spirometer", "glucometer", "lancet", "lancets",
    "walker", "wheelchair", "cane", "crutches", "commode",
})


def is_equipment_or_supply(name: str) -> bool:
    """Whether a medication-list row names equipment, a gas or a supply rather than a drug.

    A classification, never a deletion: a line the chart numbered is reported to the
    reader, and what this decides is whether it is counted as a prescription when a
    sentence or a chip has to distinguish the two.
    """
    text = f" {' '.join(str(name or '').split()).lower()} "
    if any(f" {phrase} " in text for phrase in EQUIPMENT_OR_SUPPLY_PHRASES):
        return True
    words = re.split(r"[^a-z0-9]+", text)
    return any(word in EQUIPMENT_OR_SUPPLY_WORDS for word in words if word)


# --- Inline citation pill & section labels ---------------------------------
# Shared across presentation (`a2ui.py`) and engine/telemetry streaming (`stages.py`,
# `deterministic.py`, `graph.py`) so live SSE drafts can paint exact section/journal
# pill names from the first token rather than numbered `[1]`/`[7]` placeholders.

SECTION_LABELS: dict[str, str] = {
    "name": "Patient name",
    "unit_no": "Unit number",
    "admission_date": "Admission date",
    "discharge_date": "Discharge date",
    "date_of_birth": "Date of birth",
    "sex": "Sex",
    "service": "Service",
    "allergies": "Allergies",
    "attending": "Attending physician",
    "chief_complaint": "Chief complaint",
    "major_procedure": "Major surgical or invasive procedure",
    "history_of_present_illness": "History of present illness",
    "review_of_systems": "Review of systems",
    "past_medical_history": "Past medical history",
    "past_surgical_history": "Past surgical history",
    "social_history": "Social history",
    "family_history": "Family history",
    "physical_exam": "Physical exam",
    "pertinent_results": "Pertinent results",
    "brief_hospital_course": "Brief hospital course",
    "medications_on_admission": "Medications on admission",
    "medications_on_transfer": "Medications on transfer",
    "discharge_medications": "Discharge medications",
    "discharge_disposition": "Discharge disposition",
    "admission_diagnosis": "Admission diagnosis",
    "discharge_diagnosis": "Discharge diagnosis",
    "discharge_condition": "Discharge condition",
    "discharge_instructions": "Discharge instructions",
    "discharge_summary": "Discharge summary",
    "followup_instructions": "Follow-up instructions",
    "facility": "Facility",
    "complications": "Complications",
    "consults": "Consultations",
    "assessment_and_plan": "Assessment and plan",
    "pathology": "Pathology",
    "infections": "Infections",
    "transfusions": "Blood transfusions",
    "indications": "Indications for the surgery",
    "laboratory_trends": "Laboratory trends",
}

PILL_SECTION_LABELS: dict[str, str] = {
    "name": "Patient Name",
    "unit_no": "Unit Number",
    "admission_date": "Admission Date",
    "discharge_date": "Discharge Date",
    "date_of_birth": "Date of Birth",
    "sex": "Sex",
    "service": "Service",
    "allergies": "Allergies",
    "attending": "Attending Physician",
    "chief_complaint": "Chief Complaint",
    "major_procedure": "Major Procedure",
    "history_of_present_illness": "History of Present Illness",
    "review_of_systems": "Review of Systems",
    "past_medical_history": "Past Medical History",
    "past_surgical_history": "Past Surgical History",
    "social_history": "Social History",
    "family_history": "Family History",
    "physical_exam": "Physical Exam",
    "pertinent_results": "Pertinent Results",
    "brief_hospital_course": "Brief Hospital Course",
    "medications_on_admission": "Medications on Admission",
    "medications_on_transfer": "Medications on Transfer",
    "discharge_medications": "Discharge Medications",
    "discharge_disposition": "Discharge Disposition",
    "admission_diagnosis": "Admission Diagnosis",
    "discharge_diagnosis": "Discharge Diagnosis",
    "discharge_condition": "Discharge Condition",
    "discharge_instructions": "Discharge Instructions",
    "discharge_summary": "Discharge Summary",
    "followup_instructions": "Follow-up Instructions",
    "facility": "Facility",
    "complications": "Complications",
    "consults": "Consultations",
    "assessment_and_plan": "Assessment and Plan",
    "pathology": "Pathology",
    "infections": "Infections",
    "transfusions": "Blood Transfusions",
    "indications": "Indications for the Surgery",
    "laboratory_trends": "Laboratory Trends",
}

_TITLE_CASE_MINOR = frozenset({
    "of", "and", "the", "in", "on", "at", "to", "a", "an", "for", "with", "by",
})


def section_label(section: object) -> str:
    """Human heading for a chart section key."""
    name = str(section or "")
    return SECTION_LABELS.get(name, name.replace("_", " ") or "Note passage")


def title_case(text: object) -> str:
    """Phrase in Title Case, preserving words that already carry internal capitals."""
    words = " ".join(str(text or "").split()).split(" ")
    cased: list[str] = []
    for position, word in enumerate(words):
        if not word:
            continue
        if word.isupper() or any(character.isupper() for character in word[1:]):
            cased.append(word)
            continue
        lowered = word.lower()
        if 0 < position < len(words) - 1 and lowered in _TITLE_CASE_MINOR:
            cased.append(lowered)
            continue
        cased.append(word[:1].upper() + word[1:])
    return " ".join(cased)


def pill_section_label(section: object) -> str:
    """Title-cased inline pill label for a chart section key."""
    name = str(section or "")
    return PILL_SECTION_LABELS.get(name) or title_case(section_label(section))


def pill_journal_label(name: object, abbreviation: object = "") -> str:
    """Compact Title Case inline pill label for a PubMed record."""
    full = " ".join(str(name or "").split())
    short = " ".join(str(abbreviation or "").split())
    chosen = short or full
    return title_case(chosen) if chosen else "PubMed"


def cite_labels_from(
    evidence_state: Any = None,
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Map current citation indices (`'1'`, `'2'`) and PMIDs to inline pill labels."""
    out: dict[str, str] = {}
    if evidence_state is not None:
        items = getattr(evidence_state, "items", None)
        if isinstance(items, Mapping):
            for ref_id, item in items.items():
                sec = getattr(item, "section", None)
                if sec:
                    out[str(ref_id)] = pill_section_label(sec)
        literature = getattr(evidence_state, "literature", None)
        if isinstance(literature, Mapping):
            for pmid, item in literature.items():
                meta = getattr(item, "metadata", None) or {}
                out[str(pmid)] = pill_journal_label(
                    meta.get("journal"), meta.get("journal_abbrev"),
                )
    for call in tool_calls or ():
        if not isinstance(call, Mapping):
            continue
        response = call.get("response")
        if not isinstance(response, Mapping):
            continue
        for passage in response.get("passages") or ():
            if not isinstance(passage, Mapping):
                continue
            cite = passage.get("cite")
            sec = passage.get("section")
            if isinstance(cite, int) and not isinstance(cite, bool) and sec:
                out.setdefault(str(cite), pill_section_label(sec))
        for article in response.get("articles") or ():
            if not isinstance(article, Mapping):
                continue
            pmid = str(article.get("pmid") or "").strip()
            if pmid:
                out.setdefault(
                    pmid,
                    pill_journal_label(article.get("journal"), article.get("journal_abbrev")),
                )
    return out

