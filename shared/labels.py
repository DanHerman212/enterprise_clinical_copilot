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

#: The curated map. A key absent from it is title-cased by `humanize`, and reported
#: unchanged by `label_for` — two different fallbacks, both deliberate: display wants
#: something readable, and a guard that prints a key wants the key.
FEATURE_LABELS: dict[str, str] = {
    "age": "age",
    "gender": "sex",
    "has_procedure": "procedure performed",
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

