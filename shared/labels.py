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
