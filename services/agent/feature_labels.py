"""Feature label map — the single source of truth for how model features are
named in the UI.

The risk canvas's FactorBars and the agent's prose must name a feature the
same way — a reviewer reading "medication order count" on the bars but
`medication_order_count` in the prose sees a bug. Every feature that can
appear as a top factor gets a human label here; the canvas attaches it to
each factor and the agent prompt embeds the same map so the live agent's
words match the UI.

This module lives with the agent (not the site) because the canvas is
composed here — presentation of model output belongs to the service that
owns the model contract.

Keys are the aggregated PARENT features returned by predict_readmission
(one-hot groups summed to their parent — see mlops/src/encoding.py groups()).
"""

import re

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
    2. Otherwise normalize the raw key to Title Case, handling both
       camelCase and snake_case: medicationOrderCount and
       medication_order_count both become "Medication Order Count". This is
       the safety net for any feature not in the curated map.
    """
    known = FEATURE_LABELS.get(feature)
    if known is not None:
        return known
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", feature)
    words = [w for w in re.split(r"[\s_\-]+", spaced) if w]
    return " ".join(w.capitalize() for w in words) if words else feature
