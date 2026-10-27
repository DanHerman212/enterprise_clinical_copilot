"""The human-readable profile of one admission, decoded from its model row.

Why this exists
---------------
`predict_readmission` scores a tabular row, and the attributions it returns name
features such as `admission_type`. But the row is one-hot encoded, and nothing in
the retrieval tools — which search discharge-note text — ever states
"Admission type: URGENT". So the agent could report that admission type drives a
patient's risk and then be unable to say what the admission type actually was.
With the CLINICAL SCOPE grounding rules it did the only honest thing available
and answered "the discharge note does not document this", which is correct and
useless.

The fix is not a text-to-SQL tool. The encoding is static, the admission id is
known, and the levels are a closed set, so the row the model was scored on can be
read back as words here — once, deterministically, with no schema reasoning at
runtime.

What this is NOT
----------------
It does not infer, impute or "best guess" a level. A patient whose dummies are all
zero, or whose row carries a level this table does not name, is reported as not
recorded. An absent fact and an inferred one are the same thing to a reader and
must not be to this module.

Decoding rule
-------------
One group, one label: the single column holding 1 wins. Several 1s, or none, is
not a level and is reported as absent rather than resolved by column order.
"""

from __future__ import annotations

from typing import Any

#: Human wording for each encoded level. The MIMIC value is kept in brackets where
#: it is a term of art, because a clinician and an analyst should both recognise
#: it, and because renaming it ("Emergency") would lose the distinction between
#: an emergency admission and a direct emergency transfer.
LEVEL_LABELS: dict[str, dict[str, str]] = {
    "admission_type": {
        "admission_type_ew_emer": "Emergency (EW EMER.)",
        "admission_type_eu_obs": "Observation (EU OBSERVATION)",
        "admission_type_obs_admit": "Observation admit (OBSERVATION ADMIT)",
        "admission_type_urgent": "Urgent (URGENT)",
        "admission_type_direct_emer": "Emergency, direct admission (DIRECT EMER.)",
        "admission_type_ambulatory_obs": (
            "Ambulatory observation (AMBULATORY OBSERVATION)"
        ),
        "admission_type_direct_obs": "Direct observation (DIRECT OBSERVATION)",
    },
    "discharge_location": {
        "discharge_location_home": "Home",
        "discharge_location_home_health": "Home with home health",
        "discharge_location_snf": "Skilled nursing facility",
        "discharge_location_rehab": "Inpatient rehabilitation",
        "discharge_location_ltac": "Long-term acute care",
        "discharge_location_hospice": "Hospice",
        "discharge_location_ama": "Left against medical advice",
        "discharge_location_psych": "Psychiatric facility",
        "discharge_location_assisted_living": "Assisted living",
    },
    "insurance": {
        "insurance_medicare": "Medicare",
        "insurance_medicaid": "Medicaid",
        "insurance_private": "Private",
        "insurance_other": "Other",
    },
    "race": {
        "race_white": "White",
        "race_black": "Black",
        "race_asian": "Asian",
        "race_hispanic": "Hispanic",
        "race_amind": "American Indian or Alaska Native",
        "race_nhpi": "Native Hawaiian or Pacific Islander",
    },
}

#: Counts that are already numeric and need no decoding. Named here rather than
#: read generically so the manifest cannot quietly grow the whole feature vector.
COUNT_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "Age (years)"),
    ("prior_admission_count", "Prior admissions"),
    ("prior_inpatient_days", "Prior inpatient days"),
    ("recent_ed_visits", "ED visits in the prior year"),
    ("index_los_days", "Length of stay (days)"),
)

#: What the manifest states, in the order a reader wants it.
FIELD_ORDER: tuple[str, ...] = (
    "encounter_type",
    "age", "gender", "admission_type", "discharge_location", "insurance", "race",
    "prior_admission_count", "prior_inpatient_days", "recent_ed_visits",
    "index_los_days",
)

#: `encounter_type` is a plain column rather than a one-hot group, and it leads the
#: manifest because everything after it is conditional on it: an outpatient visit has no
#: admission type, no discharge disposition and no length of stay, so a reader who meets
#: those fields first has been told four things about a stay that did not happen.
#:
#: It is a property of the record, not of the model. The encoder does not consume it —
#: `feature_order()` excludes it, so it never reaches the booster — and it is carried on
#: the row because the row is the only place the record's own facts live.
ENCOUNTER_FIELD = "encounter_type"
ENCOUNTER_LABELS: dict[str, str] = {
    "inpatient": "inpatient admission",
    "outpatient": "outpatient encounter (no inpatient stay)",
}

ABSENT = "Not recorded"


def _is_level(value: Any) -> bool:
    """True only for a genuine one-hot hit. None and 0 are both "no"."""
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def _decode_group(row: dict[str, Any], columns: dict[str, str]) -> str:
    hits = [label for column, label in columns.items() if _is_level(row.get(column))]
    return hits[0] if len(hits) == 1 else ABSENT


def _decode_count(value: Any) -> str:
    if value is None:
        return ABSENT
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ABSENT
    # A count is an integer in every case here; `age` is the one that can be
    # fractional in the source and is reported the same way.
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _decode_encounter(value: Any) -> str:
    """The encounter type as the row states it, or `ABSENT`.

    Read verbatim and normalised to lower case: this column is not a level of an
    encoding, so there is nothing to resolve. A value the vocabulary does not name is
    reported as not recorded rather than guessed at, for the same reason the one-hot
    groups are — an absent fact and an inferred one are the same thing to a reader.
    """
    text = str(value or "").strip().lower()
    return ENCOUNTER_LABELS.get(text, ABSENT)


def decode_profile(row: dict[str, Any]) -> dict[str, str]:
    """The readable profile of one feature row, keyed by `FIELD_ORDER`."""
    if not row:
        return {}

    profile: dict[str, str] = {
        ENCOUNTER_FIELD: _decode_encounter(row.get(ENCOUNTER_FIELD)),
    }
    for field, label in COUNT_FIELDS:
        profile[field] = _decode_count(row.get(field))

    # `gender` is `CAST(gender = 'M' AS INT64)` in the mart, so 1 is male and 0 is
    # female. Anything else — including NULL — is not a sex we can state.
    gender = row.get("gender")
    if _is_level(gender):
        profile["gender"] = "Male"
    elif gender is not None and str(gender).strip() not in ("", "None"):
        try:
            profile["gender"] = "Female" if float(gender) == 0.0 else ABSENT
        except (TypeError, ValueError):
            profile["gender"] = ABSENT
    else:
        profile["gender"] = ABSENT

    for group, columns in LEVEL_LABELS.items():
        profile[group] = _decode_group(row, columns)

    return {field: profile.get(field, ABSENT) for field in FIELD_ORDER}
