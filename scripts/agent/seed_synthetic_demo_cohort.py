"""seed_synthetic_demo_cohort — emit the synthetic demo cohort seed artifact.

Replaces seed_demo_cohort.py for the synthetic cohort: instead of scoring a
MIMIC test-split pool against the live endpoint, this reads the already-scored
synthetic cohort (eval/results/synthetic_cohort.json) and renders the same
demo_cohort.json contract the Django loader expects.

Only the *data* is synthetic; the name-assignment and summary-building logic
is the same deterministic, era-matched code as the real seed path, so the
cohort quality bar is unchanged.

    python scripts/seed_synthetic_demo_cohort.py
      → writes data/demo_cohort.json (harness) + copies to the site's
        demo/data/demo_cohort.json

Env:
    SITE_DATA   site demo/data dir to also write (default:
                ../../danielmherman/demo/data — resolves next to the repo)
    COHORT_SOURCE  scored cohort JSON to render (default: hybrid_cohort.json —
                the full hybrid-108 demo cohort; previously synthetic_cohort.json)
"""

import hashlib
import json
import os
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
COHORT_SOURCE = Path(os.environ.get(
    "COHORT_SOURCE",
    HARNESS_ROOT / "eval" / "results" / "hybrid_cohort.json",
))
OUTPUT_PATH = HARNESS_ROOT / "data" / "demo_cohort.json"
# The site lives as a sibling workspace folder: Desktop/danielmherman.
SITE_DATA = Path(os.environ.get(
    "SITE_DATA",
    HARNESS_ROOT.parents[1].parent / "danielmherman" / "demo" / "data",
))

FIRST_NAMES = {
    ("F", "older"): ["Margaret", "Dorothy", "Eleanor", "Ruth", "Joan", "Barbara",
                     "Shirley", "Doris", "Betty", "Marjorie", "Gloria", "Rosemary"],
    ("F", "middle"): ["Linda", "Susan", "Deborah", "Karen", "Patricia", "Cynthia",
                      "Sandra", "Denise", "Yvonne", "Paula", "Theresa", "Loretta"],
    ("F", "younger"): ["Ashley", "Jasmine", "Megan", "Brittany", "Alicia", "Chelsea",
                       "Danielle", "Kayla", "Erica", "Monique", "Sierra", "Vanessa"],
    ("M", "older"): ["Harold", "Walter", "Eugene", "Raymond", "Clarence", "Herbert",
                     "Stanley", "Leonard", "Arthur", "Melvin", "Chester", "Vernon"],
    ("M", "middle"): ["Gregory", "Dennis", "Randall", "Curtis", "Bruce", "Alan",
                      "Terrence", "Douglas", "Roger", "Neil", "Lamar", "Vincent"],
    ("M", "younger"): ["Tyler", "Devin", "Brandon", "Jordan", "Marcus", "Trevor",
                       "Dustin", "Corey", "Andre", "Shane", "Jared", "Malik"],
}

SURNAMES = [
    "Ellison", "Whitfield", "Barnhart", "Castellano", "Okafor", "Delgado",
    "Lindqvist", "Ferraro", "Hollingsworth", "Nakamura", "Boyle", "Mensah",
    "Vasquez", "Kowalski", "Abernathy", "Rasmussen", "Guerrero", "Tran",
    "Bellamy", "Sokolov", "Ibrahim", "Prentice", "Marchetti", "Duval",
    "Yeager", "Ashford", "Cardoso", "Nyland", "Petrov", "Ramachandran",
    "Fontaine", "Stroud", "Ocampo", "Berkowitz", "Lindgren", "Achebe",
]

ADMISSION_TYPES = {
    "admission_type_ew_emer": "emergency",
    "admission_type_eu_obs": "emergency observation",
    "admission_type_obs_admit": "observation",
    "admission_type_urgent": "urgent",
    "admission_type_direct_emer": "direct emergency",
    "admission_type_ambulatory_obs": "ambulatory observation",
    "admission_type_direct_obs": "direct observation",
}

DISCHARGE_LOCATIONS = {
    "discharge_location_home": "home",
    "discharge_location_home_health": "home with services",
    "discharge_location_snf": "skilled nursing",
    "discharge_location_rehab": "rehab",
    "discharge_location_ltac": "long-term care",
    "discharge_location_hospice": "hospice",
    "discharge_location_psych": "psychiatric care",
    "discharge_location_ama": "against medical advice",
    "discharge_location_assisted_living": "assisted living",
}


def _era(age: float) -> str:
    if age >= 68:
        return "older"
    return "middle" if age >= 45 else "younger"


def _one_hot_label(row: dict, mapping: dict[str, str], default: str) -> str:
    for column, label in mapping.items():
        if row.get(column) == 1:
            return label
    return default


def _summary(features: dict) -> str:
    """Clinical descriptor from the synthetic features, never outcome-based."""
    sex = "M" if features.get("gender") == 1 else "F"
    parts = [
        f"{int(round(features['age']))}{sex}",
        _one_hot_label(features, ADMISSION_TYPES, "elective") + " admission",
    ]
    priors = int(round(features.get("prior_admission_count") or 0))
    if priors:
        parts.append(f"{priors} prior admission{'s' if priors != 1 else ''}")
    if features.get("oncology_flag") == 1:
        parts.append("oncology history")
    if features.get("has_procedure") == 1:
        parts.append(f"{int(round(features['procedure_count']))} procedure"
                     f"{'s' if int(round(features['procedure_count'])) != 1 else ''}")
    parts.append(f"{int(round(features['index_los_days']))}-day stay")
    discharge = _one_hot_label(features, DISCHARGE_LOCATIONS, "")
    if discharge:
        parts.append(f"discharged to {discharge}")
    return " \u00b7 ".join(parts)


def _assign_name(features: dict, hadm_id: int, used: set[str]) -> str:
    sex = "M" if features.get("gender") == 1 else "F"
    firsts = FIRST_NAMES[(sex, _era(features["age"]))]
    digest = hashlib.sha256(f"name:{hadm_id}".encode()).digest()
    first_i = int.from_bytes(digest[:4], "big")
    last_i = int.from_bytes(digest[4:8], "big")
    for bump in range(len(firsts) * len(SURNAMES)):
        name = (
            f"{firsts[(first_i + bump) % len(firsts)]} "
            f"{SURNAMES[(last_i + bump // len(firsts)) % len(SURNAMES)]}"
        )
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("Name pool exhausted; add more surnames.")


def main() -> int:
    if not COHORT_SOURCE.exists():
        raise SystemExit(f"synthetic cohort not found: {COHORT_SOURCE}")

    data = json.loads(COHORT_SOURCE.read_text())
    patients_in = data["patients"]
    print(f"source: {COHORT_SOURCE} ({len(patients_in)} patients)")

    used: set[str] = set()
    patients = []
    for p in patients_in:
        features = p["features"]
        patients.append(
            {
                "hadm_id": int(p["hadm_id"]),
                "display_name": _assign_name(features, int(p["hadm_id"]), used),
                "age": int(round(features["age"])),
                "sex": "M" if features.get("gender") == 1 else "F",
                "summary": _summary(features),
                "split_name": "test",
            }
        )

    print(f"\n{'hadm_id':>10} {'band':<10} {'prob':>7} {'name':<26} {'summary'}")
    by_id = {int(p["hadm_id"]): p for p in patients_in}
    for patient in patients:
        src = by_id[patient["hadm_id"]]
        print(f"{patient['hadm_id']:>10} {src['band']:<10} "
              f"{src['probability']:>7.4f} {patient['display_name']:<26} "
              f"{patient['summary']}")

    bands = {}
    for p in patients_in:
        bands.setdefault(p["band"], 0)
        bands[p["band"]] += 1
    print(f"\nbands: {bands} | total {len(patients)}")
    print(f"sex: M={sum(1 for p in patients if p['sex'] == 'M')} "
          f"F={sum(1 for p in patients if p['sex'] == 'F')} | "
          f"age {min(p['age'] for p in patients)}-{max(p['age'] for p in patients)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"patients": patients}, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH.relative_to(HARNESS_ROOT.parents[1])}")

    if SITE_DATA:
        SITE_DATA.mkdir(parents=True, exist_ok=True)
        site_path = SITE_DATA / "demo_cohort.json"
        site_path.write_text(json.dumps({"patients": patients}, indent=2) + "\n")
        print(f"copied → {site_path}")

    print("\nNext: reseed the site DB — python manage.py seed_demo_patients --prune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
