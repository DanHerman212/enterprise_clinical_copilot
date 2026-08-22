"""build_hybrid_fixtures — capture HYBRID demo payloads for offline UI dev.

Hybrid twin of build_synthetic_fixtures.py + capture_synthetic_rag_fixtures.py.
The demo is real-system-on-hybrid-data: real MTSamples notes + parsed/filled
features + the real served model. These fixtures let the UI render offline
with EXACTLY the payloads the deployed endpoints return for the hybrid cohort.

  * predict fixtures  — run the REAL serving predictor (ReadmissionPredictor,
    the bundle on GCS) on hybrid_cohort.json feature rows.
  * rag fixtures      — run the REAL rag_search against the hybrid index for
    the chip queries on a primary hybrid patient.
  * cohort_risk.json  — predict payload for every hybrid patient (the site's
    risk_for() source).
  * Also writes demo_cohort.json (site seed: deterministic names/summaries)
    via seed_synthetic_demo_cohort.py logic pointed at the hybrid cohort.

Provenance is marked HYBRID — de-identified transcription samples (MTSamples)
with parsed/filled features. Never "synthetic".

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/build_hybrid_fixtures.py
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
COHORT_SOURCE = HARNESS_ROOT / "eval" / "results" / "hybrid_cohort.json"
OUT_DIR = HARNESS_ROOT / "data" / "demo_fixtures"
SITE_FIXTURES = Path(os.environ.get(
    "SITE_FIXTURES",
    HARNESS_ROOT.parents[1].parent / "danielmherman" / "demo" / "data" / "demo_fixtures",
))
SITE_DATA = Path(os.environ.get(
    "SITE_DATA",
    HARNESS_ROOT.parents[1].parent / "danielmherman" / "demo" / "data",
))
PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"

# Hybrid patients to emit per-patient predict fixtures for: one per band with
# full chip support (low=90000001, borderline=90000009, high=90000023). The
# high patient moved from 90000017 to 90000023 when the cohort was re-scored
# with corrected gender (90000017 is now borderline; 90000023 is the top high
# at 0.3262).
PREDICT_PATIENTS = [90000001, 90000009, 90000023]

# The primary patient for rag fixtures (borderline — all chips return passages).
PRIMARY_HADM = int(os.environ.get("PRIMARY_HADM", "90000009"))

# The exact queries the site's fixture chips issue (fixtures.py _CHIP_QUERY).
CHIP_QUERIES = {
    "risk": "sepsis and elevated lactate on broad-spectrum antibiotics",
    "meds": "medications",
    "summarize": "summarize the hospital course and discharge diagnosis",
}

# --- name/summary assignment (same deterministic logic as the seed) ---------
FIRST_NAMES = {
    ("F", "older"): ["Margaret", "Dorothy", "Eleanor", "Ruth", "Joan", "Barbara"],
    ("F", "middle"): ["Linda", "Susan", "Deborah", "Karen", "Patricia", "Cynthia"],
    ("F", "younger"): ["Ashley", "Jasmine", "Megan", "Brittany", "Alicia", "Chelsea"],
    ("M", "older"): ["Harold", "Walter", "Eugene", "Raymond", "Clarence", "Herbert"],
    ("M", "middle"): ["Gregory", "Dennis", "Randall", "Curtis", "Bruce", "Alan"],
    ("M", "younger"): ["Tyler", "Devin", "Brandon", "Jordan", "Marcus", "Trevor"],
}
SURNAMES = [
    "Ellison", "Whitfield", "Barnhart", "Castellano", "Okafor", "Delgado",
    "Lindqvist", "Ferraro", "Hollingsworth", "Nakamura", "Boyle", "Mensah",
    "Vasquez", "Kowalski", "Abernathy", "Rasmussen", "Guerrero", "Tran",
    "Bellamy", "Sokolov", "Ibrahim", "Prentice", "Marchetti", "Duval",
    "Yeager", "Ashford", "Cardoso", "Nyland", "Petrov", "Ramachandran",
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
    return "older" if age >= 68 else ("middle" if age >= 45 else "younger")


def _one_hot_label(row: dict, mapping: dict, default: str) -> str:
    for col, label in mapping.items():
        if row.get(col) == 1:
            return label
    return default


def _summary(features: dict) -> str:
    sex = "M" if features.get("gender") == 1 else "F"
    parts = [f"{int(round(features['age']))}{sex}",
             _one_hot_label(features, ADMISSION_TYPES, "elective") + " admission"]
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
    return " · ".join(parts)


def _assign_name(features: dict, hadm_id: int, used: set) -> str:
    sex = "M" if features.get("gender") == 1 else "F"
    firsts = FIRST_NAMES[(sex, _era(features["age"]))]
    digest = hashlib.sha256(f"name:{hadm_id}".encode()).digest()
    first_i = int.from_bytes(digest[:4], "big")
    last_i = int.from_bytes(digest[4:8], "big")
    for bump in range(len(firsts) * len(SURNAMES)):
        name = (f"{firsts[(first_i + bump) % len(firsts)]} "
                f"{SURNAMES[(last_i + bump // len(firsts)) % len(SURNAMES)]}")
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("Name pool exhausted.")


# --- serving predictor (same as synthetic twin) ------------------------------

def _bundle() -> tuple[str, str]:
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT, location=LOCATION)
    models = [m for m in aiplatform.Model.list(order_by="create_time desc")
              if m.display_name.startswith("readmission-final-")]
    if not models:
        raise SystemExit("No 'readmission-final-*' model found.")
    return models[0].gca_resource.artifact_uri.rstrip("/"), models[0].display_name


def _run_predictions(feature_rows: dict[int, dict], version: str) -> dict[int, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "mlops" / "pipelines" / "serving" / "cpr"))
    from predictor import ReadmissionPredictor  # noqa: E402
    uri, _ = _bundle()
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            p = ReadmissionPredictor()
            p.load(uri)
            for hadm_id, features in feature_rows.items():
                matrix = p.preprocess({"instances": [features]})
                probs, contribs = p.predict(matrix)
                out = p.postprocess((probs, contribs))["predictions"][0]
                results[hadm_id] = _shape(out, hadm_id, version)
        finally:
            os.chdir(cwd)
    return results


def _shape(out: dict, hadm_id: int, version: str) -> dict:
    top_factors = [
        {"feature": f["feature"], "contribution": round(float(f["attribution"]), 4),
         "direction": "increases" if float(f["attribution"]) > 0 else "decreases"}
        for f in out["top_factors"][:5]
    ]
    return {
        "hadm_id": hadm_id,
        "probability": round(float(out["probability"]), 6),
        "threshold": float(out["threshold"]),
        "decision": int(out["prediction"]),
        "base_value": round(float(out["base_value"]), 6),
        "top_factors": top_factors,
        "model_version": version,
        "feature_source": "hybrid",
        "provenance": "HYBRID — local run of the serving predictor on hybrid "
                      "cohort features (de-identified transcription samples, "
                      "parsed/filled; 2026-08-21)",
    }


def _seed_demo_cohort(cohort: dict) -> None:
    """Write demo_cohort.json (site seed) from hybrid features."""
    used: set[str] = set()
    patients = []
    for p in cohort["patients"]:
        f = p["features"]
        patients.append({
            "hadm_id": int(p["hadm_id"]),
            "display_name": _assign_name(f, int(p["hadm_id"]), used),
            "age": int(round(f["age"])),
            "sex": "M" if f.get("gender") == 1 else "F",
            "summary": _summary(f),
            "split_name": "test",
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"patients": patients}, indent=2) + "\n"
    (OUT_DIR / "demo_cohort.json").write_text(payload)
    (SITE_DATA / "demo_cohort.json").write_text(payload)
    print(f"demo_cohort: {len(patients)} patients -> site demo/data/demo_cohort.json")


def main() -> int:
    if not COHORT_SOURCE.exists():
        raise SystemExit(f"hybrid cohort not found: {COHORT_SOURCE} — run "
                         f"build_hybrid_cohort.py first")
    cohort = json.loads(COHORT_SOURCE.read_text())
    patients_in = cohort["patients"]
    feature_rows = {int(p["hadm_id"]): p["features"] for p in patients_in}
    print(f"source: {COHORT_SOURCE} ({len(patients_in)} patients)")

    version = _bundle()[1]
    results = _run_predictions(feature_rows, version)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for hadm_id in PREDICT_PATIENTS:
        payload = results[hadm_id]
        path = OUT_DIR / f"predict_{hadm_id}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path.name)
        print(f"predict {hadm_id}: probability={payload['probability']} "
              f"decision={payload['decision']} -> {path.name}")

    cohort_risk = {str(h): results[h] for h in sorted(feature_rows)}
    path = OUT_DIR / "cohort_risk.json"
    path.write_text(json.dumps(cohort_risk, indent=2))
    written.append(path.name)
    print(f"cohort_risk: {len(cohort_risk)} hybrid patients -> {path.name}")

    _seed_demo_cohort(cohort)

    # Copy predict fixtures to the site's fixture dir.
    if SITE_FIXTURES:
        SITE_FIXTURES.mkdir(parents=True, exist_ok=True)
        for name in written:
            (SITE_FIXTURES / name).write_text((OUT_DIR / name).read_text())
        print(f"copied {len(written)} fixtures -> {SITE_FIXTURES}")

    return 0


async def _capture_rag() -> None:
    """Capture live rag fixtures for the primary hybrid patient."""
    os.environ["DISCHARGE_TABLE"] = f"{PROJECT}.readmission.hybrid_notes"
    sys.path.insert(0, str(HARNESS_ROOT))
    from mcp_server.tools.rag_search import rag_search  # noqa: E402

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for key, query in CHIP_QUERIES.items():
        res = await rag_search(PRIMARY_HADM, query, top_k=5)
        if res.get("error"):
            print(f"ERROR {key}: {res['error']}")
            raise SystemExit(1)
        payload = {
            **res,
            "provenance": f"HYBRID — live rag_search on the hybrid index "
                          f"({PRIMARY_HADM}, de-identified transcription "
                          f"samples; 2026-08-21)",
        }
        name = f"rag_{key}_{PRIMARY_HADM}.json"
        (OUT_DIR / name).write_text(json.dumps(payload, indent=2))
        written.append(name)
        print(f"{key}: returned={payload['returned']} -> {name}")

    if SITE_FIXTURES:
        SITE_FIXTURES.mkdir(parents=True, exist_ok=True)
        for name in written:
            (SITE_FIXTURES / name).write_text((OUT_DIR / name).read_text())
        print(f"copied {len(written)} rag fixtures -> {SITE_FIXTURES}")


if __name__ == "__main__":
    rc = main()
    if rc != 0:
        raise SystemExit(rc)
    asyncio.run(_capture_rag())
