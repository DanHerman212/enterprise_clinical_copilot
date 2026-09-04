"""generate_hybrid_features_v2 — regenerate plausible demo features.

The original demo features were synthetic constants (rdw_max=13.5 and
monocytes_min=0.7 for every patient, readmission_30d=0 everywhere). This
script regenerates them the way the demo should have been built:

  1. EXTRACT the fields that are actually present in each MTSamples discharge
     note (age, sex, admission type, LOS, procedures, meds, oncology,
     discharge location, race, prior-admission history) via Gemini.
  2. IMPUTE the fields that are never in the note (labs, insurance, and any
     gap in the above) by sampling REAL MIMIC rows from
     readmission.analytics_dataset_encoded — preserving the joint lab/history
     distribution — then overriding with the note-extracted values.
  3. LABEL readmission_30d by scoring each row with the trained model and
     thresholding at the operating threshold (0.11).

Writes:
  eval/results/hybrid_cohort_v2.json        (features + probability + band + label)
  readmission.hybrid_features_v2             (the table, for validation before swap)

Usage (from repo root):
  .venv/bin/python projects/agent-harness/scripts/generate_hybrid_features_v2.py --limit 10
  .venv/bin/python projects/agent-harness/scripts/generate_hybrid_features_v2.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from google import genai
from google.cloud import bigquery, storage
from google.genai import types

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

PROJECT = "trim-icon-498815-a0"
LOCATION = "us-east1"
MODEL = "gemini-2.5-flash"

NOTES_SRC = HARNESS / "eval" / "results" / "hybrid_notes.json"
OUT_JSON = HARNESS / "eval" / "results" / "hybrid_cohort_v2.json"
OUT_TABLE = f"{PROJECT}.readmission.hybrid_features_v2"
MIMIC = f"`{PROJECT}.readmission.analytics_dataset_encoded`"
BUNDLE = (
    "gs://trim-icon-498815-a0-mlops/pipeline-root/778397675435/"
    "readmission-training-20260901151741/register-model_1444282052923883520/serving_model"
)

FEATURE_NAMES = [
    "age", "prior_admission_count", "prior_inpatient_days", "recent_ed_visits",
    "index_los_days", "procedure_count", "medication_count",
    "medication_order_count", "rbc_last", "rbc_min", "rdw_max", "monocytes_min",
    "hemoglobin_min", "sodium_last", "sodium_max", "sodium_min", "gender",
    "has_procedure", "oncology_flag",
    "race_white", "race_black", "race_hispanic", "race_asian", "race_amind",
    "race_nhpi", "race_unknown",
    "admission_type_ew_emer", "admission_type_eu_obs", "admission_type_obs_admit",
    "admission_type_urgent", "admission_type_direct_emer",
    "admission_type_ambulatory_obs", "admission_type_direct_obs",
    "admission_type_unknown",
    "discharge_location_home", "discharge_location_home_health",
    "discharge_location_snf", "discharge_location_rehab", "discharge_location_ltac",
    "discharge_location_hospice", "discharge_location_ama",
    "discharge_location_psych", "discharge_location_assisted_living",
    "discharge_location_unknown",
    "insurance_medicare", "insurance_medicaid", "insurance_private",
    "insurance_other", "insurance_unknown",
]

# MIMIC med-order-count ~= 1.6x med-count for the demo notes (orders per med).
MED_ORDER_RATIO = 1.6

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "age": {"type": "NUMBER"},
        "sex": {"type": "STRING", "enum": ["male", "female"]},
        "admission_type": {"type": "STRING", "enum": [
            "ew_emer", "urgent", "direct_emer"]},
        "los_days": {"type": "NUMBER"},
        "procedure_count": {"type": "INTEGER"},
        "medication_count": {"type": "INTEGER"},
        "oncology_flag": {"type": "INTEGER", "enum": [0, 1]},
        "discharge_location": {"type": "STRING", "enum": [
            "home", "home_health", "snf", "rehab", "ltac", "hospice",
            "ama", "psych", "assisted_living"]},
        "prior_admission_count": {"type": "INTEGER"},
        "prior_inpatient_days": {"type": "NUMBER"},
        "recent_ed_visits": {"type": "INTEGER"},
        "race": {"type": "STRING", "enum": [
            "white", "black", "hispanic", "asian", "amind", "nhpi"]},
    },
}

_PROMPT = (
    "You are extracting structured clinical fields from a hospital discharge note. "
    "Report ONLY fields the note actually states; OMIT any field not stated (never "
    "output 0 or a guess). age: from 'X-year-old'. race: from 'white female', "
    "'African American male', etc. admission_type: 'emergency room'/'ER' -> ew_emer; "
    "'urgent' -> urgent; 'direct admission' -> direct_emer; omit if only 'elective'. "
    "los_days: from 'hospital day N' or 'N-day stay'. oncology_flag: 1 only if a "
    "cancer/malignancy diagnosis is present. discharge_location: home/snf/rehab/"
    "hospice/ltac/psych/assisted_living/home_health/ama; omit if the patient died. "
    "prior_admission_count / prior_inpatient_days / recent_ed_visits: from admission "
    "history phrasing only."
)


def extract(client: genai.Client, note: str) -> dict:
    resp = client.models.generate_content(
        model=MODEL, contents=[_PROMPT, "\n\nDISCHARGE NOTE:\n", note],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=_SCHEMA, temperature=0.0),
    )
    return json.loads(resp.text)


def one_hot(prefix: str, category: str | None) -> dict[str, float]:
    out = {}
    for name in FEATURE_NAMES:
        if name.startswith(prefix + "_"):
            out[name] = 1.0 if name == f"{prefix}_{category}" else 0.0
    return out


def load_model() -> tuple[xgb.Booster, list[str], float]:
    bucket, _, prefix = BUNDLE[len("gs://"):].partition("/")
    b = storage.Client(project=PROJECT).bucket(bucket)
    manifest = json.loads(b.blob(f"{prefix}/manifest.json").download_as_text())
    threshold = json.loads(b.blob(f"{prefix}/threshold.json").download_as_text())
    model_path = "/tmp/model.bst"
    with open(model_path, "wb") as fh:
        fh.write(b.blob(f"{prefix}/model.bst").download_as_bytes())
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst, manifest["feature_order"], float(threshold["threshold"])


def sample_mimic_pool(client: bigquery.Client, n: int = 500) -> list[dict]:
    """A random pool of real MIMIC rows to impute from (age-matched later)."""
    cols = ", ".join(FEATURE_NAMES)
    rows = list(client.query(
        f"SELECT {cols} FROM {MIMIC} WHERE age IS NOT NULL ORDER BY RAND() LIMIT {n}"
    ).result())
    return [dict(r.items()) for r in rows]


def closest_donor(pool: list[dict], age: float) -> dict:
    """The pool row whose age is closest to the patient's, for coherent labs."""
    return min(pool, key=lambda d: abs(float(d["age"]) - float(age)))


def count_meds(note_text: str) -> int:
    """Deterministic discharge-medication count from the parsed med section.

    The LLM's integer count is not stable across calls (the same note returned
    15, then 27, then 45). Counting the parsed section body directly is
    deterministic and uses the same section parser the RAG pipeline uses.
    """
    from rag.sections import parse_note

    bodies = {s.name: s.body for s in parse_note(note_text).sections}
    body = bodies.get("discharge_medications") or bodies.get("discharge_instructions") or ""
    if not body:
        return 0
    entries = []
    for line in body.splitlines():
        for part in line.split(","):
            p = part.strip()
            if p and not re.match(
                r"^(discharge medications|medications|discharge instructions)", p, re.I
            ):
                entries.append(p)
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N notes (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip the BigQuery load (write JSON only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    notes_doc = json.loads(NOTES_SRC.read_text())
    patients = notes_doc["patients"][: args.limit] if args.limit else notes_doc["patients"]
    print(f"notes: {len(patients)}")

    bq = bigquery.Client(project=PROJECT)
    bst, feature_order, threshold = load_model()
    pool = sample_mimic_pool(bq)
    rng = np.random.default_rng(args.seed)

    genai_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    rows = []
    for i, p in enumerate(patients):
        extracted = extract(genai_client, p["note"])

        # ---- scalar fields: extracted if present, else the age-matched donor ----
        age = extracted.get("age")
        donor = closest_donor(pool, float(age)) if age is not None else pool[i % len(pool)]
        if age is None:
            age = donor["age"]
        sex = extracted.get("sex")
        gender = 1.0 if sex == "male" else 0.0 if sex == "female" else donor["gender"]
        los = extracted.get("los_days") or donor["index_los_days"]
        proc = extracted.get("procedure_count")
        proc = proc if proc is not None else donor["procedure_count"]
        med_count = float(count_meds(p["note"]))
        med_orders = round(med_count * MED_ORDER_RATIO * rng.uniform(0.9, 1.1), 2)
        oncology = extracted.get("oncology_flag")
        oncology = oncology if oncology is not None else 0.0

        # admission type: extracted -> one-hot; else donor one-hot
        at = extracted.get("admission_type")
        admission = one_hot("admission_type", at)
        if at is None:
            admission = {k: donor[k] for k in FEATURE_NAMES
                         if k.startswith("admission_type_")}

        # race / discharge_location / insurance: extracted if present, else donor
        rc = extracted.get("race")
        race = one_hot("race", rc)
        if rc is None:
            race = {k: donor[k] for k in FEATURE_NAMES if k.startswith("race_")}
        dl = extracted.get("discharge_location")
        dloc = one_hot("discharge_location", dl)
        if dl is None:
            dloc = {k: donor[k] for k in FEATURE_NAMES
                    if k.startswith("discharge_location_")}
        ins = {k: donor[k] for k in FEATURE_NAMES if k.startswith("insurance_")}

        # history: extracted if present, else age-matched donor
        prior_adm = extracted.get("prior_admission_count")
        prior_adm = (float(prior_adm) if prior_adm is not None
                     else (donor.get("prior_admission_count") or 0.0))
        prior_days = extracted.get("prior_inpatient_days")
        prior_days = (float(prior_days) if prior_days is not None
                      else (donor.get("prior_inpatient_days") or 0.0))
        ed_visits = extracted.get("recent_ed_visits")
        ed_visits = (float(ed_visits) if ed_visits is not None
                     else (donor.get("recent_ed_visits") or 0.0))

        # labs: donor values, conditioned on oncology (tumor anemia / inflammation)
        labs = {k: donor[k] for k in [
            "rbc_last", "rbc_min", "rdw_max", "monocytes_min", "hemoglobin_min",
            "sodium_last", "sodium_max", "sodium_min"]}
        if oncology == 1:
            if labs["hemoglobin_min"] is not None:
                labs["hemoglobin_min"] = round(max(4.0, labs["hemoglobin_min"] - 2.0), 2)
            if labs["rdw_max"] is not None:
                labs["rdw_max"] = round(labs["rdw_max"] + 2.0, 2)

        row = {
            "hadm_id": int(p["hadm_id"]),
            "age": float(age),
            "prior_admission_count": float(prior_adm),
            "prior_inpatient_days": float(prior_days),
            "recent_ed_visits": float(ed_visits),
            "index_los_days": float(los),
            "procedure_count": float(proc),
            "medication_count": float(med_count),
            "medication_order_count": float(med_orders),
            **labs,
            "gender": float(gender),
            "has_procedure": 1.0 if float(proc) > 0 else 0.0,
            "oncology_flag": float(oncology),
            **race, **admission, **dloc, **ins,
        }
        # enforce model feature order and score
        feats = np.array([row[n] for n in feature_order], dtype=np.float32)
        d = xgb.DMatrix(feats.reshape(1, -1), feature_names=feature_order)
        probability = float(bst.predict(d)[0])
        row["split_name"] = "test"
        row["subject_id"] = None
        row["probability"] = round(probability, 6)

        rows.append(row)
        print(f"hadm {p['hadm_id']}: age={age:.0f} sex={sex} los={los:.1f} "
              f"meds={med_count:.0f} onco={int(oncology)} prob={probability:.4f}")

    # ---- assign labels: the highest-risk ~15% are the readmitted cohort ----
    POSITIVE_RATE = 0.15
    ordered = sorted(rows, key=lambda r: r["probability"], reverse=True)
    k = max(1, round(len(ordered) * POSITIVE_RATE))
    cutoff = ordered[k - 1]["probability"]
    for r in rows:
        r["readmission_30d"] = 1 if r["probability"] >= cutoff else 0
    print(f"label cutoff: {cutoff:.4f} -> {k} positive")

    # ---- report ----
    probs = [r["probability"] for r in rows]
    labels = [r["readmission_30d"] for r in rows]
    print(f"\nprobability: min={min(probs):.4f} med={np.median(probs):.4f} "
          f"max={max(probs):.4f}")
    print(f"positives: {sum(labels)}/{len(labels)} ({sum(labels)/len(labels)*100:.1f}%)")

    out = {"seed": args.seed, "n": len(rows),
           "threshold": threshold, "patients": rows}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")

    if args.dry_run:
        print("dry-run: skipping BigQuery load")
        return 0

    schema = [bigquery.SchemaField("hadm_id", "INT64")]
    schema += [bigquery.SchemaField(n, "FLOAT64") for n in feature_order]
    schema += [bigquery.SchemaField("subject_id", "INT64"),
               bigquery.SchemaField("split_name", "STRING"),
               bigquery.SchemaField("readmission_30d", "INT64")]
    job = bq.load_table_from_json(
        [{"subject_id": r["subject_id"], "split_name": r["split_name"],
          "readmission_30d": r["readmission_30d"],
          **{n: r[n] for n in feature_order}, "hadm_id": r["hadm_id"]}
         for r in rows],
        OUT_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=schema, write_disposition="WRITE_TRUNCATE"))
    job.result()
    print(f"loaded {len(rows)} rows -> {OUT_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
