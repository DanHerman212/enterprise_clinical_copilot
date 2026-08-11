"""build_fixtures.py — capture REAL demo payloads for offline UI development.

The serving endpoints are torn down (cost), but the model bundle is on GCS
(storage = pennies) and BigQuery is free. This script therefore produces
GENUINE payloads the UI can render against while the endpoints are down:

  predict  — runs the real serving predictor (ReadmissionPredictor) locally,
             so probability, threshold, base_value and the native-TreeSHAP
             top_factors are the same numbers the deployed endpoint returns.
  rag      — the real passages captured in the 2026-08-11 live integration
             test (ids, sections, scores), with the full note text pulled from
             BigQuery (the tool returns full note text by note_id).

Provenance is recorded per fixture in data/demo_fixtures/README.md, so the UI
is never built against data we cannot defend.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/build_fixtures.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from google.cloud import aiplatform, bigquery

sys.path.insert(0, ".")

from mcp_server.config import DISCHARGE_TABLE, LOCATION, PROJECT  # noqa: E402

# The real passages from the 2026-08-11 live integration test. ids/sections/
# scores are exactly what the deployed index returned.
CAPTURED_RAG = {
    "sepsis_and_lactate": {
        "query": "sepsis and elevated lactate on broad-spectrum antibiotics",
        "hadm_id": 20724182,
        "note_id": "13479418-DS-24",
        "returned": 4,
        "passages": [
            ("13479418-DS-24_brief_hospital_course_2", "brief_hospital_course", 0.2186),
            ("13479418-DS-24_history_of_present_illness_1", "history_of_present_illness", 0.2069),
            ("13479418-DS-24_discharge_instructions_1", "discharge_instructions", 0.2032),
            ("13479418-DS-24_discharge_diagnosis_1", "discharge_diagnosis", 0.2031),
        ],
    },
    "medications": {
        "query": "medications",
        "hadm_id": 20724182,
        "note_id": "13479418-DS-24",
        "returned": 1,
        "passages": [
            ("13479418-DS-24_discharge_medications_1", "discharge_medications", 0.2479),
        ],
        "score_note": ("score not captured in the live smoke test (text was "
                        "printed, score was not); id/section/text are real, "
                        "score is representative"),
    },
}

# Demo patients to build predict fixtures for. 20724182 is the patient whose
# rag passages we captured; 20924467 is the known borderline case.
PREDICT_PATIENTS = [20724182, 20924467, 22489815, 25828809]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_fixtures"


# --------------------------------------------------------------------------- #
# Prediction fixtures — real, via the local serving predictor.
# --------------------------------------------------------------------------- #

def _bundle() -> tuple[str, str]:
    """(bundle_uri, model_version) of the newest readmission-final-* model."""
    aiplatform.init(project=PROJECT, location=LOCATION)
    models = [
        m for m in aiplatform.Model.list(order_by="create_time desc")
        if m.display_name.startswith("readmission-final-")
    ]
    if not models:
        raise SystemExit("No 'readmission-final-*' model found.")
    return models[0].gca_resource.artifact_uri.rstrip("/"), models[0].display_name


def _patient_features(bq: bigquery.Client, hadm_id: int, feature_order: list[str]):
    """Row of the encoded feature table in feature order (None for missing)."""
    rows = list(bq.query(
        f"SELECT * FROM readmission.analytics_dataset_encoded "
        f"WHERE hadm_id = {hadm_id} LIMIT 1"
    ).result())
    if not rows:
        raise SystemExit(f"No feature row for hadm_id={hadm_id}")
    row = dict(rows[0].items())
    return [row.get(c) if row.get(c) is not None else None for c in feature_order]


def _run_predictions(bq, hadm_ids: list[int], version: str) -> dict[int, dict]:
    """Run the REAL serving predictor for many patients with one model load.

    The bundle is downloaded once into a scratch dir; the booster stays in
    memory, so every patient is scored against the same model as the deployed
    endpoint. Returns {hadm_id: payload}.
    """
    from google.cloud.aiplatform.prediction.predictor import Predictor  # noqa
    from google.cloud.aiplatform.utils import prediction_utils  # noqa
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "mlops" / "pipelines" / "serving" / "cpr"))
    from predictor import ReadmissionPredictor  # noqa: E402

    uri = _bundle()[0]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            p = ReadmissionPredictor()
            p.load(uri)
            order = p._feature_order
            for hadm_id in hadm_ids:
                features = _patient_features(bq, hadm_id, order)
                matrix = p.preprocess({"instances": [features]})
                probs, contribs = p.predict(matrix)
                out = p.postprocess((probs, contribs))["predictions"][0]
                results[hadm_id] = _shape(out, hadm_id, version)
        finally:
            os.chdir(cwd)
    return results


def _shape(out: dict, hadm_id: int, version: str) -> dict:
    """Shape one prediction like the predict_readmission tool response."""
    top_factors = [
        {
            "feature": f["feature"],
            "contribution": round(float(f["attribution"]), 4),
            "direction": "increases" if float(f["attribution"]) > 0 else "decreases",
        }
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
        "feature_source": "BigQuery",
        "provenance": "REAL — local run of the serving predictor (2026-08-11)",
    }


# --------------------------------------------------------------------------- #
# RAG fixtures — real captured passages + full note text from BigQuery.
# --------------------------------------------------------------------------- #

def _note_text(bq: bigquery.Client, note_id: str) -> str:
    rows = bq.query(
        f"SELECT text FROM `{DISCHARGE_TABLE}` WHERE note_id = '{note_id}' LIMIT 1"
    ).result()
    rows = list(rows)
    if not rows:
        raise SystemExit(f"No note text for {note_id}")
    return rows[0]["text"]


def _rag_payload(bq: bigquery.Client, spec: dict) -> dict:
    text = _note_text(bq, spec["note_id"])
    passages = [
        {
            "id": pid,
            "section": section,
            "text": text,
            "score": score,
        }
        for pid, section, score in spec["passages"]
    ]
    payload = {
        "hadm_id": spec["hadm_id"],
        "query": spec["query"],
        "returned": len(passages),
        "passages": passages,
        "provenance": "REAL — passages from 2026-08-11 live integration test; "
                      "text from BigQuery",
    }
    if spec.get("score_note"):
        payload["provenance"] += f" | {spec['score_note']}"
    return payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bq = bigquery.Client(project=PROJECT)

    # All demo-cohort patients (for the list dots), plus the named fixture
    # patients (all of which are in the cohort).
    cohort_path = OUT_DIR.parent / "demo_cohort.json"
    cohort = json.loads(cohort_path.read_text())["patients"]
    all_hadm = sorted(int(p["hadm_id"]) for p in cohort)
    named = sorted(set(PREDICT_PATIENTS) | set(all_hadm))

    version = _bundle()[1]
    results = _run_predictions(bq, named, version)

    written = []
    # Per-patient predict files for the named fixture patients.
    for hadm_id in PREDICT_PATIENTS:
        payload = results[hadm_id]
        path = OUT_DIR / f"predict_{hadm_id}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path.name)
        print(f"predict {hadm_id}: probability={payload['probability']} "
              f"decision={payload['decision']} -> {path.name}")

    # Cohort-wide risk cache (real numbers for every demo patient).
    cohort_risk = {str(h): results[h] for h in all_hadm}
    path = OUT_DIR / "cohort_risk.json"
    path.write_text(json.dumps(cohort_risk, indent=2))
    written.append(path.name)
    print(f"cohort_risk: {len(cohort_risk)} patients -> {path.name}")

    for name, spec in CAPTURED_RAG.items():
        payload = _rag_payload(bq, spec)
        path = OUT_DIR / f"rag_{name}_{spec['hadm_id']}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path.name)
        print(f"rag {name}: returned={payload['returned']} -> {path.name}")

    readme = OUT_DIR / "README.md"
    readme.write_text(_readme())
    print(f"wrote {len(written) + 1} files to {OUT_DIR}")
    return 0


def _readme() -> str:
    return """# Demo Fixtures — REAL payloads for offline UI development

Built by `scripts/build_fixtures.py` (2026-08-11) while the serving endpoints
are torn down for cost. Every fixture is genuine — provenance is recorded in
each file's `provenance` field.

## Predict (`predict_<hadm>.json`)
- Computed by running the REAL serving predictor (`ReadmissionPredictor`) locally
  against the serving bundle on GCS + features from BigQuery.
- probability / threshold / decision / base_value / top_factors (native-TreeSHAP)
  are the same numbers the deployed endpoint returns.

## RAG (`rag_*.json`)
- `rag_sepsis_and_lactate_20724182.json` — passages returned by the live index
  on 2026-08-11 (query "sepsis and elevated lactate on broad-spectrum
  antibiotics"). ids / sections / scores are exactly what the deployed index
  returned; `text` is the full discharge note from BigQuery.
- `rag_medications_20724182.json` — same provenance, but the score was not
  captured in the live smoke test (text was printed, score was not), so the
  score is a representative value and is flagged in `provenance`.

## Regenerating
- Predict: rerun `build_fixtures.py` any time (needs only GCS + BigQuery).
- RAG: requires the index endpoint live (`deploy_index.py --mode deploy`).
"""


if __name__ == "__main__":
    raise SystemExit(main())
