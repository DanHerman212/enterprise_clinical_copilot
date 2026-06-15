#!/usr/bin/env python3
"""
compute_hospital_baseline.py — HOSPITAL score AUCPR baseline.

Queries the HOSPITAL score table from BigQuery, computes AUCPR on the
training split, and logs the result to Vertex AI Experiments. This
baseline is the floor the ML model must beat — subsequent pipeline runs
( feature selection, training ) log to the same experiment for comparison.

Usage:
    python projects/mlops/scripts/compute_hospital_baseline.py

Logs to Vertex AI Experiment: readmission-mlops
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

# Ensure projects/mlops is on sys.path for src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bigquery import run_query
from src.config import PROJECT_ID, FULL_TABLE_REF, LABEL_COLUMN

# ---------------------------------------------------------------------------
# Experiment identity — reused by all downstream pipeline runs
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "readmission-mlops"
HOSPITAL_TABLE = f"{PROJECT_ID}.readmission.hospital_score"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_aucpr(y_true: pd.Series, y_score: pd.Series) -> float:
    """Compute area under the precision-recall curve."""
    return float(average_precision_score(y_true, y_score))


def log_to_vertex_experiments(
    hospital_aucpr: float,
    n_patients: int,
    label_prevalence: float,
    score_distribution: dict,
) -> str:
    """Log the HOSPITAL baseline to Vertex AI Experiments.

    Returns the run name for reference.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID)

    run_name = f"hospital-baseline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with aiplatform.start_run(run_name, experiment=EXPERIMENT_NAME) as run:
        run.log_params({
            "method": "HOSPITAL_score",
            "split": "train",
            "max_points": 13,
            "reference": "Donzé et al., JAMA Intern Med. 2013",
        })
        run.log_metrics({
            "hospital_aucpr": hospital_aucpr,
            "n_patients": n_patients,
            "label_prevalence": label_prevalence,
            "score_low_pct": score_distribution.get("low_pct", 0),
            "score_intermediate_pct": score_distribution.get("intermediate_pct", 0),
            "score_high_pct": score_distribution.get("high_pct", 0),
        })

    return run_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 60)
    print(f"HOSPITAL Score Baseline — {EXPERIMENT_NAME}")
    print(f"Table: {HOSPITAL_TABLE}")
    print("=" * 60)

    # 1. Query the HOSPITAL score table (train split only)
    sql = f"""
        SELECT hospital_score, {LABEL_COLUMN}
        FROM `{HOSPITAL_TABLE}`
        WHERE split_name = 'train'
    """
    print("\nQuerying HOSPITAL score table …")
    df = run_query(sql)
    print(f"  Pulled {len(df):,} rows")

    # 2. Compute AUCPR
    aucpr = compute_aucpr(df[LABEL_COLUMN], df["hospital_score"])
    prevalence = df[LABEL_COLUMN].mean()

    # Score distribution
    score_bins = pd.cut(
        df["hospital_score"],
        bins=[-1, 4, 6, 20],
        labels=["low", "intermediate", "high"],
    )
    dist = score_bins.value_counts(normalize=True).to_dict()

    print(f"\n  HOSPITAL AUCPR:   {aucpr:.4f}")
    print(f"  Label prevalence: {prevalence:.4f}")
    print(f"  Score distribution:")
    for tier in ["low", "intermediate", "high"]:
        pct = dist.get(tier, 0) * 100
        print(f"    {tier:<14s}: {pct:5.1f}%")

    # 3. Log to Vertex AI Experiments
    print(f"\nLogging to Vertex AI Experiments …")
    try:
        run_name = log_to_vertex_experiments(
            hospital_aucpr=aucpr,
            n_patients=len(df),
            label_prevalence=prevalence,
            score_distribution={
                "low_pct": dist.get("low", 0) * 100,
                "intermediate_pct": dist.get("intermediate", 0) * 100,
                "high_pct": dist.get("high", 0) * 100,
            },
        )
        print(f"  ✓ Logged to experiment '{EXPERIMENT_NAME}'")
        print(f"  ✓ Run: {run_name}")
    except Exception as e:
        print(f"  ⚠ Vertex AI Experiments logging failed: {e}")
        print(f"  (AUCPR was computed successfully — retry logging after "
              f"authenticating with 'gcloud auth application-default login')")
        return 1

    # 4. Write local summary for downstream consumers
    summary = {
        "method": "HOSPITAL_score",
        "aucpr": aucpr,
        "n_patients": len(df),
        "label_prevalence": prevalence,
        "score_distribution": {
            k: round(v * 100, 1) for k, v in dist.items()
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = (
        _PROJECT_ROOT / "artifacts" / "hospital_baseline.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Local summary → {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
