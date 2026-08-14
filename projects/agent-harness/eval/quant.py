"""Quantitative half of the golden-set eval — score the FULL test split with the
deployed endpoint and report threshold-free + operating-point metrics.

This is the "prove the math on 1,000 patients" pass. It also writes
``holdout_scored.jsonl`` (hadm_id, probability, label) that sample.py consumes
to build the calibrated strata.

Usage (from the harness root):
    .venv/bin/python eval/quant.py
"""

import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

import numpy as np
from google.cloud import bigquery
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from mcp_server.config import PROJECT, TABLE_FQN
from mcp_server.endpoint import endpoint
from mcp_server.features import to_vector
from mcp_server.features.manifest import feature_order

BATCH = 64
RESULTS = HARNESS / "eval" / "results"
SCORED_PATH = RESULTS / "holdout_scored.jsonl"
METRICS_PATH = RESULTS / "quant_metrics.json"


def main() -> int:
    bq = bigquery.Client(project=PROJECT)
    order = feature_order()

    rows = [
        dict(r)
        for r in bq.query(
            f"SELECT * FROM `{TABLE_FQN}` WHERE split_name = 'test' "
            f"ORDER BY hadm_id"
        ).result()
    ]
    if not rows:
        print("no test-split rows; aborting")
        return 1
    print(f"Test split: {len(rows)} rows")

    ep = endpoint()
    probs: list[float] = []
    threshold: float | None = None
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        preds = ep.predict(instances=[to_vector(r, order) for r in chunk]).predictions
        for row, pred in zip(chunk, preds):
            p = float(pred["probability"])
            probs.append(p)
            if threshold is None:
                threshold = float(pred["threshold"])
    if len(probs) != len(rows):
        print(f"FAILED: {len(probs)} probs for {len(rows)} rows")
        return 1

    labels = np.array([int(r["readmission_30d"]) for r in rows])
    scores = np.array(probs)

    # --- metrics ---
    aucpr = average_precision_score(labels, scores)
    auroc = roc_auc_score(labels, scores)
    brier = brier_score_loss(labels, scores)
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    # operating point at the bundle threshold
    prec_at = float(np.interp(threshold, np.sort(scores), precisions[np.argsort(scores)])) if False else None
    # precision/recall at threshold from the PR curve via the nearest score
    idx = int(np.searchsorted(np.sort(scores), threshold, side="left"))
    p_at_t = float(precisions[idx]) if idx < len(precisions) else None
    r_at_t = float(recalls[idx]) if idx < len(recalls) else None
    frac_pos, mean_pred = calibration_curve(labels, scores, n_bins=10)

    record = {
        "rows": len(rows),
        "readmission_rate": round(float(labels.mean()), 4),
        "threshold": threshold,
        "aucpr": round(aucpr, 4),
        "auroc": round(auroc, 4),
        "brier": round(brier, 4),
        "precision_at_threshold": round(p_at_t, 4) if p_at_t is not None else None,
        "recall_at_threshold": round(r_at_t, 4) if r_at_t is not None else None,
        "calibration_curve": {
            "fraction_positive": [round(float(x), 4) for x in frac_pos],
            "mean_predicted": [round(float(x), 4) for x in mean_pred],
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(record, indent=2) + "\n")

    with SCORED_PATH.open("w") as fh:
        for row, p in zip(rows, probs):
            fh.write(json.dumps({
                "hadm_id": int(row["hadm_id"]),
                "probability": round(float(p), 6),
                "readmission_30d": int(row["readmission_30d"]),
            }) + "\n")

    print(f"\n=== QUANT SUMMARY ===")
    print(f"  rows={record['rows']}  rate={record['readmission_rate']}")
    print(f"  AUCPR={aucpr:.4f}  AUROC={auroc:.4f}  Brier={brier:.4f}")
    print(f"  P@thr={record['precision_at_threshold']}  R@thr={record['recall_at_threshold']}")
    print(f"  wrote {SCORED_PATH.name} + {METRICS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
