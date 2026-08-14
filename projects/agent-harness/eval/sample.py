"""Build the stratified golden sample from the DEMO holdout (split_name='demo').

Scores the 3,402 demo-split admissions with the deployed endpoint, bins them
into calibrated risk bands (low <0.10, borderline 0.10-0.20, high >=0.20) and
samples a fixed, reproducible, risk-weighted subset (40 high / 40 borderline /
20 low) for the agent narrative eval.

Usage (harness root):
    .venv/bin/python eval/sample.py
"""

import json
import random
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

from google.cloud import bigquery

from mcp_server.config import PROJECT, TABLE_FQN
from mcp_server.endpoint import endpoint
from mcp_server.features import to_vector
from mcp_server.features.manifest import feature_order

RESULTS = HARNESS / "eval" / "results"
OUT = RESULTS / "golden_sample.json"
SEED = 20260814
BATCH = 64

# Calibrated to this model's distribution (threshold 0.12, range ~0.04-0.32).
BANDS = [
    ("low", 0.0, 0.10, 20),
    ("borderline", 0.10, 0.20, 40),
    ("high", 0.20, 1.01, 40),
]


def main() -> int:
    bq = bigquery.Client(project=PROJECT)
    order = feature_order()

    rows = [
        dict(r)
        for r in bq.query(
            f"SELECT * FROM `{TABLE_FQN}` WHERE split_name = 'demo' "
            f"ORDER BY hadm_id"
        ).result()
    ]
    if not rows:
        print("no demo-split rows; aborting")
        return 1
    print(f"Demo split: {len(rows)} rows")

    ep = endpoint()
    threshold = None
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        preds = ep.predict(instances=[to_vector(r, order) for r in chunk]).predictions
        for row, pred in zip(chunk, preds):
            row["probability"] = round(float(pred["probability"]), 6)
            if threshold is None:
                threshold = float(pred["threshold"])
    print(f"Threshold from endpoint: {threshold}")

    rng = random.Random(SEED)
    chosen = []
    for name, lo, hi, want in BANDS:
        cand = [r for r in rows if lo <= r["probability"] < hi]
        rng.shuffle(cand)
        take = min(want, len(cand))
        chosen.extend(cand[:take])
        flag = "" if take == want else f"  (only {take} available)"
        print(f"  {name:11} [{lo:.2f},{hi:.2f})  avail={len(cand):4d}  take={take}{flag}")

    def _band(p):
        return next(b[0] for b in BANDS if b[1] <= p < b[2])

    sample = [
        {
            "hadm_id": int(r["hadm_id"]),
            "probability": r["probability"],
            "threshold": threshold,
            "readmission_30d": int(r["readmission_30d"]),
            "band": _band(r["probability"]),
        }
        for r in sorted(chosen, key=lambda r: r["probability"])
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"seed": SEED, "n": len(sample), "patients": sample}, indent=2) + "\n"
    )
    print(f"\nWrote golden sample: {len(sample)} patients -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
