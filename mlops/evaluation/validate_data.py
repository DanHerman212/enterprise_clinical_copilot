#!/usr/bin/env python3
"""
validate_data.py — Evidently AI data validation gate (v0.7.21).

Reads the analytics_dataset training split as the reference baseline, then
compares the validation and test splits for distributional drift and data
quality regressions. Produces HTML/JSON reports and a machine-readable
summary that downstream steps (feature selection, training) consume as
their entry gate.

Usage:
    python mlops/evaluation/validate_data.py

Exit codes:
    0 — gate passed (or warnings only, no failures)
    1 — gate failed (drift or quality breach)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure mlops is on sys.path so `from mlops.data import …` works
# regardless of the working directory the script is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from evidently import Report
from evidently.metrics import (
    DatasetMissingValueCount,
    DriftedColumnsCount,
    MissingValueCount,
    ValueDrift,
)

from mlops.data.bigquery import run_query
from mlops.data.config import (
    CATEGORICAL_FEATURES,
    EVIDENTLY_CONFIG,
    FULL_TABLE_REF,
    LABEL_COLUMN,
    MISSINGNESS_POLICY_CSV,
    NON_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    SPLIT_COLUMN,
    SPLITS,
    VALIDATION_DIR,
)

# ---------------------------------------------------------------------------
# Paths & thresholds
# ---------------------------------------------------------------------------

RID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = VALIDATION_DIR / RID
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = EVIDENTLY_CONFIG["drift_threshold"]
MAX_DRIFTED_SHARE = EVIDENTLY_CONFIG["max_drifted_share"]
MISSINGNESS_TOLERANCE = EVIDENTLY_CONFIG["missingness_tolerance_pct"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_splits() -> dict[str, pd.DataFrame]:
    """Query all analysis splits from BigQuery in a single scan."""
    split_list = ", ".join(f"'{s}'" for s in SPLITS.values())
    sql = f"""
        SELECT *
        FROM `{FULL_TABLE_REF}`
        WHERE {SPLIT_COLUMN} IN ({split_list})
    """
    print(f"Querying {FULL_TABLE_REF} …")
    df = run_query(sql)
    print(f"  Pulled {len(df):,} rows × {len(df.columns)} cols")

    splits: dict[str, pd.DataFrame] = {}
    for name, value in SPLITS.items():
        mask = df[SPLIT_COLUMN] == value
        splits[name] = (
            df.loc[mask].drop(columns=[SPLIT_COLUMN]).reset_index(drop=True)
        )
        print(f"  {name:>10s}: {len(splits[name]):,} rows")
    return splits


def load_missingness_policy() -> pd.DataFrame | None:
    """Load the missingness policy CSV if available."""
    policy_path = Path(MISSINGNESS_POLICY_CSV)
    if not policy_path.exists():
        print(f"  ⚠ Missingness policy not found at {policy_path} — "
              "skipping expected-null checks.")
        return None
    return pd.read_csv(policy_path)


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only feature columns + label, dropping ID/meta columns."""
    drop_cols = [c for c in NON_FEATURE_COLUMNS
                 if c != LABEL_COLUMN and c in df.columns]
    return df.drop(columns=drop_cols)


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert nullable pandas dtypes (Int64, boolean) to standard numpy types.

    BigQuery returns Int64 and boolean (nullable) which Evidently 0.7.21
    does not recognise as valid numeric/categorical column types.
    """
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if str(dtype) == "Int64":
            df[col] = df[col].astype("float64")
        elif str(dtype) == "boolean":
            df[col] = df[col].astype("int64")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (excludes IDs, meta, and label)."""
    exclude = set(NON_FEATURE_COLUMNS)
    return [c for c in df.columns if c not in exclude]


def _extract_counter(widget: dict, label: str) -> str:
    """Pull a counter value from a widget's counters list."""
    for counter in widget.get("params", {}).get("counters", []):
        if counter.get("label") == label:
            return counter.get("value", "0")
    return "0"


def _metric_params(result: dict) -> dict:
    """Extract the metric params dict from a metric_result entry."""
    w0 = (result.get("widget") or [{}])[0]
    return (
        w0.get("metric_value_location", {})
        .get("metric", {})
        .get("params", {})
    )


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------


def run_data_quality(
    train: pd.DataFrame, policy: pd.DataFrame | None
) -> dict[str, Any]:
    """Missingness check on the train split against the policy baseline."""
    print("\n── Data Quality (train split) ──")

    feat_cols = feature_columns(train)
    metrics: list = [DatasetMissingValueCount()]
    for col in feat_cols:
        metrics.append(MissingValueCount(column=col))

    report = Report(metrics=metrics)
    snapshot = report.run(current_data=train, reference_data=None)
    snapshot.save_html(str(OUT_DIR / "data_quality.html"))
    snapshot.save_json(str(OUT_DIR / "data_quality.json"))
    print("  ✓ Saved data_quality.html / .json")

    raw = snapshot.dump_dict()
    mr = raw.get("metric_results", {})

    violations: list[str] = []
    if policy is not None:
        policy_lookup = dict(zip(policy["column"], policy["null_pct"]))
        for col in feat_cols:
            expected_null = policy_lookup.get(col, 0)
            current_null_pct = 0.0
            for _mid, result in mr.items():
                params = _metric_params(result)
                if (params.get("type") == "evidently:metric_v2:MissingValueCount"
                        and params.get("column") == col):
                    share_str = _extract_counter(result["widget"][0], "Share")
                    current_null_pct = float(share_str)
                    break

            if current_null_pct > expected_null + MISSINGNESS_TOLERANCE:
                msg = (
                    f"  ❌ {col}: null% {current_null_pct:.1f}% exceeds "
                    f"expected {expected_null:.1f}% + "
                    f"{MISSINGNESS_TOLERANCE:.0f}% tolerance"
                )
                print(msg)
                violations.append(msg)
            elif current_null_pct > expected_null + 0.01:
                print(
                    f"  ⚠ {col}: null% {current_null_pct:.1f}% > "
                    f"expected {expected_null:.1f}% (within tolerance)"
                )

    status = "fail" if violations else "pass"
    print(f"  Data quality gate: {status.upper()}")
    return {"status": status, "violations": violations}


# ---------------------------------------------------------------------------
# Data Drift
# ---------------------------------------------------------------------------


def run_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    """Per-column + aggregate drift of *current* vs *reference* (train)."""
    print(f"\n── Data Drift: train vs {split_name} ──")

    feat_cols = feature_columns(reference)
    metrics: list = [DriftedColumnsCount(drift_share=DRIFT_THRESHOLD)]
    for col in feat_cols:
        metrics.append(ValueDrift(column=col))

    report = Report(metrics=metrics)
    snapshot = report.run(current_data=current, reference_data=reference)
    snapshot.save_html(str(OUT_DIR / f"drift_train_vs_{split_name}.html"))
    snapshot.save_json(str(OUT_DIR / f"drift_train_vs_{split_name}.json"))
    print(f"  ✓ Saved drift_train_vs_{split_name}.html / .json")

    raw = snapshot.dump_dict()
    mr = raw.get("metric_results", {})

    # Aggregate drift share
    drift_share = 0.0
    for _mid, result in mr.items():
        params = _metric_params(result)
        if params.get("type") == "evidently:metric_v2:DriftedColumnsCount":
            share_str = _extract_counter(result["widget"][0], "Share")
            drift_share = float(share_str) / 100.0
            break

    # Per-column drift
    drifted_cols: list[dict] = []
    for _mid, result in mr.items():
        params = _metric_params(result)
        if params.get("type") == "evidently:metric_v2:ValueDrift":
            col = params.get("column", "?")
            for widget in result.get("widget", []):
                if widget.get("type") == "counter":
                    for c in widget.get("params", {}).get("counters", []):
                        if c.get("label") == "Drift score":
                            score = float(c.get("value", "0"))
                            if score > DRIFT_THRESHOLD:
                                drifted_cols.append({
                                    "feature": col,
                                    "drift_score": score,
                                })
                            break

    # Gate decision
    if drift_share > MAX_DRIFTED_SHARE:
        status = "fail"
    elif drift_share > 0:
        status = "warn"
    else:
        status = "pass"

    print(f"  Drifted columns: {len(drifted_cols)} "
          f"({drift_share:.1%} of features)")
    for dc in drifted_cols[:5]:
        print(f"    • {dc['feature']}: {dc['drift_score']:.4f}")
    if len(drifted_cols) > 5:
        print(f"    … and {len(drifted_cols) - 5} more")
    print(f"  Data drift gate: {status.upper()}")

    return {
        "status": status,
        "drift_share": drift_share,
        "n_drifted": len(drifted_cols),
        "drifted_columns": drifted_cols,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 60)
    print(f"Evidently Data Validation — {RID}")
    print(f"Table:   {FULL_TABLE_REF}")
    print(f"Drift threshold: {DRIFT_THRESHOLD}  |  "
          f"Max drifted share: {MAX_DRIFTED_SHARE}")
    print("=" * 60)

    # 1. Load data
    splits = load_splits()
    train_raw = splits["train"]
    val_raw = splits["validation"]
    test_raw = splits["test"]

    train = _normalize_dtypes(drop_non_features(train_raw))
    val = _normalize_dtypes(drop_non_features(val_raw))
    test = _normalize_dtypes(drop_non_features(test_raw))

    feat_cols = feature_columns(train)
    print(f"\nFeature columns: {len(feat_cols)}")
    print(f"Label prevalence (train): {train[LABEL_COLUMN].mean():.4f}")

    # 2. Load missingness policy
    policy = load_missingness_policy()

    # 3. Data quality on train
    quality_result = run_data_quality(train, policy)

    # 4. Data drift: val vs train, test vs train
    drift_val = run_data_drift(train, val, "validation")
    drift_test = run_data_drift(train, test, "test")

    # 5. Overall gate
    all_statuses = [
        quality_result["status"],
        drift_val["status"],
        drift_test["status"],
    ]
    if "fail" in all_statuses:
        overall = "fail"
    elif "warn" in all_statuses:
        overall = "warn"
    else:
        overall = "pass"

    # 6. Write summary
    summary = {
        "rid": RID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "table": FULL_TABLE_REF,
        "evidently_version": "0.7.21",
        "config": {
            "drift_threshold": DRIFT_THRESHOLD,
            "max_drifted_share": MAX_DRIFTED_SHARE,
            "missingness_tolerance_pct": MISSINGNESS_TOLERANCE,
        },
        "splits": {
            "train": {"n_rows": len(train_raw), "n_features": len(feat_cols)},
            "validation": {"n_rows": len(val_raw)},
            "test": {"n_rows": len(test_raw)},
        },
        "data_quality": quality_result,
        "data_drift": {
            "train_vs_validation": {
                k: v for k, v in drift_val.items()
                if k != "drifted_columns"
            },
            "train_vs_test": {
                k: v for k, v in drift_test.items()
                if k != "drifted_columns"
            },
        },
        "top_drifted_train_vs_validation": drift_val["drifted_columns"][:10],
        "top_drifted_train_vs_test": drift_test["drifted_columns"][:10],
        "overall_gate": overall,
    }

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n✓ Summary → {summary_path}")

    # Latest symlink
    latest = VALIDATION_DIR / "summary.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(summary_path.name)

    print(f"\n{'=' * 60}")
    print(f"OVERALL GATE: {overall.upper()}")
    print(f"{'=' * 60}")

    return 0 if overall != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
