"""
fairness_audit — Subgroup NPV/PPV analysis across demographic slices.

Runs in parallel with evaluate_test after train-final.
"""

import json
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from kfp import dsl
from ._image import TRAINING_IMAGE

# Subgroup columns and how to derive them from the feature set.
# Age is bucketed; race/gender/insurance are categorical features
# stored as pandas category dtype in the parquet files.
FAIRNESS_SUBGROUPS = {
    "gender": {"column": "gender"},
    "race": {"column": "race"},
    "insurance": {"column": "insurance"},
}


def _bucket_age(age_series: pd.Series) -> pd.Series:
    bins = [0, 45, 65, 80, 200]
    labels = ["18-44", "45-64", "65-79", "80+"]
    return pd.cut(age_series, bins=bins, labels=labels, right=False)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute NPV, PPV from binary predictions at default 0.5 threshold."""
    y_pred_bin = (y_pred >= 0.5).astype(int)

    tp = ((y_pred_bin == 1) & (y_true == 1)).sum()
    fp = ((y_pred_bin == 1) & (y_true == 0)).sum()
    tn = ((y_pred_bin == 0) & (y_true == 0)).sum()
    fn = ((y_pred_bin == 0) & (y_true == 1)).sum()

    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
    }


def run_fairness_audit(
    *,
    x_test_path: str,
    y_test_path: str,
    model_artifact_path: str,
    fairness_report_json: str,
    fairness_html_path: str,
    max_ppv_gap: float = 0.15,
    max_npv_gap: float = 0.15,
) -> tuple[bool, bool]:
    """Audit fairness.  Returns (ppv_pass, npv_pass)."""
    X_test = pd.read_parquet(x_test_path)
    y_test = pd.read_parquet(y_test_path).iloc[:, 0]
    model = joblib.load(model_artifact_path)

    y_pred = model.predict_proba(X_test)[:, 1]

    # Reconstruct readable subgroup values from category dtype.
    report = {"overall": _compute_metrics(y_test.values, y_pred)}
    ppv_ok = True
    npv_ok = True

    for subgroup_name, cfg in FAIRNESS_SUBGROUPS.items():
        col = cfg["column"]
        if col == "age":
            groups = _bucket_age(X_test["age"] if hasattr(X_test["age"], "cat") else X_test["age"])
        elif col in X_test.columns:
            series = X_test[col]
            if hasattr(series, "cat"):
                groups = series.astype(str)
            else:
                groups = series
        else:
            continue

        report[subgroup_name] = {}
        for label in sorted(groups.dropna().unique()):
            mask = (groups == label).values
            if mask.sum() < 50:  # skip tiny subgroups
                continue
            report[subgroup_name][str(label)] = _compute_metrics(
                y_test.values[mask], y_pred[mask],
            )

        # Check gaps.
        ppvs = [v["ppv"] for v in report[subgroup_name].values() if not np.isnan(v["ppv"])]
        npvs = [v["npv"] for v in report[subgroup_name].values() if not np.isnan(v["npv"])]
        if ppvs and (max(ppvs) - min(ppvs)) > max_ppv_gap:
            ppv_ok = False
        if npvs and (max(npvs) - min(npvs)) > max_npv_gap:
            npv_ok = False

    with open(fairness_report_json, "w") as f:
        json.dump(report, f, indent=2)

    # Dump a simple HTML version so the UI can render it.
    html_content = "<h2>Fairness Audit Report</h2><pre>" + json.dumps(report, indent=2) + "</pre>"
    with open(fairness_html_path, "w") as f:
        f.write(html_content)

    print(f"  PPV across subgroups: {'PASS' if ppv_ok else 'GAP > {:.0%}'.format(max_ppv_gap)}")
    print(f"  NPV across subgroups: {'PASS' if npv_ok else 'GAP > {:.0%}'.format(max_npv_gap)}")
    for subgroup_name, metrics in report.items():
        if subgroup_name == "overall":
            continue
        print(f"\n  {subgroup_name}:")
        for label, m in metrics.items():
            print(f"    {str(label):<30s}  n={m['n']:>6d}  PPV={m['ppv']:.3f}  NPV={m['npv']:.3f}")

    return (ppv_ok, npv_ok)


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[],
)
def fairness_audit(
    x_test: dsl.Input[dsl.Dataset],
    y_test: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    fairness_html: dsl.Output[dsl.HTML],
) -> NamedTuple(
    "FairnessOutputs",
    [("ppv_pass", bool), ("npv_pass", bool)],
):
    """KFP component: subgroup fairness audit (NPV/PPV)."""
    from pipelines.components.fairness_audit import run_fairness_audit

    # KFP v2 doesn't native output raw JSON as an artifact type perfectly, 
    # but the HTML artifact guarantees UI rendering.
    import os
    fairness_report_json = os.path.join(os.path.dirname(fairness_html.path), "report.json")
    
    ppv_pass, npv_pass = run_fairness_audit(
        x_test_path=x_test.path, y_test_path=y_test.path,
        model_artifact_path=model_artifact.path,
        fairness_report_json=fairness_report_json,
        fairness_html_path=fairness_html.path,
    )
    return (ppv_pass, npv_pass)
