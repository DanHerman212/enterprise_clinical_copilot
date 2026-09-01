"""
fairness_audit — subgroup **error-rate parity** audit at the operating threshold.

Design (agreed with stakeholders)
---------------------------------
* **Primary signal — Equal Opportunity (TPR / recall parity).** For a
  readmission *screen*, the paramount question is whether the model
  systematically *misses* readmissions (under-flags) in a subgroup. A high
  false-negative rate in a group means that group is under-served, so TPR parity
  is the headline fairness metric.
* **Secondary signal — Predictive Equality (FPR parity).** Over-flagging a group
  wastes care-management capacity; a real equity concern, but generally less
  harmful than missing a readmission.
* **PPV / NPV are reported for context only.** They are prevalence-sensitive, so
  a "PPV gap" partly reflects different subgroup base rates rather than model
  bias — hence they are not the parity signal.

Taming small-sample noise
-------------------------
* ``race`` is rolled up from ~28 fine-grained MIMIC levels into standardized OMB
  buckets so the gap is not dominated by the Law of Small Numbers.
* Every rate is reported with a **95% Wilson confidence interval** so point-
  estimate gaps over small subgroups are not over-interpreted.
* Levels smaller than ``_MIN_SUBGROUP`` are skipped.

Guardrail posture
-----------------
This audit is a diagnostic **yellow light**: the TPR/FPR gaps are surfaced to the
experiment UI and returned as advisory pass flags, but they do **not** hard-gate
model registration (small noisy subgroups + data drift would break builds).
Human review decides deployment.

Runs in parallel with evaluate_test after train-final.
"""

import json
import math
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from kfp import dsl
from ._image import TRAINING_IMAGE, component
from ._artifact_integrity import load as _model_load

# Levels smaller than this are skipped (rates too unstable to compare).
_MIN_SUBGROUP = 50


def _bucket_age(age: pd.Series) -> pd.Series:
    """Coarse clinical age bands from a raw numeric age column."""
    bins = [0, 45, 65, 80, 200]
    labels = ["18-44", "45-64", "65-79", "80+"]
    return pd.cut(
        pd.to_numeric(age, errors="coerce"), bins=bins, labels=labels, right=False
    )


def _omb_race(raw) -> str:
    """Map a raw MIMIC race/ethnicity string to a standardized OMB bucket.

    Rolls the fine-grained levels into 6 stable categories (+ Other/Unknown) so a
    parity gap is not an artifact of tiny per-level sample sizes.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "Other/Unknown"
    s = str(raw).upper()
    if "HISPANIC" in s or "LATINO" in s or "SOUTH AMERICAN" in s:
        return "Hispanic or Latino"
    if "BLACK" in s or "AFRICAN" in s:
        return "Black or African American"
    if "ASIAN" in s:
        return "Asian"
    if "AMERICAN INDIAN" in s or "ALASKA NATIVE" in s:
        return "American Indian or Alaska Native"
    if "NATIVE HAWAIIAN" in s or "PACIFIC ISLANDER" in s:
        return "Native Hawaiian or Pacific Islander"
    if "WHITE" in s:
        return "White"
    return "Other/Unknown"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion ``k / n``."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _group_rates(y_true: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    """Error-rate + predictive metrics for one slice, with Wilson CIs on TPR/FPR."""
    tp = int(((y_pred_bin == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true == 0)).sum())
    tn = int(((y_pred_bin == 0) & (y_true == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true == 1)).sum())
    pos = tp + fn  # actual positives  -> TPR denominator
    neg = fp + tn  # actual negatives  -> FPR denominator

    tpr = tp / pos if pos > 0 else float("nan")
    fpr = fp / neg if neg > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    tpr_lo, tpr_hi = _wilson_ci(tp, pos)
    fpr_lo, fpr_hi = _wilson_ci(fp, neg)

    def _r(x):
        return round(float(x), 4) if not (isinstance(x, float) and math.isnan(x)) else float("nan")

    return {
        "n": int(len(y_true)),
        "prevalence": _r(np.mean(y_true)) if len(y_true) else float("nan"),
        "tpr": _r(tpr), "tpr_ci": [_r(tpr_lo), _r(tpr_hi)],
        "fpr": _r(fpr), "fpr_ci": [_r(fpr_lo), _r(fpr_hi)],
        "ppv": _r(ppv), "npv": _r(npv),
    }


def _onehot_to_label(frame: pd.DataFrame, col_to_label: dict[str, str]) -> pd.Series | None:
    """Reconstruct a categorical label Series from its one-hot columns.

    The one-hot block is mutually exclusive (exactly one column is 1 per row,
    incl. the ``_unknown`` catch-all), so the winning column identifies the
    level. Returns ``None`` if none of the expected columns are present.
    """
    cols = [c for c in col_to_label if c in frame.columns]
    if not cols:
        return None
    winner = frame[cols].idxmax(axis=1)
    return winner.map(col_to_label).astype("string")


def _derive_subgroups(sens: pd.DataFrame) -> dict[str, pd.Series]:
    """Build audit slices from the (one-hot ENCODED) feature frame.

    The sensitive attributes are reconstructed from the model's numeric feature
    columns: ``gender`` (0/1), ``race_*`` / ``insurance_*`` one-hot blocks, and
    the numeric ``age`` passthrough — so no separate raw passthrough is needed
    and the slices exactly match what the model consumed.
    """
    from src import encoding

    groups: dict[str, pd.Series] = {}

    if "gender" in sens:
        # gender is encoded 1 == male (see src.encoding.BINARY_FEATURES).
        groups["gender"] = (
            pd.to_numeric(sens["gender"], errors="coerce")
            .map({1: "M", 0: "F"})
            .astype("string")
            .fillna("Unknown")
        )

    race_map = {f"race_{slug}": label for slug, label in encoding.RACE_BUCKETS}
    race_map["race_unknown"] = "Other/Unknown"
    race = _onehot_to_label(sens, race_map)
    if race is not None:
        groups["race"] = race

    ins_map = {
        f"insurance_{slug}": raw
        for raw, slug in encoding.ONEHOT_DIRECT["insurance"]
    }
    ins_map["insurance_unknown"] = "Unknown"
    insurance = _onehot_to_label(sens, ins_map)
    if insurance is not None:
        groups["insurance"] = insurance

    if "age" in sens:
        groups["age_bucket"] = _bucket_age(sens["age"]).astype("string")
    return groups


def run_fairness_audit(
    *,
    x_test_path: str,
    y_test_path: str,
    model_artifact_path: str,
    fairness_report_json: str,
    fairness_html_path: str,
    tuned_threshold: float = 0.5,
    max_tpr_gap: float = 0.15,
    max_fpr_gap: float = 0.15,
) -> dict:
    """Audit error-rate parity at the tuned threshold; write JSON + HTML; return report.

    Sensitive attributes (gender, race, insurance, age) are read directly from
    ``X_test`` — they are model features (categoricals keep their raw string
    levels), so no separate passthrough is needed. The report carries
    per-subgroup TPR/FPR (each with a 95% Wilson CI) plus PPV/NPV for context,
    the max TPR/FPR gap per subgroup, and advisory pass flags (Equal Opportunity
    = TPR parity, Predictive Equality = FPR parity).
    """
    X_test = pd.read_parquet(x_test_path)
    y_test = pd.read_parquet(y_test_path).iloc[:, 0].to_numpy().astype(int)
    model = _model_load(model_artifact_path)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred_bin = (proba >= tuned_threshold).astype(int)

    report = {
        "threshold": float(tuned_threshold),
        "primary_signal": "equal_opportunity_tpr_parity",
        "secondary_signal": "predictive_equality_fpr_parity",
        "max_tpr_gap_threshold": float(max_tpr_gap),
        "max_fpr_gap_threshold": float(max_fpr_gap),
        "min_subgroup_n": _MIN_SUBGROUP,
        "overall": _group_rates(y_test, y_pred_bin),
        "subgroups": {},
        "gaps": {},
    }

    eo_pass = True  # Equal Opportunity  (TPR parity, primary)
    pe_pass = True  # Predictive Equality (FPR parity, secondary)

    for name, series in _derive_subgroups(X_test).items():
        series = series.reset_index(drop=True)
        levels = {}
        for label in sorted(str(x) for x in series.dropna().unique()):
            mask = (series == label).to_numpy()
            if mask.sum() < _MIN_SUBGROUP:
                continue
            levels[label] = _group_rates(y_test[mask], y_pred_bin[mask])
        if not levels:
            continue
        report["subgroups"][name] = levels

        tprs = [v["tpr"] for v in levels.values() if not math.isnan(v["tpr"])]
        fprs = [v["fpr"] for v in levels.values() if not math.isnan(v["fpr"])]
        tpr_gap = (max(tprs) - min(tprs)) if len(tprs) > 1 else float("nan")
        fpr_gap = (max(fprs) - min(fprs)) if len(fprs) > 1 else float("nan")
        report["gaps"][name] = {
            "tpr_gap": round(tpr_gap, 4) if not math.isnan(tpr_gap) else None,
            "fpr_gap": round(fpr_gap, 4) if not math.isnan(fpr_gap) else None,
        }
        if not math.isnan(tpr_gap) and tpr_gap > max_tpr_gap:
            eo_pass = False
        if not math.isnan(fpr_gap) and fpr_gap > max_fpr_gap:
            pe_pass = False

    report["equal_opportunity_pass"] = eo_pass
    report["predictive_equality_pass"] = pe_pass

    with open(fairness_report_json, "w") as f:
        json.dump(report, f, indent=2)
    with open(fairness_html_path, "w") as f:
        f.write(_build_fairness_html(report))

    print(f"  Equal Opportunity  (TPR parity, primary):    {'PASS' if eo_pass else 'REVIEW'}")
    print(f"  Predictive Equality (FPR parity, secondary): {'PASS' if pe_pass else 'REVIEW'}")
    for name, g in report["gaps"].items():
        print(f"    {name:12s} TPR gap={g['tpr_gap']}  FPR gap={g['fpr_gap']}")
    return report


def _fmt_ci(ci) -> str:
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "—"
    return f"[{lo:.2f}, {hi:.2f}]"


def _build_fairness_html(report: dict) -> str:
    """Render the parity audit as an HTML report (tables; no image deps)."""
    thr = report["threshold"]
    eo = report["equal_opportunity_pass"]
    pe = report["predictive_equality_pass"]
    max_tpr = report["max_tpr_gap_threshold"]
    max_fpr = report["max_fpr_gap_threshold"]

    def _light(ok):
        color = "#2ca02c" if ok else "#e6a700"
        text = "PASS" if ok else "REVIEW"
        return f'<b style="color:{color}">{text}</b>'

    sections = []
    for name, levels in report["subgroups"].items():
        gap = report["gaps"].get(name, {})
        rows = []
        for label, m in levels.items():
            rows.append(
                f"<tr><td>{label}</td><td>{m['n']:,}</td>"
                f"<td>{m['prevalence']:.3f}</td>"
                f"<td><b>{m['tpr']:.3f}</b> {_fmt_ci(m['tpr_ci'])}</td>"
                f"<td>{m['fpr']:.3f} {_fmt_ci(m['fpr_ci'])}</td>"
                f"<td>{m['ppv']:.3f}</td><td>{m['npv']:.3f}</td></tr>"
            )
        sections.append(
            f"<h3>{name} &nbsp;<small>(TPR gap = {gap.get('tpr_gap')}, "
            f"FPR gap = {gap.get('fpr_gap')})</small></h3>"
            '<table border="1" cellpadding="5" style="border-collapse:collapse">'
            "<tr><th>Level</th><th>n</th><th>Prevalence</th>"
            "<th>TPR / Recall (95% CI)</th><th>FPR (95% CI)</th>"
            "<th>PPV</th><th>NPV</th></tr>" + "".join(rows) + "</table>"
        )

    ov = report["overall"]
    return f"""
<h2>Fairness Audit — Error-Rate Parity (threshold = {thr:.4f})</h2>
<p><b>Primary:</b> Equal Opportunity (TPR / recall parity) — does the model miss
readmissions more in some groups? &nbsp; {_light(eo)} (gap threshold {max_tpr:.2f})<br>
<b>Secondary:</b> Predictive Equality (FPR parity) — does it over-flag some
groups? &nbsp; {_light(pe)} (gap threshold {max_fpr:.2f})</p>
<p><i>PPV / NPV are shown for context only; they are prevalence-sensitive and are
not the parity signal. This audit is a diagnostic yellow light — gaps flag
subgroups for human review, they do not block registration.</i></p>
<p><b>Overall:</b> n={ov['n']:,}, prevalence={ov['prevalence']:.3f},
TPR={ov['tpr']:.3f} {_fmt_ci(ov['tpr_ci'])}, FPR={ov['fpr']:.3f} {_fmt_ci(ov['fpr_ci'])}.</p>
{''.join(sections)}
"""


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-aiplatform"],
)
def fairness_audit(
    x_test: dsl.Input[dsl.Dataset],
    y_test: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    tuned_threshold: float,
    fairness_html: dsl.Output[dsl.HTML],
    max_tpr_gap: float = 0.15,
    max_fpr_gap: float = 0.15,
    project_id: str = "",
    location: str = "us-east1",
    experiment_name: str = "",
    pipeline_job_name: str = "",
) -> NamedTuple(
    "FairnessOutputs",
    [("equal_opportunity_pass", bool), ("predictive_equality_pass", bool)],
):
    """KFP component: subgroup error-rate parity audit at the tuned threshold."""
    import os

    from pipelines.components.fairness_audit import run_fairness_audit
    from pipelines.components._experiment import companion_run, safe_log_metrics

    report_json = os.path.join(os.path.dirname(fairness_html.path), "report.json")
    report = run_fairness_audit(
        x_test_path=x_test.path,
        y_test_path=y_test.path,
        model_artifact_path=model_artifact.path,
        fairness_report_json=report_json,
        fairness_html_path=fairness_html.path,
        tuned_threshold=tuned_threshold,
        max_tpr_gap=max_tpr_gap,
        max_fpr_gap=max_fpr_gap,
    )

    # Surface the parity gaps to the experiment UI (diagnostic; never gates).
    flat: dict[str, float] = {}
    for name, g in report.get("gaps", {}).items():
        if g.get("tpr_gap") is not None:
            flat[f"fairness_tpr_gap_{name}"] = g["tpr_gap"]
        if g.get("fpr_gap") is not None:
            flat[f"fairness_fpr_gap_{name}"] = g["fpr_gap"]
    tpr_gaps = [v for k, v in flat.items() if k.startswith("fairness_tpr_gap_")]
    fpr_gaps = [v for k, v in flat.items() if k.startswith("fairness_fpr_gap_")]
    if tpr_gaps:
        flat["fairness_max_tpr_gap"] = max(tpr_gaps)
    if fpr_gaps:
        flat["fairness_max_fpr_gap"] = max(fpr_gaps)

    with companion_run(
        project_id=project_id, location=location,
        experiment=experiment_name, pipeline_job_name=pipeline_job_name,
    ) as ap:
        safe_log_metrics(ap, flat)

    return (report["equal_opportunity_pass"], report["predictive_equality_pass"])
