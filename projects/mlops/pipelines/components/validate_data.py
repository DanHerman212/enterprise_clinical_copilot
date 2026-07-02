"""
validate_data — Evidently AI data quality and drift gate.

Compares the train (reference) and validation (current) feature distributions
with Evidently. The gate HARD-FAILS the pipeline when the share of drifted
columns exceeds ``max_drifted_share`` (override with ``fail_on_drift=False``).

Evidently 0.4.34 is pinned against the base image's NumPy 1.26.4 (verified);
an unpinned install would pull an Evidently that requires NumPy >= 2 and a
different API.
"""

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from kfp import dsl

from ._image import TRAINING_IMAGE, component


def _drift_share(report: Report) -> float:
    """Extract the share of drifted columns (0..1) from an Evidently report."""
    for metric in report.as_dict()["metrics"]:
        result = metric.get("result", {})
        if "share_of_drifted_columns" in result:
            return float(result["share_of_drifted_columns"])
    raise RuntimeError("Evidently report did not contain a drift share metric.")


def run_validate_data(
    *,
    x_train_path: str,
    x_val_path: str,
    drift_report_html: str,
    quality_report_html: str,
    max_drifted_share: float = 0.2,
    fail_on_drift: bool = True,
) -> bool:
    """Run Evidently drift + quality checks.

    Returns True if the drift share is within ``max_drifted_share``. If it is
    exceeded, raises ``ValueError`` (hard-fail) unless ``fail_on_drift`` is
    False, in which case it returns False.
    """
    X_train = pd.read_parquet(x_train_path)
    X_val = pd.read_parquet(x_val_path)

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=X_train, current_data=X_val)
    drift_report.save_html(drift_report_html)

    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(reference_data=X_train, current_data=X_val)
    quality_report.save_html(quality_report_html)

    drift_share = _drift_share(drift_report)
    passed = drift_share <= max_drifted_share

    print(f"  Drift share: {drift_share:.1%}  (threshold: {max_drifted_share:.1%})")
    print(f"  Gate: {'PASS' if passed else 'FAIL'}")

    if not passed and fail_on_drift:
        raise ValueError(
            f"Data drift share ({drift_share:.1%}) exceeds the threshold "
            f"({max_drifted_share:.1%}). Failing the pipeline."
        )
    return passed


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["evidently==0.4.34", "numpy<2", "pandas", "pyarrow"],
)
def validate_data(
    x_train: dsl.Input[dsl.Dataset],
    x_val: dsl.Input[dsl.Dataset],
    max_drifted_share: float,
    drift_html: dsl.Output[dsl.HTML],
    quality_html: dsl.Output[dsl.HTML],
    fail_on_drift: bool = True,
) -> bool:
    """KFP component: Evidently AI data validation gate (hard-fail on drift)."""
    from pipelines.components.validate_data import run_validate_data

    return run_validate_data(
        x_train_path=x_train.path, x_val_path=x_val.path,
        drift_report_html=drift_html.path,
        quality_report_html=quality_html.path,
        max_drifted_share=max_drifted_share,
        fail_on_drift=fail_on_drift,
    )
