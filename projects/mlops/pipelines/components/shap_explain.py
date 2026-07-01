"""
shap_explain — Tree SHAP global importance and per-patient attributions.

Runs in parallel with evaluate_test after train-final.
"""

from __future__ import annotations

from typing import NamedTuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from kfp import dsl
from ._image import TRAINING_IMAGE


def run_shap_explain(
    *,
    x_test_path: str,
    model_artifact_path: str,
    shap_summary_png: str,
    shap_values_parquet: str,
    top_n: int = 20,
) -> dict[str, float]:
    """Compute SHAP values, save plot and parquet.  Returns top-N feature importance."""
    X_test = pd.read_parquet(x_test_path)
    model = joblib.load(model_artifact_path)

    # SHAP needs the underlying booster, not the sklearn wrapper.
    booster = model.get_booster()

    # For efficiency, sample if test set is large.
    n_sample = min(5000, len(X_test))
    if len(X_test) > n_sample:
        rng = np.random.RandomState(42)
        X_sample = X_test.iloc[rng.choice(len(X_test), n_sample, replace=False)]
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # --- Global importance plot ---
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.35)))
    shap.summary_plot(
        shap_values, X_sample, plot_type="bar", max_display=top_n, show=False,
    )
    ax.set_title("SHAP Feature Importance (Top Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(shap_summary_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Feature importance as dict ---
    importance = {
        col: float(np.abs(shap_values[:, i]).mean())
        for i, col in enumerate(X_sample.columns)
    }
    sorted_importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n],
    )

    # --- Per-patient SHAP values ---
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    shap_df.to_parquet(shap_values_parquet, index=False)

    print(f"  Top-{top_n} SHAP features:")
    for feat, val in sorted_importance.items():
        print(f"    {feat:<35s} {val:.4f}")

    return sorted_importance


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[],
)
def shap_explain(
    x_test_path: dsl.Input[dsl.Dataset],
    model_artifact_path: dsl.Input[dsl.Artifact],
) -> NamedTuple(
    "SHAPOutputs",
    [("shap_summary_png", str), ("shap_values_parquet", str)],
):
    """KFP component: Tree SHAP global importance + per-patient values."""
    run_shap_explain(
        x_test_path=x_test_path,
        model_artifact_path=model_artifact_path,
        shap_summary_png=shap_summary_png,
        shap_values_parquet=shap_values_parquet,
    )
    return (shap_summary_png, shap_values_parquet)
