"""
shap_explain — Tree SHAP global importance and per-patient attributions.

Runs in parallel with evaluate_test after train-final.
"""

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
    shap_beeswarm_png: str,
    shap_waterfall_png: str,
    shap_local_plots_dir: str,
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

    import xgboost as xgb
    X_matrix = xgb.DMatrix(X_sample, enable_categorical=True)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_matrix)
    explanation = explainer(X_matrix)
    explanation.feature_names = X_sample.columns.tolist()

    # --- Global importance plot (Bar) ---
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.35)))
    shap.summary_plot(
        shap_values, X_sample, plot_type="bar", max_display=top_n, show=False,
    )
    ax.set_title("SHAP Feature Importance (Bar)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(shap_summary_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Global interpretability plot (Beeswarm) ---
    # This matches the user's provided example image.
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.35)))
    shap.summary_plot(
        shap_values, X_sample, plot_type="dot", max_display=top_n, show=False,
    )
    ax.set_title("SHAP Feature Importance (Beeswarm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(shap_beeswarm_png, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Local interpretability plots (Waterfall) ---
    import os
    os.makedirs(shap_local_plots_dir, exist_ok=True)
    
    # We select 3 different samples: min, median, and max predicted risk
    preds = booster.predict(X_matrix)
    indices_to_plot = [
        np.argmin(preds),             # lowest risk
        np.argsort(preds)[len(preds)//2],  # median risk
        np.argmax(preds)              # highest risk
    ]
    labels = ["lowest_risk", "median_risk", "highest_risk"]
    
    for idx, label in zip(indices_to_plot, labels):
        # We need an Explanation object for the waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation[idx], max_display=top_n, show=False)
        plt.title(f"Local SHAP Explanation ({label}: pred={preds[idx]:.4f})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{shap_local_plots_dir}/shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        
    # Also save one representative waterfall plot to output artifact
    import shutil
    shutil.copy(f"{shap_local_plots_dir}/shap_waterfall_highest_risk.png", shap_waterfall_png)

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
    shap_markdown: dsl.Output[dsl.Markdown],
) -> NamedTuple(
    "SHAPOutputs",
    [
        ("shap_summary_png", str), 
        ("shap_beeswarm_png", str),
        ("shap_waterfall_png", str),
        ("shap_local_plots_dir", str),
        ("shap_values_parquet", str)
    ],
):
    """KFP component: Tree SHAP global importance + per-patient values."""
    from pipelines.components.shap_explain import run_shap_explain

    import os
    
    # We will use temporary paths for the PNGs, then render them into the Markdown artifact.
    # Vertex UI does not render raw .png string paths intuitively.
    temp_dir = os.path.dirname(shap_markdown.path)
    shap_summary_png = os.path.join(temp_dir, "shap_summary.png")
    shap_beeswarm_png = os.path.join(temp_dir, "shap_beeswarm.png")
    shap_waterfall_png = os.path.join(temp_dir, "shap_waterfall.png")
    shap_local_plots_dir = os.path.join(temp_dir, "shap_local_plots")
    shap_values_parquet = os.path.join(temp_dir, "shap_values.parquet")

    run_shap_explain(
        x_test_path=x_test_path,
        model_artifact_path=model_artifact_path,
        shap_summary_png=shap_summary_png,
        shap_beeswarm_png=shap_beeswarm_png,
        shap_waterfall_png=shap_waterfall_png,
        shap_local_plots_dir=shap_local_plots_dir,
        shap_values_parquet=shap_values_parquet,
    )

    # Encode images to base64 so they render directly inside the Vertex Markdown UI widget
    import base64
    def _to_b64(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    md_content = f"""
# Model Interpretability (SHAP)

## Global Importance (Beeswarm)
<img src="data:image/png;base64,{_to_b64(shap_beeswarm_png)}" width="800"/>

## Global Importance (Bar)
<img src="data:image/png;base64,{_to_b64(shap_summary_png)}" width="800"/>

## Local Importance (Waterfall - Highest Risk)
<img src="data:image/png;base64,{_to_b64(shap_waterfall_png)}" width="800"/>
    """
    
    with open(shap_markdown.path, "w") as f:
        f.write(md_content)

    return (
        shap_summary_png, 
        shap_beeswarm_png,
        shap_waterfall_png,
        shap_local_plots_dir,
        shap_values_parquet
    )
