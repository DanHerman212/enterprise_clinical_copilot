"""
register_model — register the native XGBoost booster in Vertex AI Model Registry
behind the **pre-built XGBoost serving container** with a Vertex Explainable AI
(Sampled Shapley) spec.

Decoupled serving pattern (zero custom container maintenance):

    <artifact_uri>/
        model.bst                    # native booster (booster.save_model)
        manifest.json                # feature_order + one-hot -> parent groups
        explanation_metadata.json    # Sampled Shapley input/output mapping
        threshold.json               # operating threshold (decision layer only)

All feature encoding is static in BigQuery (analytics_dataset_encoded), so the
model consumes a fixed-order numeric vector. The endpoint returns a probability
(objective=binary:logistic); Sampled Shapley attributes that probability per
one-hot column, and the agent-side serving glue aggregates attributions back to
parent features using ``manifest.json`` groups (valid by Shapley additivity).
"""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import NamedTuple

from kfp import dsl
from ._image import TRAINING_IMAGE, component

# Vertex AI pre-built XGBoost prediction container (CPU). Region-matched at
# submit time via the pipeline parameter; this is the default.
DEFAULT_PREBUILT_XGB_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
)

# Sampled Shapley path count (approximation budget). Higher = lower variance,
# higher latency. 25 is a reasonable interactive-demo default.
DEFAULT_PATH_COUNT = 25


def assemble_serving_bundle(
    *,
    booster_path: str,
    manifest_path: str,
    bundle_dir: str,
    path_count: int,
    tuned_threshold: float | None = None,
    beta: float | None = None,
) -> dict:
    """Copy model.bst + manifest into the bundle dir and write XAI metadata.

    Returns the explanation_metadata dict (also persisted alongside the model).
    """
    os.makedirs(bundle_dir, exist_ok=True)

    # The pre-built XGBoost container loads a model file named exactly model.bst.
    shutil.copy(booster_path, os.path.join(bundle_dir, "model.bst"))
    shutil.copy(manifest_path, os.path.join(bundle_dir, "manifest.json"))

    with open(manifest_path) as f:
        manifest = json.load(f)
    feature_order = manifest["feature_order"]

    # Sampled Shapley over the fixed-order numeric vector. BAG_OF_FEATURES +
    # index_feature_mapping gives per-column attributions keyed by name; the
    # agent tool then sums each one-hot group back to its parent feature.
    # (Tensor names are subject to Phase-4 endpoint verification.)
    explanation_metadata = {
        "inputs": {
            "features": {
                "input_tensor_name": "features",
                "encoding": "BAG_OF_FEATURES",
                "index_feature_mapping": feature_order,
            }
        },
        "outputs": {"probability": {"output_tensor_name": "probability"}},
    }
    with open(os.path.join(bundle_dir, "explanation_metadata.json"), "w") as f:
        json.dump(explanation_metadata, f, indent=2)

    if tuned_threshold is not None:
        with open(os.path.join(bundle_dir, "threshold.json"), "w") as f:
            json.dump(
                {
                    "threshold": float(tuned_threshold),
                    "beta": None if beta is None else float(beta),
                    "path_count": int(path_count),
                    "note": (
                        "Operating threshold for the decision layer only; the "
                        "endpoint returns calibrated probabilities."
                    ),
                },
                f,
                indent=2,
            )

    return explanation_metadata


def run_register_model(
    *,
    project_id: str,
    location: str,
    booster_path: str,
    manifest_path: str,
    bundle_dir: str,
    bundle_uri: str,
    serving_container_image_uri: str,
    test_aucpr: float,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    tuned_threshold: float,
    beta: float = 2.0,
    path_count: int = DEFAULT_PATH_COUNT,
) -> str:
    """Assemble the bundle, register with the pre-built container + XAI, return name."""
    prebuilt_image = serving_container_image_uri or DEFAULT_PREBUILT_XGB_IMAGE

    explanation_metadata = assemble_serving_bundle(
        booster_path=booster_path,
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
        path_count=path_count,
        tuned_threshold=tuned_threshold,
        beta=beta,
    )

    from google.cloud import aiplatform
    from google.cloud.aiplatform.explain import (
        ExplanationMetadata,
        ExplanationParameters,
    )

    aiplatform.init(project=project_id, location=location)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-final-{ts}"

    xai_metadata = ExplanationMetadata(
        inputs=explanation_metadata["inputs"],
        outputs=explanation_metadata["outputs"],
    )
    xai_parameters = ExplanationParameters(
        {"sampled_shapley_attribution": {"path_count": path_count}}
    )

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=bundle_uri,  # the DIRECTORY containing model.bst
        serving_container_image_uri=prebuilt_image,
        explanation_metadata=xai_metadata,
        explanation_parameters=xai_parameters,
        labels={"pipeline": "readmission-training", "stage": "final"},
    )

    print(f"  Registered: {model.resource_name}")
    print(f"  Display:    {display_name}")
    print(f"  Bundle:     {bundle_uri}")
    print(f"  Serving:    {prebuilt_image}  (pre-built XGBoost + Sampled Shapley)")
    print(f"  Path count: {path_count}")
    print(f"  Test AUCPR:    {test_aucpr:.4f}")
    print(f"  HPO val AUCPR: {hpo_val_aucpr:.4f}")
    print(f"  Benchmark:     {benchmark_aucpr:.4f}")
    print(f"  Threshold:     {tuned_threshold:.4f}  (F{beta:g}, in threshold.json)")
    return model.resource_name


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-aiplatform"],
)
def register_model(
    project_id: str,
    booster_model: dsl.Input[dsl.Model],
    manifest: dsl.Input[dsl.Artifact],
    serving_container_image_uri: str,
    test_aucpr: float,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    tuned_threshold: float,
    serving_model: dsl.Output[dsl.Model],
    location: str = "us-east1",
    beta: float = 2.0,
    path_count: int = DEFAULT_PATH_COUNT,
) -> NamedTuple("RegistryOutputs", [("model_id", str)]):
    """KFP component: register the booster with the pre-built container + XAI."""
    from pipelines.components.register_model import run_register_model

    model_id = run_register_model(
        project_id=project_id,
        location=location,
        booster_path=booster_model.path,
        manifest_path=manifest.path,
        bundle_dir=serving_model.path,
        bundle_uri=serving_model.uri,
        serving_container_image_uri=serving_container_image_uri,
        test_aucpr=test_aucpr,
        hpo_val_aucpr=hpo_val_aucpr,
        benchmark_aucpr=benchmark_aucpr,
        tuned_threshold=tuned_threshold,
        beta=beta,
        path_count=path_count,
    )
    return (model_id,)
