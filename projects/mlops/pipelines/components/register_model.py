"""
register_model — publish the trained booster as a versioned SERVING BUNDLE in
GCS and record a provenance entry in the Vertex AI Model Registry.

Bundle-only handoff (the pipeline does NOT build or deploy a serving container):

    <artifact_uri>/
        model.bst        # native booster (booster.save_model)
        manifest.json    # feature_order + one-hot -> parent groups
        threshold.json   # operating threshold (decision layer only)

The registry entry (display name ``readmission-final-<ts>``) points at this
bundle via ``artifact_uri`` and tags the CPR serving image for provenance. The
servable model is built and deployed separately by ``scripts/deploy_cpr.py``,
which discovers the latest bundle from this record's ``artifact_uri``, wraps it
in the Custom Prediction Routine container (probability + native TreeSHAP
attributions in one response), and deploys it to the Vertex endpoint.

All feature encoding is static in BigQuery (analytics_dataset_encoded), so the
model consumes a fixed-order numeric vector and returns a calibrated probability
(objective=binary:logistic).
"""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import NamedTuple

from kfp import dsl
from ._image import TRAINING_IMAGE, component

# The Custom Prediction Routine (CPR) serving image is recorded on the
# provenance entry so lineage points at the correct serving image; the actual
# servable model (with the CPR container spec) is built + deployed by
# scripts/deploy_cpr.py, not by this pipeline component. When no explicit URI
# is passed, it is derived from the run's own project/location — no project id
# is baked into the source.


def assemble_serving_bundle(
    *,
    booster_path: str,
    manifest_path: str,
    bundle_dir: str,
    tuned_threshold: float | None = None,
    beta: float | None = None,
) -> None:
    """Copy model.bst + manifest into the bundle dir and write threshold.json."""
    os.makedirs(bundle_dir, exist_ok=True)

    # The pre-built XGBoost container loads a model file named exactly model.bst.
    shutil.copy(booster_path, os.path.join(bundle_dir, "model.bst"))
    shutil.copy(manifest_path, os.path.join(bundle_dir, "manifest.json"))

    if tuned_threshold is not None:
        with open(os.path.join(bundle_dir, "threshold.json"), "w") as f:
            json.dump(
                {
                    "threshold": float(tuned_threshold),
                    "beta": None if beta is None else float(beta),
                    "note": (
                        "Operating threshold for the decision layer only; the "
                        "endpoint returns calibrated probabilities."
                    ),
                },
                f,
                indent=2,
            )


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
) -> str:
    """Assemble the bundle, record a provenance model entry, return its name."""
    from google.cloud import aiplatform

    serving_image = serving_container_image_uri or (
        f"{location}-docker.pkg.dev/{project_id}/readmission/readmission-cpr:latest"
    )

    assemble_serving_bundle(
        booster_path=booster_path,
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
        tuned_threshold=tuned_threshold,
        beta=beta,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-final-{ts}"

    aiplatform.init(project=project_id, location=location)
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=bundle_uri,
        serving_container_image_uri=serving_image,
        labels={"pipeline": "readmission-training", "stage": "final"},
    )
    model_name = model.resource_name

    print(f"  Registered: {model_name}")
    print(f"  Display:    {display_name}")
    print(f"  Bundle:     {bundle_uri}")
    print(f"  Serving:    {serving_image}  (CPR provenance; deployed by deploy_cpr.py)")
    print(f"  Test AUCPR:    {test_aucpr:.4f}")
    print(f"  HPO val AUCPR: {hpo_val_aucpr:.4f}")
    print(f"  Benchmark:     {benchmark_aucpr:.4f}")
    print(f"  Threshold:     {tuned_threshold:.4f}  (F{beta:g}, in threshold.json)")
    return model_name


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
) -> NamedTuple("RegistryOutputs", [("model_id", str)]):
    """KFP component: publish the serving bundle + a CPR provenance record."""
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
    )
    return (model_id,)
