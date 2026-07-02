"""
register_model — Assemble the serving bundle and register it in Vertex AI.

For real-time serving (Option 2), the model is served by a custom predictor
that reconstructs the exact training encoding. That predictor needs three files
together in one GCS directory (the ``artifact_uri`` the endpoint mounts):

    model.joblib     # fitted XGBClassifier
    imputer.joblib   # MissingnessImputer fit on train
    schema.json      # feature order + ordered category levels

This component copies those inputs into a single output directory and registers
that directory with the custom serving container image.
"""

import os
import shutil
from datetime import datetime, timezone
from typing import NamedTuple

from kfp import dsl
from ._image import TRAINING_IMAGE


def assemble_serving_bundle(
    *,
    model_path: str,
    imputer_path: str,
    schema_path: str,
    bundle_dir: str,
) -> None:
    """Copy the model, imputer, and schema into a serving-bundle directory."""
    os.makedirs(bundle_dir, exist_ok=True)
    shutil.copy(model_path, os.path.join(bundle_dir, "model.joblib"))
    shutil.copy(imputer_path, os.path.join(bundle_dir, "imputer.joblib"))
    shutil.copy(schema_path, os.path.join(bundle_dir, "schema.json"))


def run_register_model(
    *,
    project_id: str,
    location: str,
    model_path: str,
    imputer_path: str,
    schema_path: str,
    bundle_dir: str,
    bundle_uri: str,
    serving_container_image_uri: str,
    test_aucpr: float,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
) -> str:
    """Assemble the serving bundle, register the model, return resource name."""
    assemble_serving_bundle(
        model_path=model_path,
        imputer_path=imputer_path,
        schema_path=schema_path,
        bundle_dir=bundle_dir,
    )

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-final-{ts}"

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=bundle_uri,  # the DIRECTORY containing the bundle
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        labels={"pipeline": "readmission-training", "stage": "final"},
    )

    print(f"  Registered: {model.resource_name}")
    print(f"  Display:    {display_name}")
    print(f"  Bundle:     {bundle_uri}")
    print(f"  Serving:    {serving_container_image_uri}")
    print(f"  Test AUCPR:    {test_aucpr:.4f}")
    print(f"  HPO val AUCPR: {hpo_val_aucpr:.4f}")
    print(f"  Benchmark:     {benchmark_aucpr:.4f}")
    return model.resource_name


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-aiplatform"],
)
def register_model(
    project_id: str,
    model_artifact: dsl.Input[dsl.Model],
    imputer: dsl.Input[dsl.Artifact],
    schema: dsl.Input[dsl.Artifact],
    serving_container_image_uri: str,
    test_aucpr: float,
    hpo_val_aucpr: float,
    benchmark_aucpr: float,
    serving_model: dsl.Output[dsl.Model],
    location: str = "us-east1",
) -> NamedTuple("RegistryOutputs", [("model_id", str)]):
    """KFP component: assemble serving bundle and register in Vertex AI."""
    model_id = run_register_model(
        project_id=project_id,
        location=location,
        model_path=model_artifact.path,
        imputer_path=imputer.path,
        schema_path=schema.path,
        bundle_dir=serving_model.path,
        bundle_uri=serving_model.uri,
        serving_container_image_uri=serving_container_image_uri,
        test_aucpr=test_aucpr,
        hpo_val_aucpr=hpo_val_aucpr,
        benchmark_aucpr=benchmark_aucpr,
    )
    return (model_id,)
