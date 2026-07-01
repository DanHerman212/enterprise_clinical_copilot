"""
register_model — Upload trained model to Vertex AI Model Registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

from google.cloud import aiplatform
from kfp import dsl
from ._image import TRAINING_IMAGE


def run_register_model(
    *,
    project_id: str,
    model_artifact_uri: str,
    test_aucpr: float,
    final_val_aucpr: float,
    benchmark_aucpr: float,
) -> str:
    """Register model in Vertex AI Model Registry.  Returns resource name."""
    aiplatform.init(project=project_id, location="us-east1")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-final-{ts}"

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
        ),
        labels={
            "pipeline": "readmission-training",
            "stage": "final",
        },
    )

    print(f"  Registered: {model.resource_name}")
    print(f"  Display:    {display_name}")
    print(f"  Test AUCPR:      {test_aucpr:.4f}")
    print(f"  Final val AUCPR: {final_val_aucpr:.4f}")
    print(f"  Benchmark AUCPR: {benchmark_aucpr:.4f}")
    return model.resource_name


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=[],
)
def register_model(
    project_id: str,
    model_artifact_path: dsl.Input[dsl.Artifact],
    test_aucpr: float,
    final_val_aucpr: float,
    benchmark_aucpr: float,
) -> NamedTuple("RegistryOutputs", [("model_id", str)]):
    """KFP component: register final model in Vertex AI Model Registry."""
    model_id = run_register_model(
        project_id=project_id,
        model_artifact_uri=model_artifact_path.uri,
        test_aucpr=test_aucpr,
        final_val_aucpr=final_val_aucpr,
        benchmark_aucpr=benchmark_aucpr,
    )
    return (model_id,)
