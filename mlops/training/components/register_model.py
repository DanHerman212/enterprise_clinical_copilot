"""
register_model — publish the trained booster as a versioned SERVING BUNDLE in
GCS and record a provenance entry in the Vertex AI Model Registry.

Bundle-only handoff (the pipeline does NOT build or deploy a serving container):

    <artifact_uri>/
        model.bst        # native booster (booster.save_model)
        manifest.json    # feature_order + one-hot -> parent groups
        threshold.json   # operating threshold (decision layer only)
        gate_metrics.json  # gate metrics persisted with the bundle (ECC-71)

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

import hashlib
import json
import os
import re
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
    gate_metrics: dict | None = None,
) -> None:
    """Write every bundle file, then digest every bundle file.

    The writes and the digests are in one function on purpose: the digest list is
    the list of files written here, so a file cannot be added without being
    covered, and a name that cannot resolve cannot be listed. Both were possible
    when the digests were taken by a fixed list that skipped what was absent, and
    `gate_metrics.json` was written after the digests were taken.
    """
    os.makedirs(bundle_dir, exist_ok=True)
    written = []

    # The pre-built XGBoost container loads a model file named exactly model.bst.
    shutil.copy(booster_path, os.path.join(bundle_dir, "model.bst"))
    shutil.copy(manifest_path, os.path.join(bundle_dir, "manifest.json"))
    written += ["model.bst", "manifest.json"]

    if tuned_threshold is not None:
        with open(os.path.join(bundle_dir, "threshold.json"), "w") as f:
            json.dump(
                {
                    "threshold": float(tuned_threshold),
                    "beta": None if beta is None else float(beta),
                    "note": (
                        "Operating threshold for the decision layer only, chosen by "
                        "the calibrate-threshold step on out-of-fold train "
                        "probabilities to maximise F-beta; never baked into the "
                        "model, so it can change without retraining. Probabilities "
                        "are the booster's own logistic output on the probability "
                        "scale (objective=binary:logistic, scale_pos_weight fixed "
                        "at 1.0); no post-hoc calibration is applied. The "
                        "reliability curve in the evaluation report is a "
                        "diagnostic that no gate reads."
                    ),
                },
                f,
                indent=2,
            )
        written.append("threshold.json")

    # The numbers that say why this model was allowed to register, written before
    # the digests so they are covered by them. They are provenance for a reader of
    # the bundle — the registry labels carry the same values — but a bundle file
    # the endpoint does not verify is the one place a silent change could hide.
    if gate_metrics is not None:
        with open(os.path.join(bundle_dir, "gate_metrics.json"), "w") as f:
            json.dump(gate_metrics, f, indent=2)
        written.append("gate_metrics.json")

    # ECC-61: write a SHA-256 digest per bundle file so the serving predictor
    # can verify integrity at load time — a tampered or corrupted artifact
    # refuses to serve instead of shipping silently.
    checksums = {}
    for name in written:
        with open(os.path.join(bundle_dir, name), "rb") as fh:
            checksums[name] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(bundle_dir, "checksums.json"), "w") as f:
        json.dump(checksums, f, indent=2)


# Registry label values allow [a-z0-9_-] only, up to 63 characters.
_NOT_LABEL_SAFE = re.compile(r"[^a-z0-9_-]")
LABEL_MAX = 63


def _label(value: float) -> str:
    """A metric as a registry label value: the decimal point is encoded."""
    return f"{value:.4f}".replace(".", "-")


def _safe(value: str) -> str:
    """Free text as a registry label value: lowercased, sanitised, truncated."""
    return _NOT_LABEL_SAFE.sub("-", value.strip().lower())[:LABEL_MAX]


def provenance_labels(
    *,
    test_aucpr: float,
    tuned_threshold: float,
    pipeline_job_name: str = "",
    git_revision: str = "",
    data_row_count: int | None = None,
) -> dict[str, str]:
    """The labels a provenance entry carries.

    The run and the revision are the point of the entry. Without them a registry
    record says what was registered and how it scored, but not which run produced
    it or which code — and the code that reads the registry cannot ask a question
    the entry cannot answer. Both values are known here rather than reconstructed
    later: the pipeline job name is the run, and the revision is resolved once at
    submit time and passed down. An unknown value is left off the entry rather
    than written as a placeholder, so "no label" means "not recorded" and never
    gets mistaken for a value.

    Kept pure so the shape and the encoding are testable without a registry.
    """
    labels = {
        "pipeline": "readmission-training",
        "stage": "final",
        "test_aucpr": _label(test_aucpr),
        "tuned_threshold": _label(tuned_threshold),
    }
    if pipeline_job_name:
        labels["run"] = _safe(pipeline_job_name)
    if git_revision:
        labels["commit"] = _safe(git_revision)
    if data_row_count is not None:
        labels["data_rows"] = str(data_row_count)
    return labels


def provenance_description(
    *,
    test_aucpr: float,
    tuned_threshold: float,
    pipeline_job_name: str = "",
    git_revision: str = "",
    data_reference: dict | None = None,
) -> str:
    """The one line a person reads when they look the model up.

    Names the run, the code and the data behind the score, because those are the
    three questions asked of a registry entry when a model behaves unexpectedly.
    An unrecorded value says so rather than being left to look recorded.
    """
    parts = [
        f"Pipeline run {pipeline_job_name or 'unknown'}",
        f"revision {git_revision or 'unrecorded'}",
    ]
    if data_reference:
        rows = data_reference.get("row_count")
        parts.append(
            f"data {data_reference.get('table', 'unknown')} "
            f"({rows if rows is not None else '?'} rows, "
            f"as of {data_reference.get('last_modified_utc') or 'unknown'})"
        )
    else:
        parts.append("data reference not recorded")
    parts.append(f"test AUCPR {test_aucpr:.4f} at threshold {tuned_threshold:.4f}")
    return " · ".join(parts)


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
    pipeline_job_name: str = "",
    git_revision: str = "",
) -> str:
    """Assemble the bundle, record a provenance model entry, return its name."""
    from google.cloud import aiplatform

    serving_image = serving_container_image_uri or (
        f"{location}-docker.pkg.dev/{project_id}/readmission/readmission-cpr:latest"
    )

    # Persist the gate metrics WITH the bundle (ECC-71) — previously they were
    # only printed to the component log and lost with it. They are handed to
    # `assemble_serving_bundle` rather than written here so that the digests it
    # takes cover them: written after the digests, the file was skipped silently.
    gate_metrics = {
        "test_aucpr": float(test_aucpr),
        "hpo_val_aucpr": float(hpo_val_aucpr),
        "benchmark_aucpr": float(benchmark_aucpr),
        "tuned_threshold": float(tuned_threshold),
        "beta": float(beta),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    assemble_serving_bundle(
        booster_path=booster_path,
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
        tuned_threshold=tuned_threshold,
        beta=beta,
        gate_metrics=gate_metrics,
    )

    # The data reference is whatever load_data observed when it read the table,
    # read back out of the manifest it wrote rather than passed as another
    # parameter: the manifest is already in the bundle, already covered by the
    # checksums, and is the one place that records it.
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        data_reference = json.load(f).get("data")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-final-{ts}"

    aiplatform.init(project=project_id, location=location)
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=bundle_uri,
        serving_container_image_uri=serving_image,
        description=provenance_description(
            test_aucpr=test_aucpr,
            tuned_threshold=tuned_threshold,
            pipeline_job_name=pipeline_job_name,
            git_revision=git_revision,
            data_reference=data_reference,
        ),
        labels=provenance_labels(
            test_aucpr=test_aucpr,
            tuned_threshold=tuned_threshold,
            pipeline_job_name=pipeline_job_name,
            git_revision=git_revision,
            data_row_count=(data_reference or {}).get("row_count"),
        ),
    )
    model_name = model.resource_name

    print(f"  Registered: {model_name}")
    print(f"  Display:    {display_name}")
    print(f"  Run:        {pipeline_job_name or '<not passed>'}")
    print(f"  Revision:   {git_revision or '<not passed>'}")
    data_ref = data_reference or {}
    print(
        f"  Data:       {data_ref.get('table', '<not recorded>')} "
        f"({data_ref.get('row_count', '?')} rows, "
        f"as of {data_ref.get('last_modified_utc', '?')})"
    )
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
    pipeline_job_name: str = "",
    git_revision: str = "",
) -> NamedTuple("RegistryOutputs", [("model_id", str)]):
    """KFP component: publish the serving bundle + a CPR provenance record."""
    from mlops.training.components.register_model import run_register_model

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
        pipeline_job_name=pipeline_job_name,
        git_revision=git_revision,
    )
    return (model_id,)
