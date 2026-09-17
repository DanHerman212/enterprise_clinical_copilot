"""
deploy_cpr.py — Register and deploy the readmission Custom Prediction Routine
(CPR) to a Vertex AI endpoint.

This is the only way a model reaches the endpoint: `scripts/agent/teardown.py`
removes the billable resources, and nothing else deploys.

A deployment is not finished when the traffic shifts — it is finished when the
model answers. After the shift this script asks the endpoint about one instance
and checks that the answer came from the deployment it just made; if it did not,
the traffic is put back where it was and the script exits non-zero. That check is
why the previous deployment is not retired until the last step.

The CPR serving image is built on Cloud Build (native linux/amd64) and is
content-addressed: it is rebuilt only when the CPR source (Dockerfile,
predictor.py, requirements.txt) changes. A newly trained model reuses the same
image — only the serving bundle (artifact_uri) changes.

The endpoint returns probability + threshold decision + native-TreeSHAP
attributions in a single response, while keeping Vertex's traffic control and
model monitoring.

Usage (from repo root):
    .venv/bin/python mlops/serving/deploy_cpr.py [--build-only] [--force-build]

Env:
    BUNDLE_URI     — GCS dir with model.bst + manifest.json [+ threshold.json]
    IMAGE_URI      — override the output image repo (tag is ignored/recomputed)
    ENDPOINT_NAME  — Vertex endpoint display name (default: readmission-endpoint)
    MACHINE_TYPE   — default: n1-standard-2
"""

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlops.data.config import get_project_id  # noqa: E402
from mlops.serving.deploy_check import (  # noqa: E402
    verify_or_rollback,
)
from mlops.serving.image_ref import digest_label  # noqa: E402

PROJECT = get_project_id()
LOCATION = "us-east1"
REPO = "readmission"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "readmission-endpoint")
MACHINE_TYPE = os.environ.get("MACHINE_TYPE", "n1-standard-2")

CPR_SRC = Path(__file__).resolve().parents[1] / "serving" / "cpr"
CLOUDBUILD = CPR_SRC / "cloudbuild.yaml"

# Image repo (no tag). The concrete tag is a content hash of the CPR source, so
# the image is reused across model versions and only rebuilt when it changes.
IMAGE_REPO = os.environ.get(
    "IMAGE_URI",
    f"{LOCATION}-docker.pkg.dev/{PROJECT}/{REPO}/readmission-cpr",
).split(":")[0]

# CPR serving-container contract (matches LocalModel.build_cpr_model output).
PREDICT_ROUTE = "/predict"
HEALTH_ROUTE = "/health"
CONTAINER_PORT = 8080

# Files whose contents determine the image tag (rebuild only when they change).
HASH_INPUTS = ["Dockerfile", "predictor.py", "requirements.txt"]

# The serving bundle is discovered from the latest pipeline provenance record
# (readmission-final-*). Set BUNDLE_URI to override (e.g. to pin a specific run).
BUNDLE_URI_OVERRIDE = os.environ.get("BUNDLE_URI")
FINAL_MODEL_PREFIX = "readmission-final-"
# The label the training pipeline writes alongside that name. Selection requires
# both, and that is the rule rather than a convenience: a record carrying the name
# without the label was not written by a pipeline run. The manual registration
# path now publishes readmission-manual-* with stage=manual, so a hand-made record
# cannot take the newest position without training anything.
PIPELINE_STAGE = "final"


def score_of(labels: dict) -> str:
    """The test AUCPR and threshold a record carries, in readable form.

    Registry label values allow [a-z0-9_-] only, so the training registration
    encodes the decimal point as a dash (0.3294 -> "0-3294").
    """
    def decode(value: str) -> str:
        # "0-1100" is 0.1100 as the registrar wrote it; str() drops the padding.
        return str(float(value.replace("-", ".", 1))) if value else "?"

    return (f"test AUCPR {decode(labels.get('test_aucpr', ''))}, "
            f"threshold {decode(labels.get('tuned_threshold', ''))}")


def select_serving_record(models: list) -> object | None:
    """The record that serves, from a list ordered newest-first.

    The rule: the model that serves is the bundle from the most recent successful
    pipeline run. A record qualifies only if the training pipeline registered it —
    the final name prefix *and* the stage label `register_model.py` writes with
    it. Creation order then picks between qualifying runs; the metric labels say
    which one was better when that matters, and the score is printed so a human
    can see it before the traffic moves.

    Kept pure, and separate from the query, so the rule is testable without a
    registry.
    """
    for m in models:
        if (m.display_name.startswith(FINAL_MODEL_PREFIX)
                and (m.labels or {}).get("stage") == PIPELINE_STAGE):
            return m
    return None


def image_tag() -> str:
    """Return a stable content hash over the CPR source that defines the image."""
    h = hashlib.sha256()
    for name in HASH_INPUTS:
        h.update(name.encode())
        h.update((CPR_SRC / name).read_bytes())
    return h.hexdigest()[:12]


def resolve_digest(image: str) -> str | None:
    """The immutable digest of a tagged image, or None if it is not published.

    A tag says what a build intended to publish; a digest says which bytes exist.
    This is the value that makes "the image that served" a fact rather than a
    name, and it used to be fetched here and thrown away: the command already
    read it, and the function returned a boolean.
    """
    r = subprocess.run(
        ["gcloud", "artifacts", "docker", "images", "describe", image,
         "--format=value(image_summary.digest)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def cloud_build(tag: str) -> None:
    """Build & push the CPR image on Cloud Build (linux/amd64), tagged by content."""
    print(f"Cloud Build: {IMAGE_REPO}:{tag}")
    subprocess.run(
        ["gcloud", "builds", "submit", str(CPR_SRC),
         "--project", PROJECT,
         "--config", str(CLOUDBUILD),
         "--substitutions", f"_IMAGE={IMAGE_REPO},_TAG={tag}"],
        check=True,
    )


def ensure_image(force: bool = False) -> tuple[str, str]:
    """Return (immutable image reference, digest) for the CPR image.

    The reference is `repo@sha256:...`, not `repo:tag`. The tag is a content hash
    of the CPR source, so it is stable across builds of the same source — but the
    build resolves apt and pip at build time, so the same tag can produce
    different bytes, and a reference by tag is a promise about the past rather
    than a record of it. Deploying by digest also puts the identity of the bytes
    into the serving-container spec, which is what the registry keeps.
    """
    tag = image_tag()
    tagged = f"{IMAGE_REPO}:{tag}"
    digest = None if force else resolve_digest(tagged)
    if digest is None:
        cloud_build(tag)
        digest = resolve_digest(tagged)
        if digest is None:
            raise SystemExit(
                f"ERROR: {tagged} was built but no digest could be read back "
                "from Artifact Registry, so the bytes that would serve cannot be "
                "named. Refusing to deploy."
            )
    else:
        print(f"CPR image up-to-date, reusing: {tagged} ({digest})")
    return f"{IMAGE_REPO}@{digest}", digest


def discover_bundle() -> tuple[str, str]:
    """(artifact_uri, display name) of the newest pipeline-registered version.

    States what it resolved and what that record scored, because "newest" and
    "better" are different claims and only a person can weigh the second.
    """
    models = list(aiplatform.Model.list(order_by="create_time desc"))
    serving = select_serving_record(models)
    if serving is None:
        raise SystemExit(
            f"No pipeline-registered '{FINAL_MODEL_PREFIX}*' model found "
            f"(stage={PIPELINE_STAGE}); run the training pipeline first or set "
            "BUNDLE_URI explicitly."
        )
    uri = serving.gca_resource.artifact_uri.rstrip("/")
    print(f"Discovered bundle from {serving.display_name}: {uri}")
    print(f"  {score_of(serving.labels or {})}")
    previous = next(
        (m for m in models
         if m.display_name != serving.display_name
         and m.display_name.startswith(FINAL_MODEL_PREFIX)
         and (m.labels or {}).get("stage") == PIPELINE_STAGE),
        None,
    )
    if previous is not None:
        print(f"  previous registration: {previous.display_name} "
              f"({score_of(previous.labels or {})})")
    return uri, serving.display_name


def resolve_bundle() -> tuple[str, str]:
    """(artifact_uri, model version) from BUNDLE_URI if set, else the registry.

    The version travels into the container as MODEL_VERSION, so a prediction can
    say which model produced it without asking the registry — the registry's
    newest record and the deployed model are two different things.
    """
    if BUNDLE_URI_OVERRIDE:
        uri = BUNDLE_URI_OVERRIDE.rstrip("/")
        return uri, os.path.basename(uri) or uri
    return discover_bundle()


def probe_instance() -> dict[str, None]:
    """One instance to ask the endpoint about, keyed by feature name.

    Every value is null, which the CPR reads as a missing measurement by design,
    so this needs no patient and no BigQuery — the question is whether the model
    we deployed runs and answers as itself. The names are the code-owned feature
    contract, which doubles as a check: a deployed bundle whose manifest
    disagrees with it refuses the request, and the deploy fails here instead of
    during a demo.
    """
    from mlops.data.encoding import feature_order

    return {name: None for name in feature_order()}


def _try_probe(ep: "aiplatform.Endpoint", instance: dict) -> float | None:
    """Best-effort prediction from whatever is serving now, for comparison."""
    try:
        response = ep.predict(instances=[instance])
        return float(response.predictions[0]["probability"])
    except Exception as exc:
        print(f"    (could not ask the current deployment for a baseline: "
              f"{type(exc).__name__})")
        return None


def main() -> None:
    aiplatform.init(project=PROJECT, location=LOCATION)
    image, digest = ensure_image(force="--force-build" in sys.argv)
    if "--build-only" in sys.argv:
        print(f"Build-only: {image}")
        return

    bundle_uri, model_version = resolve_bundle()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    display_name = f"readmission-cpr-{ts}"
    print(f"Registering model: {display_name}")
    print(f"  model_version stamped for serving: {model_version}")
    model = aiplatform.Model.upload(
        display_name=display_name,
        # By digest, so this record identifies the bytes that serve. A tag here
        # would be a mutable name in the one place that is supposed to say what ran.
        serving_container_image_uri=image,
        serving_container_predict_route=PREDICT_ROUTE,
        serving_container_health_route=HEALTH_ROUTE,
        serving_container_ports=[CONTAINER_PORT],
        # The CPR predictor downloads the bundle via storage.Client(), which
        # needs a project. Pin it explicitly so worker boot never depends on
        # ambient metadata-server project resolution (the "Model server never
        # became ready" failure was storage.Client() unable to resolve one).
        # MODEL_VERSION is the provenance record this deployment came from: the
        # endpoint returns it with every prediction, so an answer can name the
        # model that produced it without a registry lookup.
        serving_container_environment_variables={
            "GOOGLE_CLOUD_PROJECT": PROJECT,
            "MODEL_VERSION": model_version,
        },
        artifact_uri=bundle_uri,
        description=(
            f"CPR serving image {image} · bundle {bundle_uri} · "
            f"model version {model_version}"
        ),
        labels={
            "pipeline": "readmission-training",
            "stage": "cpr",
            # The digest again, registry-legal and truncated; the full value is in
            # the description. Both are read from the record, not from Artifact
            # Registry, which may have moved on.
            "image_digest": digest_label(digest),
        },
    )
    print(f"Registered: {model.resource_name}")

    # Reuse or create the endpoint.
    endpoints = [
        ep for ep in aiplatform.Endpoint.list(order_by="create_time desc")
        if ep.display_name == ENDPOINT_NAME
    ]
    ep = endpoints[0] if endpoints else aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
    print(f"Endpoint: {ep.resource_name}")

    stale = list(ep.list_models())
    if stale:
        print(f"  Existing deployments: {[dm.id for dm in stale]}")
    # What is serving now, so a failed verification can put it back exactly.
    serving_split = dict(ep.gca_resource.traffic_split or {})
    before_probability = None

    # Deploy the new model at 0% traffic FIRST (ECC-47): the old deployment
    # keeps serving, so there is no outage window, and a failed deploy leaves
    # the previous model intact — rollback is simply "stop here".
    print("Deploying CPR model at 0% traffic (5–10 min) …")
    try:
        model.deploy(
            endpoint=ep,
            deployed_model_display_name=display_name,
            machine_type=MACHINE_TYPE,
            min_replica_count=1,
            max_replica_count=1,
            traffic_percentage=0,
        )
        new_dm_id = next(
            dm.id for dm in ep.list_models() if dm.display_name == display_name
        )
    except Exception:
        print("Deploy failed — the previous deployment is still serving 100% "
              "traffic (rollback = no-op).")
        raise

    instance = probe_instance()
    if serving_split:
        # A baseline for the same question, answered by the model that serves
        # now. Best effort: the point of the comparison is information for the
        # operator, not a gate.
        before_probability = _try_probe(ep, instance)

    print(f"Shifting traffic to {new_dm_id} …")
    ep.update(traffic_split={new_dm_id: 100})

    # A deployment is not finished until the model answers. This is the only
    # step that can tell a working deploy from one that loads and serves
    # nothing, and it is why the previous deployment is not retired until now.
    probability = verify_or_rollback(
        ep,
        instance=instance,
        new_dm_id=new_dm_id,
        expected_version=model_version,
        serving_split=serving_split,
    )

    delta = ""
    if before_probability is not None:
        delta = f" (the previous deployment answered {before_probability:.4f})"
    print(f"  Verified: probability {probability:.4f}{delta}")

    for dm in ep.list_models():
        if dm.id != new_dm_id:
            print(f"  Undeploying stale model {dm.id} …")
            ep.undeploy(deployed_model_id=dm.id)
    print(f"Deployed and verified. Endpoint: {ep.resource_name}")


if __name__ == "__main__":
    main()
