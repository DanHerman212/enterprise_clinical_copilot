"""Gap 12 — the image that serves is identified by its bytes, not by a name.

The CPR image tag is a content hash of the source, so it looks pinned and is not:
the build resolves apt and pip at build time, so the same tag can produce
different bytes, and a mutable `latest` tag was published beside it and was the
default image recorded on a registry entry. Meanwhile the deploy script resolved
the digest — the one immutable identifier of the bytes — and used it as a
boolean, so the deployment record named a tag and nothing recorded what ran.

These tests hold the three properties that replaced it: nothing publishes a
mutable tag, the deployment is uploaded by digest, and a registration that cannot
name its container by digest refuses instead of defaulting.
"""

import ast
import pathlib
import re
import subprocess

import pytest
import yaml

from mlops.serving import deploy_cpr
from mlops.serving.image_ref import LABEL_MAX, digest_label, require_serving_image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CLOUDBUILD = REPO_ROOT / "mlops/serving/cpr/cloudbuild.yaml"
DEPLOY = REPO_ROOT / "mlops/serving/deploy_cpr.py"
SUBMIT = REPO_ROOT / "mlops/training/submit_pipeline.sh"
REGISTER = REPO_ROOT / "mlops/training/components/register_model.py"
MANUAL = REPO_ROOT / "mlops/serving/register_serving_model.py"
IMAGE_REF = REPO_ROOT / "mlops/serving/image_ref.py"
LOCAL_HARNESS = REPO_ROOT / "mlops/serving/test_cpr_local.py"

DIGEST = "sha256:" + "01a4c7e2" * 8
DIGEST_REF = f"{deploy_cpr.IMAGE_REPO}@{DIGEST}"


def test_the_build_publishes_no_mutable_tag():
    """A `latest` beside the pinned tag is a deployable name for the same bytes."""
    config = yaml.safe_load(CLOUDBUILD.read_text())

    images = config.get("images", [])
    assert images, "the build no longer publishes an image at all"
    assert all(not str(i).endswith(":latest") for i in images), images

    build_args = [str(a) for step in config["steps"] for a in step["args"]]
    assert ":latest" not in " ".join(build_args), (
        "the docker build still tags a mutable name"
    )


def test_the_deploy_reference_is_a_digest_not_a_tag(monkeypatch):
    """`ensure_image` returns bytes, and the reference it returns names them."""
    monkeypatch.setattr(deploy_cpr, "image_tag", lambda: "adf35dd1d14a")
    monkeypatch.setattr(deploy_cpr, "resolve_digest", lambda image: DIGEST)

    image, digest = deploy_cpr.ensure_image()

    assert digest == DIGEST
    assert image == DIGEST_REF
    assert "@sha256:" in image


def test_an_unresolvable_digest_stops_the_deploy(monkeypatch):
    """A build whose bytes cannot be named must not become a deployment."""
    monkeypatch.setattr(deploy_cpr, "image_tag", lambda: "adf35dd1d14a")
    monkeypatch.setattr(deploy_cpr, "resolve_digest", lambda image: None)
    monkeypatch.setattr(deploy_cpr, "cloud_build", lambda tag: None)

    with pytest.raises(SystemExit):
        deploy_cpr.ensure_image()


def test_a_rebuild_is_forced_past_the_existing_image(monkeypatch):
    """--force-build must build, not reuse, and still return a digest."""
    built = []
    monkeypatch.setattr(deploy_cpr, "image_tag", lambda: "adf35dd1d14a")
    monkeypatch.setattr(deploy_cpr, "resolve_digest", lambda image: DIGEST)
    monkeypatch.setattr(deploy_cpr, "cloud_build", built.append)

    image, _ = deploy_cpr.ensure_image(force=True)

    assert built == ["adf35dd1d14a"]
    assert image == DIGEST_REF


def test_the_upload_uses_the_resolved_reference():
    """The digest has to reach the serving-container spec, not just be printed."""
    tree = ast.parse(DEPLOY.read_text())
    uploads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "upload"
    ]

    assert len(uploads) == 1, "expected exactly one Model.upload in the deploy path"
    argument = next(
        kw for kw in uploads[0].keywords
        if kw.arg == "serving_container_image_uri"
    )
    assert getattr(argument.value, "id", "") == "image", (
        "the uploaded image must be the digest reference returned by ensure_image"
    )

    source = DEPLOY.read_text()
    assert re.search(r"image,\s*digest = ensure_image\(", source)
    assert "image_digest" in source, "the digest is not recorded on the deployment"


def test_the_digest_label_is_registry_legal():
    """The label is truncated because a digest is 71 characters."""
    label = digest_label(DIGEST)

    assert re.fullmatch(r"[a-z0-9_-]{0,63}", label)
    assert len(label) == LABEL_MAX


@pytest.mark.parametrize("value", ["", "   ", "repo:latest", "repo:adf35dd1d14a"])
def test_a_registration_that_cannot_name_its_image_refuses(value):
    """A tag is a name that can move; the entry must record bytes."""
    with pytest.raises(ValueError):
        require_serving_image(value)


def test_a_digest_reference_is_accepted():
    assert require_serving_image(DIGEST_REF) == DIGEST_REF


def test_the_registration_records_no_default_image():
    """The `:latest` fallback is what made a mutable name a recorded artifact.

    Checked over string literals rather than the file's text, and only over
    image-shaped ones: the shared rule names `:latest` in prose, and prose cannot
    be deployed.
    """
    for path in (REGISTER, MANUAL, DEPLOY, IMAGE_REF, LOCAL_HARNESS):
        tree = ast.parse(path.read_text())
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        offenders = [
            value for value in literals
            if re.search(r"[A-Za-z0-9/_.\-]:latest", value)
        ]
        assert not offenders, f"{path.name} records a mutable image: {offenders}"

    # The build config is YAML, so it is read as text — its `:latest` would be a
    # tag on the command line, and the tag list is asserted separately above.
    build = CLOUDBUILD.read_text()
    assert not re.search(r"\$\{_IMAGE\}:latest", build)


def test_the_rule_has_one_definition():
    """Three call sites, one rule — a copy in each script is how they drift."""
    defining = [
        path for path in (REGISTER, MANUAL, DEPLOY, LOCAL_HARNESS)
        if "def require_serving_image" in path.read_text()
    ]

    assert defining == [], "require_serving_image is defined outside image_ref.py"
    assert "def require_serving_image" in IMAGE_REF.read_text()
    assert "from mlops.serving.image_ref import require_serving_image" in MANUAL.read_text()
    assert "require_serving_image" in REGISTER.read_text()


def test_the_submit_script_resolves_a_digest_instead_of_a_tag():
    """The pipeline cannot know the digest, so the submit step resolves it."""
    source = SUBMIT.read_text()

    assert "value(version)" in source, "the CPR digest is not resolved at submit time"
    assert "repo@sha256:" in source or '@${CPR_DIGEST}' in source
    assert "readmission-cpr:latest" not in source


def test_the_submit_script_fails_when_no_cpr_image_exists():
    """Absent image means absent reference, which must stop the submit."""
    body = SUBMIT.read_text()

    assert "no CPR serving image found" in body
    assert "deploy_cpr.py --build-only" in body, (
        "the refusal does not say how to satisfy it"
    )


def test_the_committed_ir_bakes_no_image_reference():
    """The image must travel as a submit-time value, not be compiled into the run."""
    ir = (REPO_ROOT / "mlops/training/readmission_training_pipeline.yaml").read_text()

    assert "readmission-cpr:latest" not in ir
    assert "@sha256:" not in ir, "a resolved digest was compiled into the pipeline"


def test_the_pipeline_parameter_still_exists_for_the_submit_script_to_set():
    """If the parameter disappears, the digest has nowhere to travel."""
    result = subprocess.run(
        ["git", "grep", "-n", "serving_container_image_uri",
         "--", "mlops/training/training_pipeline.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert "serving_container_image_uri" in result.stdout
