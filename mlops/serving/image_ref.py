"""Image references that name bytes rather than names.

One rule with three call sites — the training registry entry, the manual
registry entry, and the CPR deployment — so it lives here rather than in each:
any image recorded on an artifact is recorded as `repo@sha256:...`.

Why a tag is not enough. A tag can be re-pushed, and then an artifact names bytes
that never trained; the CPR image is tagged with a content hash of its source,
which makes it look pinned, but the build resolves apt and pip at build time, so
the same tag can produce different bytes. A digest is the only reference that
says which bytes exist, and it is what Artifact Registry already returns
(`gcloud artifacts docker images describe <image>:<tag>
--format=value(image_summary.digest)`).
"""

LABEL_MAX = 63


def require_serving_image(explicit: str) -> str:
    """The serving container an artifact records, or a refusal.

    Raises rather than writing a placeholder: an entry that names an image it
    cannot identify is worse than a run that stops.
    """
    value = explicit.strip()
    if not value:
        raise ValueError(
            "a serving container image is required: a registry entry or a "
            "deployment must name it by immutable reference (repo@sha256:...), and "
            "there is deliberately no `:latest` default. For the training pipeline "
            "submit_pipeline.sh resolves the CPR digest; the CPR image can be built "
            "with `python mlops/serving/deploy_cpr.py --build-only`."
        )
    if "@sha256:" not in value:
        raise ValueError(
            f"a serving container image must be an immutable reference "
            f"(repo@sha256:...), not a tag: {value!r}. Resolve the digest with "
            f"`gcloud artifacts docker images describe <image>:<tag> "
            f"--format=value(image_summary.digest)`."
        )
    return value


def digest_label(digest: str) -> str:
    """A registry-legal label for an image digest ([a-z0-9_-], max 63 chars).

    Truncated, because a digest is 71 characters. The full value belongs in the
    description, which has no such limit; the label is for filtering, and the same
    truncation is what a filter must use.
    """
    return digest.replace(":", "-").lower()[:LABEL_MAX]
