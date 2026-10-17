"""Upload evaluation artifacts to Cloud Storage.

Why this exists: the census writes `.jsonl` files full of clinical narrative, and a
local `results/` directory is both repository bloat waiting to happen and a PHI risk
if anyone ever force-adds it. The artifacts belong somewhere with a retention policy
and an access boundary — a bucket — rather than on one engineer's disk.

Shape of a run:

    gs://ecc-eval-artifacts/runs/{run_tag}/<filename>

`run_tag` is the operator's label for the run (`eval-pilot-101726`,
`eval-v3-101726`), so a later review can name the exact set of traces it is looking
at without trusting a filename.

Failure policy: an upload that fails logs and returns None. Losing a copy of the
artifact is bad; losing the run because the copy failed is worse — the local file is
still there and can be pushed by hand.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

BUCKET = "ecc-eval-artifacts"
PREFIX = "runs"


def destination(run_tag: str, local: Path) -> str:
    """The object URI for one artifact in one run."""
    tag = run_tag.strip("/") or "unlabelled"
    return f"gs://{BUCKET}/{PREFIX}/{tag}/{Path(local).name}"


def sync_eval_to_gcs(run_tag: str, files: list[Path]) -> list[str] | None:
    """Copy `files` into the run's prefix. Returns the URIs, or None on failure.

    Two transports, because this runs in two places. The client library is used when
    it is importable, which is the case in the job container where there is no gcloud
    CLI at all. `gcloud storage cp` remains the fallback for a laptop that has an
    authenticated CLI but never installed the library — that is how every report
    before the job existed was published, and breaking it to gain a container would
    make the container the only way to check the container.
    """
    present = [Path(f) for f in files if Path(f).is_file()]
    if not present:
        logger.warning("eval upload skipped: no artifacts to send for run %s", run_tag)
        return None

    try:
        upload = _client_upload() or _cli_upload()
    except Exception:  # noqa: BLE001 — a failed copy must not fail the run
        logger.warning("eval upload failed for run %s; artifacts remain local",
                       run_tag, exc_info=True)
        return None

    uris: list[str] = []
    for local in present:
        uri = destination(run_tag, local)
        try:
            upload(local, uri)
        except Exception:  # noqa: BLE001 — same reason, per file
            logger.warning("eval artifact failed: %s", uri, exc_info=True)
            return None
        uris.append(uri)
        logger.info("eval artifact uploaded: %s", uri)
    return uris


def _client_upload():
    """Upload through google-cloud-storage, or None when it is not installed."""
    try:
        from google.cloud import storage  # noqa: PLC0415 — optional transport
    except ImportError:
        return None

    def upload(local: Path, uri: str) -> None:
        client = storage.Client()
        bucket = client.bucket(BUCKET)
        key = uri.split(f"gs://{BUCKET}/", 1)[1]
        bucket.blob(key).upload_from_filename(str(local), timeout=300)

    return upload


def _cli_upload():
    """Upload through the gcloud CLI, the transport a laptop already has."""
    def upload(local: Path, uri: str) -> None:
        subprocess.run(
            ["gcloud", "storage", "cp", str(local), uri],
            check=True, capture_output=True, timeout=300,
        )

    return upload


def upload_if_asked(args, files: list[Path]) -> None:
    """The one line both CLIs call at exit, so the wiring cannot drift."""
    if not getattr(args, "upload_gcs", False):
        return
    tag = getattr(args, "run_tag", None) or "unlabelled"
    sync_eval_to_gcs(tag, files)
