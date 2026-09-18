"""recall_gate — measure a deployed index and decide whether it may go live.

A promotion is only defensible if the measurement behind it describes the
artifact being promoted. That is the whole point of this module, and it is why
discovery is a join rather than a lookup:

  * ``run_for_index`` finds the ingest run that built a given index by reading
    each recent run's ``build-index`` manifest and comparing the tree-AH index
    it records against the index under consideration. "The newest successful
    run" would not do: a report about the previous index authorises the next one
    just as convincingly and for no reason.
  * ``report_for`` runs the recall@k job against one deployed id — the staging
    id, while the live id keeps serving — and reads the report it writes.
  * ``judge`` folds that report through the configured thresholds, and
    ``publish`` writes the evidence to the bucket next to the recall reports, so
    the verdict a promotion rests on outlives the shell that produced it.

The recall job itself needs the chunk artifact and the full ingest for the
corpus, both of which are pipeline outputs; nothing here is hardcoded because a
hardcoded artifact path stops describing the thing it names the first time the
pipeline runs again.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from google.api_core.exceptions import NotFound
from google.cloud import aiplatform, storage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.mcp.config import LOCATION, PROJECT  # noqa: E402
from services.mcp.retrieval.config import load as load_config  # noqa: E402
from services.mcp.retrieval.eval_report import write_artifacts  # noqa: E402
from services.mcp.retrieval.gate import recall_result  # noqa: E402

INGEST_PIPELINE = "rag-ingest"
BUILD_TASK = "build-index"
CHUNK_TASK = "chunk-notes"
EMBED_TASK = "embed-chunks"

RECALL_MACHINE = os.environ.get("RECALL_MACHINE_TYPE", "e2-standard-8")
SERVICE_ACCOUNT = os.environ.get(
    "PIPELINE_SA", f"mlops-pipeline@{PROJECT}.iam.gserviceaccount.com")
IMAGE = os.environ.get(
    "RAG_IMAGE_URI",
    f"{LOCATION}-docker.pkg.dev/{PROJECT}/readmission/rag-ingest:latest",
)

# A deployed index carries the measurement that authorised it in its display
# name, because that is the one field of a deployment a person reads.
LABEL_MAX = 128
REPORT_NAME = "recall_report.json"
RESULTS_NAME = "eval_results.json"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def out_dir(corpus: str, index_display: str, stamp: str = "") -> str:
    """Where one measurement's evidence lives, keyed by corpus and by index."""
    return (f"gs://{PROJECT}-mlops/rag/recall/{corpus}/{index_display}/"
            f"{stamp or _stamp()}")


def _artifact_uris(job) -> dict[str, dict[str, str]]:
    """task name -> {output name: uri} for one pipeline run."""
    found: dict[str, dict[str, str]] = {}
    for task in job.task_details or []:
        name = getattr(task, "task_name", "") or ""
        for output, entry in (getattr(task, "outputs", {}) or {}).items():
            uris = [a.uri for a in getattr(entry, "artifacts", [])]
            if uris:
                found.setdefault(name, {})[output] = uris[0]
    return found


def _read_manifest(uri: str) -> dict:
    """The manifest object at a URI, or an empty mapping if it is unreadable."""
    bucket, obj = uri.replace("gs://", "", 1).split("/", 1)
    blob = storage.Client().bucket(bucket).blob(obj)
    if not blob.exists():
        return {}
    try:
        return json.loads(blob.download_as_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _ingest_runs(**kwargs):
    """Successful ingest runs, newest first, in the project this repo targets.

    The project and location are passed explicitly: listing without them reads
    whichever project the environment happens to name, which returned nothing.
    """
    return aiplatform.PipelineJob.list(
        project=PROJECT, location=LOCATION, **kwargs)


def run_for_index(index_resource_name: str, *, jobs=None,
                  manifest_reader=None, limit: int = 10) -> dict:
    """The ingest run that built ``index_resource_name``, or refuse to proceed.

    Raises rather than falling back to the newest run: a measurement whose
    provenance is a guess is worse than no measurement, because it produces a
    promotion that looks authorised.
    """
    jobs = jobs or _ingest_runs
    manifest_reader = manifest_reader or _read_manifest

    runs = jobs(
        filter=(f'display_name:"{INGEST_PIPELINE}" '
                'AND state="PIPELINE_STATE_SUCCEEDED"'),
        order_by="create_time desc",
    )
    seen = []
    for job in runs[:limit]:
        uris = _artifact_uris(job)
        manifest_uri = uris.get(BUILD_TASK, {}).get("manifest", "")
        if not manifest_uri:
            continue
        described = manifest_reader(manifest_uri)
        if described.get("tree_ah_index") != index_resource_name:
            seen.append(f"{getattr(job, 'display_name', '?')}"
                        f" → {described.get('tree_ah_index', 'no index recorded')}")
            continue
        chunks = uris.get(CHUNK_TASK, {}).get("chunks", "")
        ingest = uris.get(EMBED_TASK, {}).get("ingest", "")
        if not chunks or not ingest:
            continue
        return {
            "run": getattr(job, "display_name", ""),
            "chunks": chunks,
            "ingest": ingest,
            "manifest": manifest_uri,
            "data_fingerprint": described.get("data_fingerprint", ""),
        }

    raise SystemExit(
        f"no successful {INGEST_PIPELINE} run records building "
        f"{index_resource_name}, so there is nothing to measure it against.\n"
        f"Checked the {min(limit, len(runs))} most recent successful runs; "
        "their manifests name:\n  " + "\n  ".join(seen or ["nothing"]) +
        "\nDeploy the index that a run actually produced, or submit an ingest "
        "for this corpus first."
    )


def report_for(*, endpoint: str, deployed_id: str, chunks: str, ingest: str,
               out_dir_uri: str, num_queries: int = 100, top_k: int = 10) -> dict:
    """Measure one deployed id and return the report the job wrote.

    The job runs in the cloud and queries the deployed id it is given, which is
    the staging id while the live id keeps answering. Nothing about this is
    optional: without a report there is no measurement, and a promotion without
    a measurement is not a decision.
    """
    aiplatform.init(project=PROJECT, location=LOCATION,
                    staging_bucket=f"gs://{PROJECT}-mlops/pipeline-root")
    job = aiplatform.CustomJob(
        display_name=f"rag-recall-k-{deployed_id}-{_stamp()}",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": RECALL_MACHINE},
            "replica_count": 1,
            "container_spec": {
                "image_uri": IMAGE,
                "command": ["python", "/app/pipelines/recall_k.py"],
                "args": [
                    "--chunks", chunks,
                    "--ingest", ingest,
                    "--endpoint", endpoint,
                    "--deployed-id", deployed_id,
                    "--out-dir", out_dir_uri,
                    "--num-queries", str(num_queries),
                    "--top-k", str(top_k),
                ],
            },
        }],
    )
    print(f"measuring {deployed_id} against {endpoint.split('/')[-1]} "
          f"({RECALL_MACHINE})…", flush=True)
    job.run(service_account=SERVICE_ACCOUNT)
    print(f"recall job: {job.resource_name}", flush=True)
    return read_report(out_dir_uri)


def read_report(out_dir_uri: str) -> dict:
    """The report a recall job wrote, or refuse if it did not write one."""
    uri = f"{out_dir_uri.rstrip('/')}/{REPORT_NAME}"
    bucket, obj = uri.replace("gs://", "", 1).split("/", 1)
    blob = storage.Client().bucket(bucket).blob(obj)
    if not blob.exists():
        raise SystemExit(f"the recall job wrote no {REPORT_NAME} to {out_dir_uri}, "
                         "so the deployment cannot be judged")
    return json.loads(blob.download_as_text())


def judge(report: dict, *, corpus: str, index_name: str,
          data_fingerprint: str = "") -> tuple[bool, object, list[str]]:
    """Fold a report through the configured thresholds.

    Returns (passed, result, failing metrics). The thresholds come from the
    corpus configuration, so changing what "good enough" means is a config
    change, not a code change.
    """
    cfg = load_config()
    result = recall_result(
        report=report,
        corpus=corpus,
        index_name=index_name,
        data_fingerprint=data_fingerprint,
        recall_min={"recall_at_10": cfg.recall_at_10_min},
        empty_max={"empty_result_rate": cfg.empty_result_rate_max},
    )
    passed, failing = result.verdict()
    return passed, result, list(failing)


def prior_measurement(corpus: str, index_display: str, *, data_fingerprint: str,
                      recall_min: float, empty_max: float,
                      client=None) -> dict | None:
    """The newest measurement already taken for this index, if it still applies.

    Standing the endpoints back up changes nothing about the index: the same
    vectors, built by the same ingest run, judged by the same thresholds.
    Re-running the recall job then spends a machine and twenty minutes
    re-deriving a verdict that is already written down, so the deploy looks for
    the written one first.

    What makes a stored measurement applicable is checked rather than assumed: it
    must have PASSED, it must name this index, it must have been taken against
    the same index contents (the data fingerprint of the ingest run that built
    it), and it must have applied the same thresholds. What is NOT covered is the
    measurement's own code — nothing in the artifact names the recall component
    that produced it, so a report from an older component passes every check
    above. That is why `--force-measure` exists and why a reuse is announced
    with its evidence URI rather than folded in silently.

    Returns ``{'report':…, 'uri':…, 'recall_at_10':…}`` or None when a fresh
    measurement has to be taken.
    """
    client = client or storage.Client()
    bucket_name = f"{PROJECT}-mlops"
    bucket = client.bucket(bucket_name)
    prefix = f"rag/recall/{corpus}/{index_display}/"
    # The stamp is `%Y%m%d-%H%M%S`, so newest-first is a reverse sort of names.
    names = sorted(
        (b.name for b in client.list_blobs(bucket_name, prefix=prefix)
         if b.name.endswith(RESULTS_NAME)),
        reverse=True,
    )
    for name in names:
        try:
            payload = json.loads(bucket.blob(name).download_as_text())
        except (json.JSONDecodeError, UnicodeDecodeError, NotFound):
            continue
        if not payload.get("passed"):
            continue
        if payload.get("index_name") and payload["index_name"] != index_display:
            continue
        if data_fingerprint and payload.get("data_fingerprint") != data_fingerprint:
            continue
        if (payload.get("thresholds") or {}).get("recall_at_10") != recall_min:
            continue
        if (payload.get("max_thresholds") or {}).get("empty_result_rate") != empty_max:
            continue
        return {
            "report": payload,
            "uri": f"gs://{bucket_name}/{name}",
            "recall_at_10": (payload.get("metrics") or {}).get("recall_at_10"),
        }
    return None


def publish(result, out_dir_uri: str) -> dict[str, str]:
    """Write the gate's artifacts beside the report in the bucket.

    ``write_artifacts`` renders into a directory, so the files are rendered
    locally and uploaded; the local copy is a temporary and the bucket is the
    record. A verdict that lives only on the machine that computed it is not
    evidence a reviewer can reach.
    """
    client = storage.Client()
    bucket_name, prefix = out_dir_uri.replace("gs://", "", 1).split("/", 1)
    bucket = client.bucket(bucket_name)
    written: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for kind, path in write_artifacts(result, Path(tmp)).items():
            dest = f"{prefix.rstrip('/')}/{Path(path).name}"
            bucket.blob(dest).upload_from_filename(str(path))
            written[kind] = f"gs://{bucket_name}/{dest}"
    return written


def label_for_recall(index_display: str, recall_at_10) -> str:
    """The deployment label, from a recall already recorded rather than a report.

    A promoted deployment carries the measurement that authorised it in its
    display name, and a reused measurement authorises a promotion just as
    directly as a fresh one — the number on the endpoint has to be the number
    the evidence holds, whichever run produced it.
    """
    suffix = f" recall@10={recall_at_10}" if recall_at_10 is not None else ""
    return f"{index_display}{suffix}"[:LABEL_MAX]


def label(index_display: str, report: dict) -> str:
    """A deployment name carrying the measurement.

    The recall this index achieved travels with the deployment, so the next
    person to look at the endpoint sees what it was measured at rather than
    having to find the report.
    """
    recall = (report or {}).get("recall", {})
    return label_for_recall(index_display, recall.get("@10"))
