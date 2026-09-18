"""deploy_rag — blue-green deploy of the demo RAG index to the public endpoint.

The one deploy path for the retrieval index; the manual wrapper scripts that
used to sit in front of it were folded into this step:

  1. Pick the newest ``rag-tree-ah-*`` index.
  2. Refuse anything over the synthetic-scale limit (ECC-36/53) — this is a
     PUBLIC endpoint, so only the demo (MTSamples) corpus may ever land on it.
  3. Find the ingest run that built it, and refuse if none does: a measurement
     has to describe the artifact it authorises.
  4. Deploy the index under a STAGING id, leaving the live id untouched.
  5. Measure the staging id with the recall@k job, and gate on that report
     against the thresholds in the corpus configuration.
  6. Pass → undeploy the old live id, promote staging → live id carrying the
     achieved recall in its name. Fail → undeploy staging, keep the live id
     serving, and exit non-zero.

Every promotion therefore carries the measurement that authorised it, written
to the bucket under ``rag/recall/<corpus>/<index>/<timestamp>/``.

The full recall@k job (``pipelines/recall_k.py``) requires an endpoint to query;
for the demo corpus it is run against the staging id BEFORE promotion. For the
MIMIC corpus this same flow would need a private (PSC/VPC) endpoint — out of
scope for the public demo and left to Step 6/7.

Usage:
    .venv/bin/python scripts/agent/deploy_rag.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from google.cloud.aiplatform_v1 import (
    IndexEndpointServiceClient,
    IndexServiceClient,
)
from google.cloud.aiplatform_v1.types import (
    DeployedIndex,
    DedicatedResources,
    MachineSpec,
)

from services.mcp.config import (  # noqa: E402
    DEPLOYED_INDEX_ID as LIVE_ID,
    INDEX_ENDPOINT_NAME as ENDPOINT_NAME,
    LOCATION,
    PROJECT,
)
from services.mcp.retrieval.config import load as load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recall_gate  # noqa: E402
from _deploy_guard import assert_synthetic_scale  # noqa: E402

PARENT = f"projects/{PROJECT}/locations/{LOCATION}"
STAGING_ID = "rag_tree_ah_staging"


def _client() -> IndexEndpointServiceClient:
    return IndexEndpointServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})


def _index_client() -> IndexServiceClient:
    return IndexServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})


def newest_tree_index() -> tuple[str, str, int]:
    """(resource_name, display_name, vectors) of the newest rag-tree-ah-* index."""
    client = _index_client()
    candidates = []
    for idx in client.list_indexes(parent=PARENT):
        if idx.display_name.startswith("rag-tree-ah-"):
            vectors = idx.index_stats.vectors_count if idx.index_stats else 0
            candidates.append((idx.create_time, idx.name, idx.display_name, vectors))
    if not candidates:
        raise SystemExit("no rag-tree-ah-* index found — run the ingest pipeline first")
    candidates.sort(reverse=True)
    for created, name, display, vectors in candidates:
        print(f"  {display:32s} vectors={vectors:<8}")
    return candidates[0][1], candidates[0][2], candidates[0][3]


def _find_endpoint(c: IndexEndpointServiceClient) -> str | None:
    for ep in c.list_index_endpoints(parent=PARENT):
        if ep.display_name == ENDPOINT_NAME:
            return ep.name
    return None


def _deploy(c: IndexEndpointServiceClient, ep_name: str, index_name: str,
            deployed_id: str, machine: str, display_name: str = "") -> None:
    di = DeployedIndex(
        id=deployed_id,
        index=index_name,
        display_name=display_name,
        dedicated_resources=DedicatedResources(
            machine_spec=MachineSpec(machine_type=machine),
            min_replica_count=1,
            max_replica_count=1,
        ),
    )
    print(f"deploying {deployed_id} -> {index_name.split('/')[-1]} ({machine})…")
    op = c.deploy_index(index_endpoint=ep_name, deployed_index=di)
    while not op.done():
        time.sleep(5)
    if op.exception() is not None:
        raise SystemExit(f"deploy failed: {op.exception()}")


def _undeploy(c: IndexEndpointServiceClient, ep_name: str, deployed_id: str) -> None:
    print(f"undeploying {deployed_id}…")
    c.undeploy_index(index_endpoint=ep_name, deployed_index_id=deployed_id)


def _deployed_ids(c: IndexEndpointServiceClient, ep_name: str) -> list[str]:
    return [d.id for d in c.get_index_endpoint(name=ep_name).deployed_indexes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    machine = cfg.machine_type
    index_name, display, vectors = newest_tree_index()
    assert_synthetic_scale(vectors, display)

    # Resolve the run that built this index before spending anything: if none
    # records building it, there is nothing to measure it against.
    source = recall_gate.run_for_index(index_name)
    evidence_dir = recall_gate.out_dir(cfg.corpus.name, display)
    print(f"index built by {source['run']}")
    print(f"  chunks: {source['chunks']}")
    print(f"  ingest: {source['ingest']}")

    c = _client()
    ep_name = _find_endpoint(c)
    if ep_name is None:
        if args.dry_run:
            print(f"dry-run: would create endpoint {ENDPOINT_NAME!r}")
        else:
            op = c.create_index_endpoint(
                parent=PARENT,
                index_endpoint={
                    "display_name": ENDPOINT_NAME,
                    "public_endpoint_enabled": True,
                },
            )
            ep_name = op.result().name
            print(f"endpoint created: {ep_name}")
    else:
        print(f"using endpoint: {ep_name}")

    live_present = LIVE_ID in _deployed_ids(c, ep_name) if not args.dry_run else False

    if args.dry_run:
        print(f"dry-run: deploy {display} as {STAGING_ID}, measure that id, "
              f"and promote to {LIVE_ID} only if recall@10 "
              f"≥ {cfg.recall_at_10_min}")
        print(f"dry-run: evidence would be written to {evidence_dir}")
        return 0

    # Blue: the candidate serves under a staging id; the live id is untouched.
    _deploy(c, ep_name, index_name, STAGING_ID, machine)

    # Measure the candidate. A report describing the index already serving
    # cannot authorise the next one, which is why the id below is the staging.
    report = recall_gate.report_for(
        endpoint=ep_name, deployed_id=STAGING_ID, chunks=source["chunks"],
        ingest=source["ingest"], out_dir_uri=evidence_dir)
    passed, result, failing = recall_gate.judge(
        report, corpus=cfg.corpus.name, index_name=display,
        data_fingerprint=source["data_fingerprint"])
    for kind, uri in sorted(recall_gate.publish(result, evidence_dir).items()):
        print(f"  {kind}: {uri}")

    if not passed:
        _undeploy(c, ep_name, STAGING_ID)
        print(f"ROLLED BACK — refused on {failing}; {LIVE_ID} still serving. "
              f"Evidence: {evidence_dir}")
        return 1

    # Green: promote, carrying the measurement on the deployment itself.
    label = recall_gate.label(display, report)
    if live_present:
        _undeploy(c, ep_name, LIVE_ID)
    _deploy(c, ep_name, index_name, LIVE_ID, machine, display_name=label)
    _undeploy(c, ep_name, STAGING_ID)
    print(f"PROMOTED {label} to {LIVE_ID}")
    print(f"evidence: {evidence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
