"""Measure the index the demo endpoint is currently serving.

This is the on-demand counterpart to the gate inside ``deploy_rag.py``: that one
measures a candidate before promoting it, this one measures what is live, for
when the current number is what you want and nothing should be deployed.

The endpoint, the deployed id and the ingest artifacts are all resolved rather
than named. Constants in this file were how it came to point at an endpoint that
no longer existed, a pipeline run from a previous month, and the live deployed
id regardless of which index was asking.

Usage:
    .venv/bin/python scripts/agent/submit_recall_job.py [--num-queries 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.cloud.aiplatform_v1 import (  # noqa: E402
    IndexEndpointServiceClient,
    IndexServiceClient,
)

import recall_gate  # noqa: E402
from services.mcp.config import (  # noqa: E402
    DEPLOYED_INDEX_ID,
    INDEX_ENDPOINT_NAME,
    LOCATION,
    PROJECT,
)
from services.mcp.retrieval.config import load as load_config  # noqa: E402

PARENT = f"projects/{PROJECT}/locations/{LOCATION}"


def _client(cls):
    return cls(client_options={
        "api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})


def served_index(deployed_id: str) -> tuple[str, str]:
    """(endpoint resource name, index resource name) for a deployed id.

    Returns what the endpoint is actually serving, so nothing is measured by a
    name that might mean something else: a stale id would otherwise be measured
    against a live endpoint and reported as if it described it.
    """
    endpoints = _client(IndexEndpointServiceClient)
    for endpoint in endpoints.list_index_endpoints(parent=PARENT):
        if endpoint.display_name != INDEX_ENDPOINT_NAME:
            continue
        for deployed in endpoint.deployed_indexes:
            if deployed.id == deployed_id:
                return endpoint.name, deployed.index
    raise SystemExit(
        f"no endpoint named {INDEX_ENDPOINT_NAME!r} is serving a deployed index "
        f"with id {deployed_id!r}. Nothing is live to measure; deploy first."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployed-id", default=DEPLOYED_INDEX_ID)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config()
    endpoint, index = served_index(args.deployed_id)
    display = _client(IndexServiceClient).get_index(name=index).display_name
    source = recall_gate.run_for_index(index)
    out_dir = recall_gate.out_dir(cfg.corpus.name, display)
    print(f"measuring {display} ({args.deployed_id} on "
          f"{endpoint.split('/')[-1]})")
    print(f"  corpus: {cfg.corpus.name}   index built by {source['run']}")

    report = recall_gate.report_for(
        endpoint=endpoint, deployed_id=args.deployed_id,
        chunks=source["chunks"], ingest=source["ingest"],
        out_dir_uri=out_dir, num_queries=args.num_queries, top_k=args.top_k)
    passed, result, failing = recall_gate.judge(
        report, corpus=cfg.corpus.name, index_name=display,
        data_fingerprint=source["data_fingerprint"])

    for metric, value in sorted(report.get("recall", {}).items()):
        print(f"  {metric} = {value}")
    for kind, uri in sorted(recall_gate.publish(result, out_dir).items()):
        print(f"  {kind}: {uri}")

    if passed:
        print(f"PASS against the configured thresholds; evidence in {out_dir}")
        return 0
    unmeasured = result.unmeasured()
    if unmeasured:
        print(f"NOT MEASURED: {unmeasured} — the report carries no value for "
              f"{'these thresholds' if len(unmeasured) > 1 else 'this threshold'}, "
              f"so no verdict on the corpus was reached. Evidence in {out_dir}")
    else:
        print(f"FAIL on {failing}; evidence in {out_dir}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
