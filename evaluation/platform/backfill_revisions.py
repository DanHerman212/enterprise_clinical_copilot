"""Recover the serving revision for traces that were never stamped with one.

Why this exists
---------------
Until 2026-10-27 no trace carried the deployment that produced it. `CODE_REVISION`
reached `/health` and the per-turn execution record, and never the trace — so a run's
answers could be dated but not attributed, and the chain was not a versioned artifact.

The forward fix stamps `rev:` on every trace at creation. That does nothing for traces
already written: Langfuse's trace API here is `get`/`list`/`delete` with no update, so an
existing trace cannot be amended.

What it can still be given is the missing link. Every completed turn writes one
`agent_execution` line to stderr carrying `langfuse_trace_id` and `code_revision`, and the
log entry itself carries `resource.labels.revision_name` from Cloud Run. This script reads
those lines over a window and emits the join as a file in the repository, which is what
makes the run an artifact: trace id → revision (→ commit, where `CODE_REVISION` is set).

Limits, stated rather than discovered later:

* Only turns that reached `record_execution` are covered. A turn killed mid-flight (a
  deploy landing, a timeout, a cancelled run) wrote no record and is reported as a gap.
* The window is bounded by the log bucket's retention. This is a one-off rescue of recent
  history, not a durable mechanism.
* The record's `code_revision` is `K_REVISION` unless the deploy sets `CODE_REVISION`;
  the log entry's `revision_name` is always the Cloud Run revision.

Usage:
    .venv/bin/python evaluation/platform/backfill_revisions.py \\
        --since 2026-10-27T00:00:00Z --out evaluation/results/trace_revisions_271026.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PROJECT = "trim-icon-498815-a0"

#: `INFO 2026-10-27 22:19:38,633 services.agent.telemetry.observability {json}`
_RECORD = re.compile(r"(\{.*\})\s*$", re.DOTALL)


def parse_execution(payload: str) -> dict[str, Any] | None:
    """The execution record inside one log line, or None when the line is not one.

    The record is printed by `logging`, so it arrives with a timestamp and a logger name
    in front of it. Everything from the first `{` that parses is the record.
    """
    match = _RECORD.search(payload or "")
    if not match:
        return None
    try:
        record = json.loads(match.group(1))
    except ValueError:
        return None
    if record.get("event") != "agent_execution":
        return None
    return record


def fold(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group entries by run tag, then by trace id, keeping the newest record per trace.

    A trace id appears once per turn and a turn writes one record, so a collision means
    a retry. The newest wins, and a disagreement is reported rather than silently
    resolved: two different revisions on one trace id would mean the id is not unique,
    which is worth knowing.
    """
    runs: dict[str, dict[str, Any]] = {}
    for entry in entries:
        record = parse_execution(str(entry.get("payload") or ""))
        if record is None:
            continue
        trace_id = str(record.get("langfuse_trace_id") or "").strip()
        if not trace_id:
            continue
        run = str(record.get("run_tag") or "").strip() or "(no run tag)"
        bucket = runs.setdefault(run, {"traces": {}, "conflicts": []})
        revisions = {
            "code_revision": record.get("code_revision"),
            "revision_name": (entry.get("resource") or {}).get("revision_name"),
            "timestamp": entry.get("timestamp"),
        }
        previous = bucket["traces"].get(trace_id)
        if previous and previous["code_revision"] != revisions["code_revision"]:
            bucket["conflicts"].append({
                "trace_id": trace_id,
                "seen": [previous["code_revision"], revisions["code_revision"]],
            })
        if not previous or str(revisions["timestamp"]) >= str(previous["timestamp"]):
            bucket["traces"][trace_id] = revisions
    return runs


def read_log(project: str, since: str, until: str | None) -> list[dict[str, Any]]:
    """Execution records from Cloud Logging, newest first.

    Shelling out to `gcloud` rather than using the client library: this is a rescue
    script, and its one dependency should be the CLI the operator already has
    authenticated.
    """
    query = f'"agent_execution" AND timestamp>="{since}"'
    if until:
        query += f' AND timestamp<="{until}"'
    command = [
        "gcloud", "logging", "read", query,
        "--project", project,
        "--limit", "5000",
        "--order", "desc",
        "--format", "json(logName,resource.labels.revision_name,textPayload,timestamp)",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"gcloud logging read failed: {result.stderr.strip()[:400]}")
    rows = json.loads(result.stdout or "[]")
    return [
        {"payload": row.get("textPayload"),
         "resource": row.get("resource", {}).get("labels") or {},
         "timestamp": row.get("timestamp")}
        for row in rows
    ]


def runs_by_trace(run_names: list[str]) -> dict[str, set[str]]:
    """Trace ids per Langfuse run name.

    The execution record does not carry `run_tag`, so run membership has to come from
    Langfuse — which is also the authority for it. The join key is the trace id, the one
    identifier both systems hold.
    """
    from langfuse import Langfuse  # noqa: PLC0415 — only needed for this mode

    client = Langfuse()
    out: dict[str, set[str]] = {}
    for run in run_names:
        ids: set[str] = set()
        page = 1
        while True:
            response = client.api.trace.list(limit=100, page=page)
            rows = getattr(response, "data", None) or []
            if not rows:
                break
            for trace in rows:
                if (trace.metadata or {}).get("run_tag") == run:
                    ids.add(str(trace.id))
            page += 1
            if page > 30:
                break
        out[run] = ids
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", required=True, help="RFC3339 start of the window")
    ap.add_argument("--until", default=None, help="RFC3339 end of the window")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--out", default=None, help="write the map here as JSON")
    ap.add_argument("--runs", default="",
                    help="comma-separated Langfuse run names to break the map down by")
    args = ap.parse_args(argv)

    runs = fold(read_log(args.project, args.since, args.until))
    if not runs:
        print("no execution records in the window", file=sys.stderr)
        return 1

    # One flat map: trace id -> the revision that served it. This is the join the
    # traces were missing, and it is independent of any grouping.
    by_trace: dict[str, dict[str, Any]] = {}
    for bucket in runs.values():
        by_trace.update(bucket["traces"])

    per_run: dict[str, Any] = {}
    wanted = [name.strip() for name in args.runs.split(",") if name.strip()]
    if wanted:
        for run, ids in runs_by_trace(wanted).items():
            covered = {trace: by_trace[trace] for trace in ids if trace in by_trace}
            missing = sorted(ids - set(covered))
            counts: dict[str, int] = {}
            for entry in covered.values():
                key = str(entry["code_revision"])
                counts[key] = counts.get(key, 0) + 1
            per_run[run] = {"traces": len(ids), "attributed": covered,
                            "unattributed": missing, "revisions": counts}

    payload = {"since": args.since, "until": args.until, "project": args.project,
               "source": "run.googleapis.com/stderr agent_execution records",
               "traces": by_trace, "runs": per_run}
    for run, bucket in sorted(per_run.items()):
        print(f"{run}: {bucket['traces']} trace(s), {len(bucket['attributed'])} attributed "
              f"-> {bucket['revisions']}"
              + (f", {len(bucket['unattributed'])} with no record"
                 if bucket["unattributed"] else ""))
    print(f"total: {len(by_trace)} trace(s) attributed")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
