"""Tag a run's traces in Langfuse, so the console can pull them up by tag.

Why the harness writes the tag rather than the agent. Putting a tag on a trace at
the moment it is created means telling the agent it is serving an evaluation,
which means changing the request it accepts and redeploying it — changing the
thing being measured, in the middle of measuring it. So the tag is written
afterwards, from outside, against the trace ids the collector already recorded.
The deployment under test is untouched.

Two behaviours of the write are stated here because they were measured rather
than assumed (2026-09-30), and both shape the code:

  * The write **merges**. Sending a trace id and a tag does not clear the trace's
    name, input, output, timestamp or steps. If it replaced them, a pass over a
    whole run would blank every trace it touched.
  * A write for a trace id that has **no trace** creates a bare, empty one. So
    this checks that each trace exists before writing to it. Tagging a dangling
    id would manufacture an empty trace and make a broken export look present.
    That check is the reason this script also reports ids that never arrived: a
    non-zero count means tracing failed for that run.

One prerequisite, outside this file: the Langfuse stack must be able to accept a
trace write. Under v4's `events_only` write mode its ingestion endpoint takes
score events only, and a trace write is refused with `Event type not accepted`.
`infra/langfuse/deploy.sh` and `infra/langfuse/docker-compose.yml` set
`LANGFUSE_MIGRATION_V4_WRITE_MODE=dual` to lift that, on the web service and the
worker both — one without the other accepts an event and then drops it.

Usage (harness root):
    # the run's traces, retagged for the day
    .venv/bin/python evaluation/agent/tag_traces.py --tag eval-093026

    # see what would be written without writing
    .venv/bin/python evaluation/agent/tag_traces.py --tag eval-093026 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HERE))

RESULTS = HERE / "results"

# Every trace file a run produces. Searched recursively, because a two-turn case
# writes one record per answer and each carries its own trace identifier.
DEFAULT_TRACES = (
    RESULTS / "traces_http.jsonl",
    RESULTS / "adversarial_http.jsonl",
)

PROJECT = "trim-icon-498815-a0"
HOST = "https://observability.danielmherman.com"


def _client():
    """A Langfuse client, from the environment or from Secret Manager.

    The keys are deliberately not in the repository, and the shell that runs an
    evaluation does not always have them exported. When they are absent they are
    read from Secret Manager inside this process, which is how the stack's own
    scripts provision themselves (`infra/langfuse/deploy.sh`). Nothing is echoed.
    """
    from langfuse import Langfuse

    have_env = all(os.environ.get(name) for name in
                   ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"))
    if have_env:
        print("credentials: from the environment")
        return Langfuse()

    print("credentials: LANGFUSE_* not set; reading from Secret Manager")
    os.environ["LANGFUSE_HOST"] = HOST
    for var, secret in (("LANGFUSE_PUBLIC_KEY", "langfuse-project-public-key"),
                        ("LANGFUSE_SECRET_KEY", "langfuse-project-secret-key")):
        proc = subprocess.run(
            ["gcloud", "secrets", "versions", "access", "latest",
             f"--secret={secret}", f"--project={PROJECT}"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"could not read {secret}: {proc.stderr.strip()}")
        os.environ[var] = proc.stdout.strip()
    return Langfuse()


def trace_ids_in(value) -> list[str]:
    """Every `langfuse_trace_id` anywhere in a record.

    Recursive because the shape differs by file: a single-turn record carries one
    at the top level, and a follow-up record carries one per turn, nested under
    `turn1` and `turn2`. Reading only the top level would silently tag half of a
    follow-up run.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "langfuse_trace_id" and isinstance(item, str) and item:
                found.append(item)
            else:
                found.extend(trace_ids_in(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(trace_ids_in(item))
    return found


def ids_from_file(path: Path) -> list[str]:
    ids: list[str] = []
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ids.extend(trace_ids_in(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return ids


def _lookup(client, trace_id: str, attempts: int = 3) -> tuple[str, list | None, str]:
    """(trace_id, tags, state), where state is `ok`, `absent` or the failure.

    The three outcomes are separated because one of them is a claim about the
    system and the others are not. This function decides whether a trace id is
    reported as never having arrived, and a read that timed out is not evidence
    of absence — collapsing the two would report tracing as broken on the
    strength of a slow afternoon. A 404 is the service saying the trace is not
    there, which is the claim worth making.

    Retried, because the store answered a single-trace read in fourteen seconds
    when measured and does not enjoy being read concurrently.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            trace = client.api.trace.get(
                trace_id, request_options={"timeout_in_seconds": 90})
            return trace_id, list(trace.tags or []), "ok"
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return trace_id, None, "absent"
            last = exc
            if attempt < attempts - 1:
                time.sleep(2 + 2 * attempt)
    return trace_id, None, f"unreadable: {type(last).__name__}"


def _write(client, trace_ids: list[str], tag: str, batch_size: int) -> dict:
    from langfuse.api.ingestion.types.ingestion_event import (
        IngestionEvent_TraceCreate,
    )
    from langfuse.api.ingestion.types.trace_body import TraceBody

    written = failed = 0
    for start in range(0, len(trace_ids), batch_size):
        chunk = trace_ids[start:start + batch_size]
        now = datetime.now(timezone.utc).isoformat()
        events = [
            IngestionEvent_TraceCreate(
                id=str(uuid.uuid4()), timestamp=now,
                body=TraceBody(id=trace_id, tags=[tag]),
            )
            for trace_id in chunk
        ]
        response = client.api.ingestion.batch(
            batch=events, request_options={"timeout_in_seconds": 180})
        written += len(response.successes or [])
        for error in response.errors or []:
            failed += 1
            print(f"  refused: {error.message} {str(error.error)[:160]}")
    return {"written": written, "failed": failed}


def _verify(client, trace_ids: list[str], tag: str, attempts: int) -> int:
    """How many of a sample carry the tag. The write is asynchronous."""
    sample = trace_ids[:5]
    tagged = 0
    for attempt in range(attempts):
        tagged = 0
        for trace_id in sample:
            _, tags, state = _lookup(client, trace_id, attempts=1)
            if state == "ok" and tags and tag in tags:
                tagged += 1
        if tagged == len(sample):
            break
        if attempt < attempts - 1:
            time.sleep(5)
    return tagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True,
                    help="the single tag to write, e.g. eval-093026")
    ap.add_argument("--traces", nargs="*", default=[str(p) for p in DEFAULT_TRACES])
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent trace reads; the store answers a single "
                         "read slowly when read concurrently")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-attempts", type=int, default=6)
    args = ap.parse_args()

    per_file = {}
    ordered: list[str] = []
    for name in args.traces:
        path = Path(name)
        ids = ids_from_file(path)
        per_file[path.name] = len(ids)
        for trace_id in ids:
            if trace_id not in ordered:
                ordered.append(trace_id)

    print(f"tag: {args.tag}")
    for name, count in per_file.items():
        print(f"  {name:<26} {count:>4} trace ids")
    print(f"  distinct                     {len(ordered):>4}")

    if not ordered:
        print("\nnothing to tag — no trace ids in those files")
        return 1

    client = _client()
    # No `auth_check()`. It reads a different endpoint from the one this script
    # uses, and against this deployment that call follows a redirect and then
    # stalls until it times out — measured 2026-09-30, while the site itself
    # answered in 0.1s and a single-trace read answered normally. It is also
    # redundant: reading a trace exercises the same credential against the
    # endpoint the work actually needs, so the classification below is the
    # readiness check.

    # Existence first: a write to an id with no trace creates an empty one.
    print(f"\nchecking {len(ordered)} traces …")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        looked_up = list(pool.map(lambda t: _lookup(client, t), ordered))

    present = [(t, tags) for t, tags, state in looked_up if state == "ok"]
    absent = [t for t, tags, state in looked_up if state == "absent"]
    unreadable = [(t, state) for t, tags, state in looked_up
                  if state not in ("ok", "absent")]
    already = [t for t, tags in present if args.tag in tags]
    todo = [t for t, tags in present if args.tag not in tags]

    print(f"  present              {len(present):>4}")
    print(f"  already carry it     {len(already):>4}")
    print(f"  to tag               {len(todo):>4}")
    print(f"  NO TRACE IN LANGFUSE {len(absent):>4}")
    if absent:
        print("  ^ tracing did not deliver these — the tag cannot reach them, and")
        print("    writing anyway would create an empty trace for each id")
        for trace_id in absent[:5]:
            print(f"    {trace_id}")
    if unreadable:
        print(f"  could not be read    {len(unreadable):>4}")
        print("  ^ a read that did not complete is not evidence that a trace is")
        print("    absent. Re-run to tag these: a trace already carrying the tag is")
        print("    skipped, so a re-run is safe.")
        for trace_id, state in unreadable[:5]:
            print(f"    {trace_id}  ({state})")

    if unreadable and not present:
        print("\nno trace could be read at all. A credential or a store fault "
              "looks like this, and neither is a missing trace; nothing was "
              "written.")
        return 1

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    if not todo:
        print("\nnothing to write")
        return 0

    print(f"\nwriting the tag onto {len(todo)} traces "
          f"(batches of {args.batch_size}) …")
    result = _write(client, todo, args.tag, args.batch_size)
    print(f"  accepted {result['written']}, refused {result['failed']}")
    client.flush()

    if args.verify_attempts:
        tagged = _verify(client, todo, args.tag, args.verify_attempts)
        print(f"\nverified: {tagged} of the first {min(5, len(todo))} carry "
              f"'{args.tag}' (the write is asynchronous, so a low count now may "
              f"rise shortly)")

    # A non-zero exit when anything is left untagged, so this step fails visibly
    # rather than reporting success over a partial tag.
    if absent or unreadable:
        print(f"\n{len(absent)} trace(s) absent and {len(unreadable)} "
              f"unreadable: the tag is incomplete. Re-run this step.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
