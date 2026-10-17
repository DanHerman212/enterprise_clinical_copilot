"""Delete evaluation-run traces from Langfuse, surgically.

Why a tool and not a one-liner: the deletion is irreversible, and the thing that picks
the victims has to be reviewable before it picks them. Dry run is the default; nothing
is deleted without `--confirm`.

Scope is a set of time windows, not a name or tag filter, and that is a compromise worth
stating. Evaluation traces are named `LangGraph` and `command.meds` exactly as the
site's own traffic is, and they carry the same `hadm:<id>` / `command:<kind>` tags —
there is no field today that says "this trace came from the harness". So the only honest
discriminator is *when* the harness ran. Windows are given explicitly on the command
line, which also means a mistake here is visible in the command rather than buried in a
filter expression.

The real fix is upstream: evaluation turns should carry a `run:<tag>` tag so this
question never has to be answered by timestamp again. Until they do, this tool exists to
clean up after the runs that predate that tag.

Usage:
    python purge_traces.py --window 20:31-20:33 --window 21:49-22:07            # dry run
    python purge_traces.py --window 20:31-20:33 --window 21:49-22:07 --confirm
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import subprocess
import sys

import requests

HOST = "https://observability.danielmherman.com"
PROJECT = "trim-icon-498815-a0"


def _secret(name: str) -> str:
    return subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         "--secret", name, "--project", PROJECT],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _credentials() -> tuple[str, str]:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY") or _secret("langfuse-project-public-key")
    sk = os.environ.get("LANGFUSE_SECRET_KEY") or _secret("langfuse-project-secret-key")
    return pk, sk


def _windows(specs: list[str], day: dt.date) -> list[tuple[dt.datetime, dt.datetime]]:
    out = []
    for spec in specs:
        start, _, end = spec.partition("-")
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
        out.append((
            dt.datetime.combine(day, dt.time(sh, sm), tzinfo=dt.timezone.utc),
            dt.datetime.combine(day, dt.time(eh, em), tzinfo=dt.timezone.utc),
        ))
    return out


def _traces_in(
    pk: str, sk: str, start: dt.datetime, end: dt.datetime
) -> list[dict]:
    """Every trace whose start falls inside one window.

    Queried per window rather than by paging through the whole day. Both bounds are
    required — this endpoint answers 422 to a from-only query — and paging deep into a
    day's traffic also produced intermittent 422s upstream, which is a failure mode
    this tool has no business depending on. A window is a page or two at most.
    """
    rows: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            f"{HOST}/api/public/traces", auth=(pk, sk),
            params={"fromTimestamp": start.isoformat(), "toTimestamp": end.isoformat(),
                    "limit": 100, "page": page}, timeout=60,
        )
        if resp.status_code != 200:
            raise SystemExit(
                f"trace list failed for {start:%H:%M}-{end:%H:%M}: "
                f"{resp.status_code} {resp.text[:200]}"
            )
        data = resp.json().get("data") or []
        rows.extend(data)
        if len(data) < 100:
            return rows
        page += 1


def _parse(trace: dict) -> dt.datetime | None:
    raw = trace.get("timestamp")
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", action="append", required=True,
                    metavar="HH:MM-HH:MM",
                    help="UTC window of an evaluation run; repeatable")
    ap.add_argument("--day", default=dt.date.today().isoformat())
    ap.add_argument("--confirm", action="store_true",
                    help="actually delete; without it this only reports")
    args = ap.parse_args()

    day = dt.date.fromisoformat(args.day)
    windows = _windows(args.window, day)
    pk, sk = _credentials()

    # Collected per window, so "inside" is defined by the query rather than by a
    # filter applied afterwards to a larger, unrelated set.
    inside: list[dict] = []
    for a, b in windows:
        found = _traces_in(pk, sk, a, b)
        print(f"window {a:%H:%M}-{b:%H:%M} UTC: {len(found)} traces")
        inside.extend(found)

    print(f"selected for deletion: {len(inside)}")
    print("  names:", dict(collections.Counter(
        (t.get("name") or "?") for t in inside).most_common(8)))

    if not args.confirm:
        print("\ndry run: nothing deleted. Re-run with --confirm to delete.")
        return 0

    deleted = failed = 0
    for trace in inside:
        tid = trace.get("id")
        resp = requests.delete(f"{HOST}/api/public/traces/{tid}",
                               auth=(pk, sk), timeout=60)
        if resp.status_code in (200, 202, 204):
            deleted += 1
        else:
            failed += 1
            print(f"  failed {tid}: {resp.status_code} {resp.text[:120]}")
    print(f"\ndeleted {deleted}, failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
