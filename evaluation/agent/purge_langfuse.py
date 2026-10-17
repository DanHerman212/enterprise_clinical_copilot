"""Empty a Langfuse project: delete every score, then every trace.

Why this exists separately from `purge_traces.py`: deleting a trace does not delete the
scores attached to it. Langfuse keeps scores as their own records, so a project can show
hundreds of scores pointing at traces that no longer exist — which is exactly the state
that made "delete the dead data" look like it was not working. Traces and scores are two
collections and both have to be emptied.

Scores go first. A score is a child of a trace by convention only; deleting the trace
first leaves the score orphaned and still listed, and deleting the parent then the child
is the order that cannot leave a half-state behind.

Dry run by default. `--confirm` deletes.

Usage:
    python purge_langfuse.py --since 2026-10-01            # report only
    python purge_langfuse.py --since 2026-10-01 --confirm
"""

from __future__ import annotations

import argparse
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
    return (
        os.environ.get("LANGFUSE_PUBLIC_KEY") or _secret("langfuse-project-public-key"),
        os.environ.get("LANGFUSE_SECRET_KEY") or _secret("langfuse-project-secret-key"),
    )


def _scores(pk: str, sk: str, since: dt.datetime) -> list[dict]:
    """Every score in the window, oldest first.

    The v2 endpoint is used because the legacy one refuses wide ranges with a 422 that
    reads like a permissions error. `fromTimestamp`/`toTimestamp` are both required for
    the same reason.
    """
    out: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            f"{HOST}/api/public/v2/scores", auth=(pk, sk),
            params={"fromTimestamp": since.isoformat(), "limit": 100, "page": page},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"  score list {page}: {resp.status_code} {resp.text[:160]}")
            return out
        data = resp.json().get("data") or []
        out.extend(data)
        if len(data) < 100:
            return out
        page += 1


def _traces_slice(pk: str, sk: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
    """Every trace in one slice, paged.

    Paging matters: a slice that contains more than the page size returns only the first
    page, so a project with thousands of traces in one hour looked like it had a hundred
    and the cleanup silently left the rest.
    """
    rows: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            f"{HOST}/api/public/traces", auth=(pk, sk),
            params={"fromTimestamp": start.isoformat(), "toTimestamp": end.isoformat(),
                    "limit": 100, "page": page},
            timeout=60,
        )
        if resp.status_code != 200:
            return rows
        data = resp.json().get("data") or []
        rows.extend(data)
        if len(data) < 100:
            return rows
        page += 1


def _delete_all(urls: list[str], pk: str, sk: str, workers: int = 16) -> tuple[int, int]:
    """Delete every URL concurrently. Returns (deleted, failed).

    Serial deletes were measured at roughly one record every 200ms, which is fine for a
    handful and hopeless for a project holding thousands: the cleanup ran for minutes and
    the count barely moved. These are independent requests against one API, so the only
    thing serialising them bought was time.
    """
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    def one(url: str) -> bool:
        try:
            resp = requests.delete(url, auth=(pk, sk), timeout=60)
            return resp.status_code in (200, 202, 204)
        except requests.RequestException:
            return False

    deleted = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ok in pool.map(one, urls):
            if ok:
                deleted += 1
            else:
                failed += 1
    return deleted, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2026-10-01", help="first day to clear (UTC)")
    ap.add_argument("--slice-minutes", type=int, default=60,
                    help="query window size; smaller is safer, larger is faster")
    ap.add_argument("--keep-tag", default=None,
                    help="do not delete a trace carrying this tag (e.g. run:eval-2026-10-17)")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    since = dt.datetime.fromisoformat(args.since).replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    pk, sk = _credentials()

    scores = _scores(pk, sk, since)
    traces: list[dict] = []
    cursor = since
    while cursor < now:
        nxt = min(cursor + dt.timedelta(minutes=args.slice_minutes), now)
        traces.extend(_traces_slice(pk, sk, cursor, nxt))
        cursor = nxt
    if args.keep_tag:
        before = len(traces)
        traces = [t for t in traces
                  if args.keep_tag not in (t.get("tags") or [])]
        kept_ids = {t.get("id") for t in
                    _traces_slice(pk, sk, since, now)
                    if args.keep_tag in (t.get("tags") or [])}
        print(f"kept {before - len(traces)} traces tagged {args.keep_tag}")
        # A kept trace's scores are part of what is being kept. Deleting them would
        # leave the surviving run as the untagged, unscored noise this cleanup exists
        # to remove.
        before_scores = len(scores)
        scores = [s for s in scores if s.get("traceId") not in kept_ids]
        print(f"kept {before_scores - len(scores)} scores on those traces")
    print(f"scores since {args.since}: {len(scores)}")
    print(f"traces since {args.since}: {len(traces)}")

    if not args.confirm:
        print("\ndry run: nothing deleted. Re-run with --confirm.")
        return 0

    deleted_scores, failed_scores = _delete_all(
        [f"{HOST}/api/public/scores/{s['id']}" for s in scores], pk, sk)
    print(f"scores deleted {deleted_scores}, failed {failed_scores}")

    deleted_traces, failed_traces = _delete_all(
        [f"{HOST}/api/public/traces/{t['id']}" for t in traces], pk, sk)
    print(f"traces deleted {deleted_traces}, failed {failed_traces}")

    remaining_scores = len(_scores(pk, sk, since))
    print(f"remaining scores since {args.since}: {remaining_scores}")
    return 0 if failed_scores == 0 and failed_traces == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
