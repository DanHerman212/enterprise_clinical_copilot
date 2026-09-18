"""Enforce the stated lifetimes of the two holdings that are not conversations.

The conversation store has a sweep (``danielmherman``'s
``purge_expired_conversations``, scheduled daily in production). The layer 8
retention decision named two other holdings of clinical text and gave each its
own lifetime, and until this script existed both were promises with nothing
behind them:

  * the **note cache** — ``~/.cache/enterprise_clinical_copilot`` (or
    ``NOTE_CACHE_DIR``): the discharge-note corpus as gzipped JSONL, plus the
    chunks and embedding inputs derived from it. Decided: ephemeral and
    regenerable, with a very short lifetime. It is deleted in full and rebuilt
    by ``scripts/fetch_note_cache.py``; nothing here is authoritative, and the
    deployed retrieval service carries its own copy in its image, so deleting the
    local one costs a re-fetch and nothing else.
  * the **evaluation trace archive** — ``evaluation/agent/results/traces.jsonl``:
    one record per (admission, prompt) with the answer and the tool calls that
    produced it. Decided: a fixed, longer lifetime, because it is the evidence
    behind an evaluation and an audit needs it to outlive the run.

Usage::

    python scripts/agent/retention.py                 # dry run, both holdings
    python scripts/agent/retention.py --apply         # enforce
    python scripts/agent/retention.py --apply --holdings note-cache

Dry run is the default because both actions are deletions of clinical text, and
a deletion should be something an operator asked for rather than something a
typo performed.

A trace record written before this script existed carries no timestamp. It
cannot be aged, so it is counted and reported and kept: dropping it would be a
deletion nobody decided, and pretending it is new would be the same mistake in
the other direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.mcp.retrieval.notes import CACHE_DIR  # noqa: E402

# The archive's lifetime. Longer than a conversation's by design (an audit reads
# it after the fact), still fixed: "keep it forever" is not a lifetime.
TRACE_RETENTION_DAYS = 90

HARNESS = Path(__file__).resolve().parents[2]
# The collector's own output path, and the same one it reads its sample from.
# Kept as a literal rather than imported from `collect.py`, which imports the
# agent graph and a model client to be read; a test asserts the two agree so the
# writer and the pruner cannot drift apart.
DEFAULT_TRACE_FILE = HARNESS / "evaluation" / "agent" / "results" / "traces.jsonl"


def note_cache_state(cache_dir: Path = None) -> list[tuple[Path, int]]:
    """(path, bytes) for every file in the note cache, or [] when it is absent."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    if not cache_dir.exists():
        return []
    return sorted(
        (path, path.stat().st_size)
        for path in cache_dir.rglob("*")
        if path.is_file()
    )


def purge_note_cache(cache_dir: Path = None, *, apply: bool = False) -> list[Path]:
    """Delete the note cache. Regenerable, so nothing is read before deleting."""
    files = note_cache_state(cache_dir)
    if not apply:
        return [path for path, _ in files]
    for path, _ in files:
        path.unlink()
    return [path for path, _ in files]


def _record_time(record: dict) -> datetime | None:
    """When a trace record was written, or None when it does not say."""
    raw = record.get("at")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def prune_traces(
    path: Path = None, *, days: int = TRACE_RETENTION_DAYS,
    now: datetime | None = None, apply: bool = False,
) -> dict:
    """Drop trace records older than ``days``. Returns what it did.

    Returns ``{'kept', 'dropped', 'undated', 'missing'}``. `undated` counts the
    records that carry no timestamp: they are always kept, and the count is
    returned rather than logged so a caller can say so out loud.
    """
    path = Path(path or DEFAULT_TRACE_FILE)
    if not path.exists():
        return {"kept": 0, "dropped": 0, "undated": 0, "missing": True}

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    kept: list[str] = []
    dropped = undated = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A truncated line is not evidence of anything, and deleting it
            # would destroy the record of a crash. Keep it, count it as undated.
            kept.append(line)
            undated += 1
            continue
        written = _record_time(record)
        if written is None:
            kept.append(line)
            undated += 1
        elif written < cutoff:
            dropped += 1
        else:
            kept.append(line)

    if apply and dropped:
        path.write_text(
            "".join(f"{line}\n" for line in kept), encoding="utf-8")
    return {"kept": len(kept), "dropped": dropped, "undated": undated,
            "missing": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="perform the deletions; without it this is a dry run")
    parser.add_argument(
        "--holdings", choices=("both", "note-cache", "traces"), default="both")
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--note-cache-dir", type=Path, default=None)
    parser.add_argument("--trace-retention-days", type=int,
                        default=TRACE_RETENTION_DAYS)
    args = parser.parse_args()
    mode = "APPLY" if args.apply else "DRY RUN"

    if args.holdings in ("both", "note-cache"):
        files = note_cache_state(args.note_cache_dir)
        total = sum(size for _, size in files)
        print(f"[{mode}] note cache ({args.note_cache_dir or CACHE_DIR}): "
              f"{len(files)} file(s), {total / 1e6:.1f} MB")
        for path, size in files:
            print(f"  {'deleting' if args.apply else 'would delete'} "
                  f"{path.name} ({size / 1e6:.1f} MB)")
        if not files:
            print("  nothing cached")
        elif not args.apply:
            print("  re-fetch with scripts/fetch_note_cache.py if needed")
        purge_note_cache(args.note_cache_dir, apply=args.apply)

    if args.holdings in ("both", "traces"):
        result = prune_traces(
            args.trace_file, days=args.trace_retention_days, apply=args.apply)
        if result["missing"]:
            print(f"[{mode}] trace archive: no file at {args.trace_file}")
        else:
            print(f"[{mode}] trace archive ({args.trace_file}): "
                  f"keeps {result['kept']} record(s) within "
                  f"{args.trace_retention_days} day(s), "
                  f"{'dropped' if args.apply else 'would drop'} "
                  f"{result['dropped']}")
            if result["undated"]:
                print(f"  {result['undated']} record(s) carry no timestamp and "
                      f"are kept: an undated record cannot be aged, and "
                      f"deleting it would be a decision nobody made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
