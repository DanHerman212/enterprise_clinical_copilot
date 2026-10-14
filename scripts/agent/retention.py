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
  * the **evaluation trace archives** — ``evaluation/agent/results/traces*.jsonl``:
    one record per (admission, prompt) with the answer and the tool calls that
    produced it. Decided: a fixed, longer lifetime, because it is the evidence
    behind an evaluation and an audit needs it to outlive the run.

    The holding is the SET, not the file. Naming one file left sixteen archives
    outside the window while a single one was inside it, which is a retention
    policy in name only; the sweep matches the pattern every harness here has
    written, and exempts the one archive that is deliberately immutable.

Usage::

    python scripts/agent/retention.py                 # dry run, both holdings
    python scripts/agent/retention.py --apply         # enforce
    python scripts/agent/retention.py --apply --holdings note-cache

Dry run is the default because both actions are deletions of clinical text, and
a deletion should be something an operator asked for rather than something a
typo performed.

A trace record written before this script existed carries no timestamp. It cannot
be aged on its own, so the ARCHIVE carrying it is aged instead: a file holding any
undated record is deleted whole once the file passes the window, because a record
with no timestamp is not kept indefinitely by default. The count is reported
first — an archive about to be purged is named, with how long it has left, so the
deletion is one an operator saw coming rather than one performed by a typo.
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
# Kept as a literal rather than imported from `collect_http.py`, which imports the
# agent graph and a model client to be read; a test asserts the two agree so the
# writer and the pruner cannot drift apart.
#
# This is the file the harness writes TODAY. It is not the only archive the window
# has to cover — see `TRACE_GLOB` — but it is the one whose name has to match the
# collector's, which is the drift the test guards.
DEFAULT_TRACE_FILE = (
    HARNESS / "evaluation" / "agent" / "results" / "traces_http.jsonl"
)

# The directory the archives live in, and the shape of their names.
#
# A window that names one file covers one file. Every harness that has written
# here has named its output `traces*.jsonl` — a date suffix, a run label, a shard
# number — so the window is applied to the PATTERN and not to a filename. Found
# live 2026-10-12: one archive inside a retention window and sixteen outside it,
# because the window named the one the collector had stopped writing.
DEFAULT_TRACE_DIR = HARNESS / "evaluation" / "agent" / "results"
TRACE_GLOB = "traces*.jsonl"

# Exempt from the window, by name, permanently. This is not a filename that
# happens to be old: it is the archive a documented evaluation procedure treats as
# immutable (alongside `judged.jsonl` and `golden_report.json`), and the one
# `judge.py` still reads by default. Deleting it would break an audit trail.
#
# TODO(retention): audit the exempt archive for identifiable clinical text.
#
#   Raised 2026-10-12 (UX Phase 1). An archive exempt from the window is exempt
#   from its only safeguard, so the exemption carries an obligation instead: this
#   file's contents must be shown to be synthetic or fully de-identified, because
#   nothing will ever remove them. It holds answers and the passages they were
#   grounded in, and it carries no timestamps, so no window can reach it and an
#   operator cannot tell its age from its contents.
#
#   Owner action, outside this script: confirm the corpus behind it is synthetic
#   or de-identified, record the finding, and either keep the exemption or lift
#   it. Until that is recorded, the exemption is an assumption rather than a
#   decision — and the assumption is the whole of the protection.
EXEMPT_TRACES = ("traces.jsonl",)


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


def _file_age_days(path: Path, now: datetime) -> int:
    """How long ago the archive was last written.

    The archive's own age, and the age that is enforced when any of its records
    cannot be aged: see `prune_trace_archives`.
    """
    written = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (now - written).days


def _entry(name: str, *, exempt: bool = False, age_days: int = 0,
           records: int = 0, kept: int = 0, dropped: int = 0,
           undated: int = 0, purged: bool = False) -> dict:
    """One line of the sweep's report.

    Every field stated rather than filled in by defaults at the call site, so
    what a report can say about an archive is visible in one place.
    """
    return {"file": name, "exempt": exempt, "age_days": age_days,
            "records": records, "kept": kept, "dropped": dropped,
            "undated": undated, "purged": purged}


def prune_trace_archives(
    directory: Path = None, *, days: int = TRACE_RETENTION_DAYS,
    now: datetime | None = None, apply: bool = False,
) -> list[dict]:
    """Apply the window to every trace archive in the results directory.

    Two levels of age, because the archives have two levels of provenance:

      * a record that says when it was written is aged by its own timestamp and
        dropped on its own;
      * an archive holding ANY record that cannot be aged is aged by the FILE's
        date instead, and the whole archive is purged once the file passes the
        window. A record with no timestamp cannot be kept indefinitely by
        default, and a window that covers only the archives that happen to be
        well formed is not a window.

    Exempt archives are listed, never read, and never deleted.

    One entry per file, in name order, so a report can say what it did to each
    one and what it could not do. The aggregate is a property of the returned
    list; a separate total is one more number that can disagree with the lines
    above it.
    """
    directory = Path(directory or DEFAULT_TRACE_DIR)
    now = now or datetime.now(timezone.utc)

    entries: list[dict] = []
    for path in sorted(directory.glob(TRACE_GLOB)):
        age_days = _file_age_days(path, now)
        if path.name in EXEMPT_TRACES:
            entries.append(_entry(path.name, exempt=True, age_days=age_days))
            continue

        # Measured without applying, so nothing is written before the decision:
        # an archive about to be purged is not rewritten on its way out.
        measured = prune_traces(path, days=days, now=now, apply=False)
        records = measured["kept"] + measured["dropped"]
        if measured["undated"] and age_days > days:
            if apply:
                path.unlink()
            entries.append(_entry(
                path.name, age_days=age_days, records=records,
                kept=measured["kept"], dropped=measured["dropped"],
                undated=measured["undated"], purged=True))
            continue
        if apply and measured["dropped"]:
            prune_traces(path, days=days, now=now, apply=True)
        entries.append(_entry(
            path.name, age_days=age_days, records=records,
            kept=measured["kept"], dropped=measured["dropped"],
            undated=measured["undated"]))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="perform the deletions; without it this is a dry run")
    parser.add_argument(
        "--holdings", choices=("both", "note-cache", "traces"), default="both")
    # Declared because the note-cache branch reads it: without this argument the
    # default dry run raised AttributeError before it printed anything, so the
    # note cache could not be reported or purged by this script at all. The
    # functions below always accepted a directory; only the CLI never offered one.
    parser.add_argument(
        "--note-cache-dir", type=Path, default=None,
        help="directory holding the note cache "
             f"(default: {CACHE_DIR}, or NOTE_CACHE_DIR when set)")
    parser.add_argument(
        "--trace-dir", type=Path, default=DEFAULT_TRACE_DIR,
        help=f"directory holding the archives (default: {DEFAULT_TRACE_DIR})")
    parser.add_argument(
        "--trace-retention-days", type=int, default=TRACE_RETENTION_DAYS)
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
        entries = prune_trace_archives(
            args.trace_dir, days=args.trace_retention_days, apply=args.apply)
        print(f"[{mode}] trace archives ({args.trace_dir}, {TRACE_GLOB}): "
              f"{len(entries)} file(s)")
        unswept: list[dict] = []
        for entry in entries:
            if entry["exempt"]:
                print(f"  exempt  {entry['file']} ({entry['age_days']}d) — "
                      f"immutable evaluation baseline, never pruned; "
                      f"audit owed (see EXEMPT_TRACES)")
                continue
            if entry["purged"]:
                print(f"  {'purged' if args.apply else 'would purge'} "
                      f"{entry['file']} ({entry['age_days']}d) — "
                      f"{entry['records']} record(s), none of them dated: "
                      f"the archive is past the "
                      f"{args.trace_retention_days}-day window")
                continue
            print(f"  {'dropped' if args.apply else 'would drop'} "
                  f"{entry['dropped']:5d} of {entry['records']:5d} "
                  f"in {entry['file']} ({entry['age_days']}d)")
            if entry["undated"]:
                unswept.append(entry)

        if unswept:
            # Said once, in full: these are inside the window, and they will be
            # deleted whole rather than record by record. An operator needs to
            # know which archives are on that clock and how long is left.
            print(f"\n  {len(unswept)} archive(s) hold records with no timestamp "
                  f"and are still inside the window. Each is aged as a FILE and "
                  f"purged whole when it passes {args.trace_retention_days} day(s):")
            for entry in unswept:
                left = args.trace_retention_days - entry["age_days"]
                print(f"    {entry['file']} ({entry['age_days']}d, "
                      f"{entry['undated']} undated record(s)) — "
                      f"{left} day(s) left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
