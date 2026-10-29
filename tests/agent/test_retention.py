"""The two holdings that are not conversations, and the lifetimes they were given.

The conversation store's sweep is tested on the site side. This file covers the
other two: the note cache (ephemeral, regenerable, deleted in full) and the
evaluation trace archive (a fixed window, because it is the evidence behind an
evaluation).

The tests that matter are the ones about not deleting something nobody decided to
delete: a dry run performs no deletion, a record with no timestamp is kept and
counted, and a truncated line is kept rather than silently treated as rubbish.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.agent import retention  # noqa: E402

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)


def _record(at=None, prompt="risk"):
    rec = {"hadm_id": 90000009, "prompt": prompt, "answer": "…"}
    if at is not None:
        rec["at"] = at
    return json.dumps(rec)


def _archive(tmp_path, lines):
    """One archive at the current collector's filename.

    Named for the collector rather than for `traces.jsonl`, which is the exempt
    baseline: a fixture that wrote that name would be writing the one file the
    sweep is not allowed to touch.
    """
    return _write(tmp_path / "traces_http.jsonl", lines)


def _write(path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


# The clock the archive-age tests use. Given rather than read, so a test about a
# ninety-day window does not change meaning with the calendar.
ARCHIVE_NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _dated(path, days_ago):
    """Backdate an archive, and return the clock the sweep should be given.

    The file is what is dated because the file's age is what the window acts on
    when its records cannot be aged.
    """
    written = (ARCHIVE_NOW - timedelta(days=days_ago)).timestamp()
    os.utime(path, (written, written))
    return ARCHIVE_NOW


def test_a_recent_record_is_kept_and_an_old_one_is_dropped(tmp_path):
    recent = (NOW - timedelta(days=5)).isoformat()
    old = (NOW - timedelta(days=120)).isoformat()
    path = _archive(tmp_path, [_record(old), _record(recent)])

    result = retention.prune_traces(path, days=90, now=NOW, apply=True)

    assert result == {"kept": 1, "dropped": 1, "undated": 0, "missing": False}
    kept = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(kept) == 1
    assert json.loads(kept[0])["at"] == recent


def test_the_window_is_the_only_thing_that_drops(tmp_path):
    """A record exactly at the cutoff stays: the promise is "at least N days"."""
    at_cutoff = (NOW - timedelta(days=90)).isoformat()
    path = _archive(tmp_path, [_record(at_cutoff)])

    result = retention.prune_traces(path, days=90, now=NOW, apply=True)

    assert result["dropped"] == 0
    assert result["kept"] == 1


def test_a_dry_run_deletes_nothing(tmp_path):
    old = (NOW - timedelta(days=400)).isoformat()
    path = _archive(tmp_path, [_record(old)])
    before = path.read_text(encoding="utf-8")

    result = retention.prune_traces(path, days=90, now=NOW, apply=False)

    assert result["dropped"] == 1, "the dry run still reports what it would do"
    assert path.read_text(encoding="utf-8") == before, "and does nothing"


def test_a_record_with_no_timestamp_is_kept_and_counted(tmp_path):
    """The archive predates its own timestamps.

    A record that does not say when it was written cannot be aged. Deleting it
    would be a decision nobody made; keeping it silently would hide that the
    window cannot be applied to part of the archive. It is kept, counted, and
    reported.
    """
    path = _archive(tmp_path, [_record(), _record((NOW - timedelta(days=500)).isoformat())])

    result = retention.prune_traces(path, days=90, now=NOW, apply=True)

    assert result["undated"] == 1
    assert result["dropped"] == 1
    assert result["kept"] == 1
    assert "at" not in json.loads(path.read_text(encoding="utf-8"))


def test_a_truncated_line_is_kept_not_treated_as_rubbish(tmp_path):
    """A half-written line is the record of a crash. Deleting it loses that."""
    path = _archive(tmp_path, ['{"hadm_id": 90000009, "prompt": "ri'])

    result = retention.prune_traces(path, days=90, now=NOW, apply=True)

    assert result["kept"] == 1
    assert result["undated"] == 1
    assert path.read_text(encoding="utf-8").strip().startswith('{"hadm_id"')


def test_a_missing_archive_is_not_an_error(tmp_path):
    result = retention.prune_traces(tmp_path / "nope.jsonl", days=90, now=NOW)

    assert result["missing"] is True


def test_the_note_cache_is_deleted_in_full_and_only_when_asked(tmp_path):
    (tmp_path / "discharge_test_split.jsonl.gz").write_bytes(b"notes")
    (tmp_path / "chunks.jsonl.gz").write_bytes(b"chunks")

    listed = retention.purge_note_cache(tmp_path, apply=False)
    assert len(listed) == 2, "the dry run reports the files"
    assert len(retention.note_cache_state(tmp_path)) == 2, "and deletes none"

    retention.purge_note_cache(tmp_path, apply=True)
    assert retention.note_cache_state(tmp_path) == []


def test_an_absent_note_cache_is_empty_rather_than_an_error(tmp_path):
    assert retention.note_cache_state(tmp_path / "not-fetched") == []


@pytest.mark.parametrize("raw", ["", "not a date", "2026-13-45T99:00:00"])
def test_an_unparseable_timestamp_counts_as_undated(raw):
    assert retention._record_time({"at": raw}) is None


def test_a_timestamp_without_a_zone_is_read_as_utc():
    assert retention._record_time({"at": "2026-09-18T12:00:00"}).tzinfo == timezone.utc


def test_the_archive_retention_sweeps_is_inside_the_sweep_pattern():
    """A window applied to a file the pattern misses is not a retention policy.

    This used to grep the collector's source to keep the writer and the pruner in
    agreement. The collector was retired on 2026-10-29, so the writer is no longer a
    module this suite can read — but the property that mattered is still checkable,
    and it is the one that actually failed twice: the archive the pruner names must
    fall inside the pattern it sweeps, and must not be exempt from it.
    """
    assert fnmatch(retention.DEFAULT_TRACE_FILE.name, retention.TRACE_GLOB)
    assert retention.DEFAULT_TRACE_FILE.name not in retention.EXEMPT_TRACES


def test_the_sweep_reaches_every_archive_and_leaves_the_baseline_alone(tmp_path):
    """One window, every archive — and the one archive that is meant to survive.

    The failure this prevents was live: the window named a single file, so sixteen
    archives of clinical text sat outside it while the report said the holding was
    covered.
    """
    recent = (NOW - timedelta(days=5)).isoformat()
    old = (NOW - timedelta(days=120)).isoformat()
    sweepable = ("traces_http.jsonl", "traces_http_2026-07-01.jsonl",
                 "traces_hybrid_108.s3.jsonl")
    for name in sweepable:
        _write(tmp_path / name, [_record(old), _record(recent)])
    # The immutable baseline: old, and not the window's to touch.
    _write(tmp_path / "traces.jsonl", [_record(old)])

    entries = retention.prune_trace_archives(
        tmp_path, days=90, now=NOW, apply=True)

    by_file = {entry["file"]: entry for entry in entries}
    assert set(by_file) == set(sweepable) | {"traces.jsonl"}
    for name in sweepable:
        assert by_file[name]["dropped"] == 1, name
        assert by_file[name]["kept"] == 1, name
        assert by_file[name]["exempt"] is False, name
    # Exempt means untouched: not read, not rewritten, still both records.
    assert by_file["traces.jsonl"]["exempt"] is True
    assert by_file["traces.jsonl"]["dropped"] == 0
    assert (tmp_path / "traces.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_an_undated_archive_older_than_the_window_is_purged_whole(tmp_path):
    """Compliance does not grant amnesty to a format.

    An archive whose records carry no timestamp cannot be kept indefinitely by
    default: the file's own date is the age that is enforced, and the archive goes
    as a unit rather than record by record.
    """
    path = _write(tmp_path / "traces_hybrid_108.jsonl", [_record(), _record()])
    now = _dated(path, 200)

    entries = retention.prune_trace_archives(
        tmp_path, days=90, now=now, apply=True)

    entry = entries[0]
    assert entry["undated"] == 2
    assert entry["records"] == 2
    assert entry["age_days"] == 200
    assert entry["purged"] is True
    assert not path.exists()


def test_an_undated_archive_inside_the_window_is_kept_and_its_clock_is_reported(
    tmp_path,
):
    """Inside the window it is kept whole — and the report says how long is left,
    because a purge an operator did not see coming is one they cannot prevent."""
    path = _write(tmp_path / "traces_hybrid_108.jsonl", [_record(), _record()])
    now = _dated(path, 49)

    entries = retention.prune_trace_archives(
        tmp_path, days=90, now=now, apply=True)

    entry = entries[0]
    assert entry["purged"] is False
    assert entry["undated"] == 2
    assert entry["kept"] == 2
    assert entry["age_days"] == 49
    assert path.exists()


def test_a_dry_run_purges_nothing(tmp_path):
    """The deletion is gated on --apply, and the dry run still reports it: a sweep
    that deleted before its report would be a sweep nobody could review."""
    path = _write(tmp_path / "traces_hybrid_108.jsonl", [_record()])
    now = _dated(path, 200)

    entries = retention.prune_trace_archives(
        tmp_path, days=90, now=now, apply=False)

    assert entries[0]["purged"] is True   # what a run would do
    assert path.exists()                  # and what this one did not


def test_an_exempt_archive_survives_the_file_clock(tmp_path):
    """The exemption has to outrank the file clock, or it is not an exemption.

    The baseline is both undated and old, which is exactly the combination the
    window now acts on — so this is the case the exemption exists for, and the
    case where getting the order of checks wrong destroys an audit trail.
    """
    path = _write(tmp_path / "traces.jsonl", [_record()])
    now = _dated(path, 400)

    entries = retention.prune_trace_archives(
        tmp_path, days=90, now=now, apply=True)

    assert entries[0]["exempt"] is True
    assert entries[0]["purged"] is False
    assert entries[0]["age_days"] == 400
    assert path.exists()


def test_the_archive_path_the_collector_uses_is_ignored_by_git():
    """The archive holds clinical text: answers and the passages behind them.

    It was ignored at its old home and not at the path the collector had drifted
    to, which is the difference between a retention window and a file that can be
    committed.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    path = "evaluation/agent/results/traces_http.jsonl"
    result = subprocess.run(
        ["git", "check-ignore", "-v", path], cwd=repo,
        capture_output=True, text=True)

    assert result.returncode == 0, (
        f"{path} is not ignored; a collector run would leave clinical text "
        f"untracked but committable")
    assert "evaluation/agent/results/" in result.stdout
