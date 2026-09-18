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
import sys
from datetime import datetime, timedelta, timezone
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
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


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


def test_the_collector_writes_the_timestamp_it_will_be_pruned_by():
    """The archive's lifetime is enforced through this field, so the writer and
    the pruner are checked against each other rather than assumed to agree."""
    source = (Path(__file__).resolve().parents[2] / "evaluation" / "agent"
              / "collect.py").read_text(encoding="utf-8")

    assert '"at": datetime.now(timezone.utc).isoformat()' in source
    assert retention._record_time({"at": "2026-09-18T12:00:00+00:00"}) is not None


def test_the_pruner_and_the_collector_point_at_the_same_file():
    """A window applied to a file nobody writes is not a retention policy.

    The collector computes its output path from its own location, and this
    module keeps its own copy of that path rather than importing the collector
    (which imports the agent graph to be read). That is a drift surface, and it
    had already drifted once: the collector's `parents[1]` resolved to a path
    that did not exist after the file moved, so a run would have written the
    archive somewhere .gitignore does not cover.
    """
    source = (Path(__file__).resolve().parents[2] / "evaluation" / "agent"
              / "collect.py").read_text(encoding="utf-8")

    assert 'RESULTS = Path(__file__).resolve().parent / "results"' in source
    assert 'OUT = RESULTS / "traces.jsonl"' in source
    assert retention.DEFAULT_TRACE_FILE == (
        Path(__file__).resolve().parents[2] / "evaluation" / "agent" / "results"
        / "traces.jsonl")


def test_the_archive_path_the_collector_uses_is_ignored_by_git():
    """The archive holds clinical text: answers and the passages behind them.

    It was ignored at its old home and not at the path the collector had drifted
    to, which is the difference between a retention window and a file that can be
    committed.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    path = "evaluation/agent/results/traces.jsonl"
    result = subprocess.run(
        ["git", "check-ignore", "-v", path], cwd=repo,
        capture_output=True, text=True)

    assert result.returncode == 0, (
        f"{path} is not ignored; a collector run would leave clinical text "
        f"untracked but committable")
    assert "evaluation/agent/results/" in result.stdout
