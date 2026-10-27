"""The revision map is only worth anything if it joins on the right key.

These tests cover the two ways it can be wrong without failing loudly: parsing a line
that is not an execution record, and folding two records for one trace id as though one
revision had served it.
"""

from __future__ import annotations

import json

from evaluation.platform.backfill_revisions import fold, parse_execution

RECORD = {
    "event": "agent_execution",
    "trace": "06f1d0cd321cac74ff9a01bbfc3d582a",
    "code_revision": "agent-00206-xlm",
    "langfuse_trace_id": "a279f4a473632cc932be3102f14fd29e",
    "run_tag": "pilot-verify-20261027-215139",
    "outcome": "ok",
}

LINE = ("INFO 2026-10-27 22:19:38,633 services.agent.telemetry.observability "
        + json.dumps(RECORD))


def _entry(payload: str, revision: str = "agent-00206-xlm",
           ts: str = "2026-10-27T22:19:38Z") -> dict:
    return {"payload": payload, "resource": {"revision_name": revision}, "timestamp": ts}


def test_the_record_is_found_behind_the_logger_prefix():
    parsed = parse_execution(LINE)
    assert parsed is not None
    assert parsed["langfuse_trace_id"] == "a279f4a473632cc932be3102f14fd29e"
    assert parsed["code_revision"] == "agent-00206-xlm"


def test_an_access_log_line_is_not_a_record():
    assert parse_execution('INFO:     169.254.169.126:27298 - "POST /ask" 200 OK') is None


def test_a_json_line_that_is_not_an_execution_record_is_skipped():
    assert parse_execution('INFO {"event": "something_else"}') is None


def test_the_map_joins_on_the_trace_id_and_carries_both_identifiers():
    runs = fold([_entry(LINE)])

    bucket = runs["pilot-verify-20261027-215139"]
    entry = bucket["traces"]["a279f4a473632cc932be3102f14fd29e"]
    assert entry["code_revision"] == "agent-00206-xlm"
    assert entry["revision_name"] == "agent-00206-xlm"
    assert bucket["conflicts"] == []


def test_two_revisions_on_one_trace_id_are_reported_not_resolved():
    """A deploy landing mid-turn would produce this. Silently keeping the newer record
    would hide the one fact worth knowing."""
    older = dict(RECORD, code_revision="agent-00205-zvh")
    runs = fold([
        _entry("INFO " + json.dumps(older), revision="agent-00205-zvh",
               ts="2026-10-27T22:19:38Z"),
        _entry(LINE, revision="agent-00206-xlm", ts="2026-10-27T22:20:38Z"),
    ])

    bucket = runs["pilot-verify-20261027-215139"]
    assert bucket["traces"]["a279f4a473632cc932be3102f14fd29e"]["revision_name"] \
        == "agent-00206-xlm"
    assert len(bucket["conflicts"]) == 1
    assert bucket["conflicts"][0]["seen"] == ["agent-00205-zvh", "agent-00206-xlm"]


def test_a_trace_without_a_run_tag_is_kept_under_a_named_bucket():
    """Live traffic has no run tag; dropping it would quietly exclude production."""
    live = dict(RECORD)
    live.pop("run_tag")
    runs = fold([_entry("INFO " + json.dumps(live))])

    assert "(no run tag)" in runs
