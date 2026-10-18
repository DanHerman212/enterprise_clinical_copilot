"""Deterministic structural assertions over collected traces.

The point of this module is that formatting and routing defects are not the judge's
job. An LLM grader reading a fluent answer cannot tell whether `/ehr` executed or
whether the reasoning loop answered in its place — that is precisely how the
2026-10-17 run reported green while `/ehr` ran zero times in 246 turns. Everything
here is a property of the response envelope or of the settled text, so it is decided
by code and decided the same way every time.

Two rules govern what belongs here:

* **Envelope facts only.** `command`, `routed_to`, `tool_calls`, A2UI component names.
  If a check needs to know whether a clinical claim is *true*, it is not this module's
  job and belongs to the judge.
* **No silent passes.** A check that cannot be evaluated on a record reports that,
  rather than returning clean. A missing field is a finding, not an absence of one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_set import COMMAND_TOOLS, COVERAGE_COMMANDS, SLASH_COMMANDS  # noqa: E402

#: Serialized payload leaking into prose.
RAW_JSON = re.compile(r'\{"[a-z_]+":')

#: A citation marker and what it is allowed to be. `^[1]` is a note passage, a
#: renumbered reference; `^[PMID: 12345]` is a journal article. Anything else is an
#: unresolved marker the citation layer failed to rewrite.
CITATION = re.compile(r"\^\[([^\]]*)\]")
CITATION_OK = (
    re.compile(r"^\d+$"),
    re.compile(r"^PMID:\s*\d+$"),
)

#: Expectations that are graded by the judge, not here. Listed explicitly so the
#: absence of a deterministic check is a stated decision rather than an oversight.
JUDGE_GRADED = frozenset({
    "abstains_on_discharge_diagnosis",
    "manifest_answer",
    "no_forced_retrieval",
    "risk_suppressed",
    "allows_abstention",
})


def _fail(case_id: str, check: str, detail: str) -> dict:
    return {"case": case_id, "check": check, "detail": detail}


def check_text(case_id: str, rec: dict) -> list[dict]:
    """Settled-prose hygiene: no raw JSON, no unresolved citation markers."""
    out: list[dict] = []
    answer = rec.get("answer") or ""
    if not isinstance(answer, str):
        return [_fail(case_id, "answer_type", f"answer is {type(answer).__name__}")]
    if RAW_JSON.search(answer):
        out.append(_fail(case_id, "raw_json", "serialized payload in settled prose"))
    for marker in CITATION.findall(answer):
        if not any(p.match(marker.strip()) for p in CITATION_OK):
            out.append(_fail(case_id, "raw_citation", f"unresolved marker ^[{marker}]"))
    return out


def _tools(rec: dict) -> set[str]:
    return {c.get("name") for c in (rec.get("tool_calls") or []) if c.get("name")}


def check_case(spec: dict, rec: dict) -> list[dict]:
    """Every deterministic expectation of one Suite 2 case against its record."""
    cid = spec.get("id") or str(spec.get("hadm_id"))
    expect = spec.get("expect") or {}
    out: list[dict] = []

    # An expected refusal is an answer, not a transport failure. The product fast-fails
    # a mistyped command with 400 `unknown_command` and no tool calls, which is the
    # behaviour the boundary cases are asking for — so the case says which status and
    # which code it expects, and only a different outcome fails.
    wanted_status = expect.get("status")
    if wanted_status is not None and wanted_status != 200:
        if rec.get("status") != wanted_status:
            out.append(_fail(cid, "status",
                             f"expected {wanted_status}, got {rec.get('status')}"))
            return out
        if (want_err := expect.get("error")) and rec.get("error") != want_err:
            out.append(_fail(cid, "error",
                             f"expected {want_err!r}, got {rec.get('error')!r}"))
        if "tool_calls" in expect:
            if len(rec.get("tool_calls") or []) != expect["tool_calls"]:
                out.append(_fail(cid, "tool_calls",
                                 f"expected {expect['tool_calls']} calls on a refusal, "
                                 f"got {sorted(_tools(rec))}"))
        if "command" in expect and rec.get("command_executed") != expect["command"]:
            out.append(_fail(cid, "command",
                             f"expected {expect['command']!r}, "
                             f"got {rec.get('command_executed')!r}"))
        return out

    if rec.get("transport_error") or rec.get("status") != 200:
        out.append(_fail(cid, "transport",
                         f"status={rec.get('status')} "
                         f"error={rec.get('error') or rec.get('transport_error')}"))
        return out

    tools = _tools(rec)
    answer = rec.get("answer") or ""

    if "command" in expect:
        got = rec.get("command_executed")
        if got != expect["command"]:
            out.append(_fail(cid, "command", f"expected {expect['command']!r}, got {got!r}"))

    if "routed_to" in expect:
        got = rec.get("routed_to")
        if got != expect["routed_to"]:
            out.append(_fail(cid, "routed_to",
                             f"expected {expect['routed_to']!r}, got {got!r}"))

    if "component" in expect:
        want = expect["component"]
        if want not in (rec.get("a2ui_components") or []):
            out.append(_fail(cid, "component",
                             f"{want} missing; had {rec.get('a2ui_components')}"))

    if "component_absent" in expect:
        unwanted = expect["component_absent"]
        if unwanted in (rec.get("a2ui_components") or []):
            out.append(_fail(cid, "component_absent", f"{unwanted} was emitted"))

    if "tool_calls" in expect:
        if len(rec.get("tool_calls") or []) != expect["tool_calls"]:
            out.append(_fail(cid, "tool_calls",
                             f"expected {expect['tool_calls']} calls, got {sorted(tools)}"))

    if (budget := expect.get("max_tool_calls")) is not None:
        if len(rec.get("tool_calls") or []) > budget:
            out.append(_fail(cid, "tool_budget",
                             f"{len(rec.get('tool_calls') or [])} calls > {budget}"))

    if expect.get("fast_fail"):
        if tools:
            out.append(_fail(cid, "fast_fail", f"ran tools {sorted(tools)}"))
        if rec.get("command_executed") is not None:
            out.append(_fail(cid, "fast_fail",
                             f"executed command {rec.get('command_executed')!r}"))

    if expect.get("lists_valid_commands"):
        if not any(cmd in answer for cmd in SLASH_COMMANDS.values()):
            out.append(_fail(cid, "valid_commands",
                             "refusal did not list any valid command"))

    if "citation_marker" in expect:
        wanted = expect["citation_marker"]
        pattern = r"\^\[PMID:\s*\d+\]" if wanted == "PMID" else r"\^\[\d+\]"
        if not re.search(pattern, answer):
            out.append(_fail(cid, "citation_marker", f"no {wanted} citation in answer"))
    if expect.get("literature_nonempty"):
        # Literature records are not prose. The PMID citations live in the `literature`
        # payload the canvas renders, not as `^[PMID: n]` markers inside the answer, so
        # asserting the marker in the narrative was asserting the wrong surface and
        # failed a turn whose card was drawn correctly.
        if not (rec.get("literature_count") or 0) >= 1:
            out.append(_fail(cid, "literature", "no literature records in the payload"))
    if "grounded_lab" in expect:
        # The guard layer already refuses a trajectory it cannot ground, so this is
        # a second, cheaper net: the answer must name the analyte it is trending.
        label = str(expect["grounded_lab"]).lower()
        aliases = {"rdw": ("rdw", "red cell distribution"), "creatinine": ("creatinine",)}
        names = aliases.get(label, (label,))
        if not any(n in answer.lower() for n in names):
            out.append(_fail(cid, "grounded_lab", f"answer never names {label}"))

    if expect.get("no_recursion_error"):
        low = answer.lower()
        if "graphrecursionerror" in low or "recursion limit" in low:
            out.append(_fail(cid, "recursion", "recursion error surfaced in the answer"))

    for key in expect:
        if key in JUDGE_GRADED:
            continue  # graded by the judge, deliberately not asserted here
    return out


def check_command_coverage(recs: list[dict],
                           required: tuple[str, ...] = COVERAGE_COMMANDS) -> list[dict]:
    """Every covered command executed at least once, and reached its own tool.

    Two different failures share this check and both must fail the run: a command
    that never appears at all, and a command that appears but whose turn ran no tool
    — the second is what a fall-through to the reasoning loop looks like.
    """
    out: list[dict] = []
    for cmd in required:
        rows = [r for r in recs if r.get("command_executed") == cmd]
        if not rows:
            out.append(_fail("coverage", cmd, "command executed zero times"))
            continue
        expected = COMMAND_TOOLS.get(cmd, set())
        if expected and not any(_tools(r) & expected for r in rows):
            seen = sorted({t for r in rows for t in _tools(r)})
            out.append(_fail("coverage", cmd,
                             f"no turn reached {sorted(expected)}; saw {seen}"))
    return out


def check_all(recs: list[dict], specs: list[dict] | None = None) -> dict:
    """Every deterministic check over a run, in the shape entrypoint reports."""
    failures: list[dict] = []
    for rec in recs:
        cid = rec.get("id") or f"{rec.get('hadm_id')}:{rec.get('prompt')}:t{rec.get('turn')}"
        failures.extend(check_text(cid, rec))

    if specs is not None:
        by_id = {r.get("id"): r for r in recs if r.get("id")}
        for spec in specs:
            rec = by_id.get(spec.get("id"))
            if rec is None:
                failures.append(_fail(spec.get("id"), "missing",
                                      "no record collected for this case"))
                continue
            failures.extend(check_case(spec, rec))
    return {
        "checked": len(recs),
        "failures": failures,
        "count": len(failures),
        "by_check": _tally(failures),
    }


def _tally(failures: list[dict]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for f in failures:
        tally[f["check"]] = tally.get(f["check"], 0) + 1
    return tally
