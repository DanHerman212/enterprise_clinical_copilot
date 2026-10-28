"""Local, un-truncated record of every item an evaluation run judges.

Langfuse keeps a score comment at 500 characters and the composite caps its own verdict
comment at 1200, which is enough to read a verdict and not enough to diagnose one. The
`00e` post-mortem could not answer "which passage did `^[2]` point at" from the stored
run at all: the tool payloads of a free-text turn are not exported as spans, so the
numbering the model was shown existed only in the harness's memory during the run.

So each judged item also writes one JSON line here — question, answer, published sources,
every tool call with its response, every dimension score, the verdict, and the judge's
FULL reasoning. It is a write-only sink: nothing in the harness reads it back, and a run
that never calls `configure` writes nothing.

The path is a run artifact, not source. `evaluation/results/runs/` is ignored by git.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_SINK: Path | None = None

#: Where a run writes unless told otherwise. Ignored by git; see `.gitignore`.
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "results" / "runs"


def configure(path: str | os.PathLike[str] | None) -> Path | None:
    """Point the sink at `path` and start a fresh file, or disable it with None.

    Truncating rather than appending: a re-run of the same name is a different run, and
    interleaving two would make the file unattributable.
    """
    global _SINK
    with _LOCK:
        if path is None:
            _SINK = None
            return None
        sink = Path(path)
        sink.parent.mkdir(parents=True, exist_ok=True)
        sink.write_text("", encoding="utf-8")
        _SINK = sink
        return sink


def configured() -> Path | None:
    """The file this process is writing, or None."""
    return _SINK


def _turns(input_doc: dict[str, Any], output: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per turn: what was asked, what came back, what the reader can click."""
    questions = [str(q) for q in (input_doc.get("turns") or [])]
    answers = output.get("turn_responses")
    if not isinstance(answers, list) or not answers:
        answers = [output.get("turn_1_response") or ""]
    sources = output.get("turn_sources")
    if not isinstance(sources, list):
        sources = []
    calls = output.get("turn_tool_calls")
    if not isinstance(calls, list):
        calls = []
    out: list[dict[str, Any]] = []
    for index in range(max(len(questions), len(answers))):
        out.append({
            "turn": index + 1,
            "question": questions[index] if index < len(questions) else "",
            "answer": answers[index] if index < len(answers) else "",
            "sources": sources[index] if index < len(sources) else [],
            "tool_calls": calls[index] if index < len(calls) else [],
        })
    return out


def _item_id(input_doc: dict[str, Any]) -> str:
    """The scenario id, from whichever field the dataset item carried it in."""
    for key in ("id", "scenario", "scenario_id", "case_id"):
        value = input_doc.get(key)
        if value:
            return str(value)
    return ""


def record(*, input: dict[str, Any], output: dict[str, Any],
           result: dict[str, Any] | None, reason: str = "judged") -> None:
    """Append one item's diagnostics. A no-op when no sink is configured."""
    sink = _SINK
    if sink is None:
        return
    result = result or {}
    entry = {
        "item_id": _item_id(input or {}),
        "reason": reason,
        "manifest": (input or {}).get("patient_manifest") or {},
        "turns": _turns(input or {}, output or {}),
        "dimensions": result.get("dimensions") or {},
        "verdict": 1 if str(result.get("verdict") or "").upper() == "PASS" else 0,
        "flags": [str(flag) for flag in (result.get("flags") or [])],
        # Un-truncated: this is the whole reason the file exists.
        "judge_comment": str(result.get("reasoning") or result.get("reason") or ""),
        "samples": result.get("_samples"),
        "trace_ids": (output or {}).get("turn_trace_ids") or [],
    }
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _LOCK:
        with sink.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
