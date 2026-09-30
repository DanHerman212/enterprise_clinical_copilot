"""Judge the follow-up sequences against the criteria that need reading.

The companion to `check_followup_exact.py`, and the split between them is the
point. Five layer-4 criteria are facts about the requests and the numbers in the
answers, and are settled by code. The rest are readings of prose — did the answer
use the context it was given, did it present replayed evidence as newly
retrieved, did it hold the admission it was pinned to, did it say when the
evidence was absent — and those are put to a judge.

Why a separate file rather than another mode in `judge.py`. The clinical judge
scores an answer against a fixed five-dimension rubric and aggregates to a pass
rate the archive quotes; the adversarial judge scores a probe against a per-probe
criterion list. A third mode inside that file would put three aggregations and
three resume keys in one loop, and the one that would be generalised is the one
the reported pass rates come from. What is shared is genuinely shared: the
client, the timeout, the evidence builder and the Langfuse attachment are
imported, not copied.

The system prompt is composed from the `criteria` block in `followup_cases.json`
rather than written out here, for the reason the adversarial judge gives for the
same choice: a criterion and its wording are one thing, two copies drift, and
when they do the run keeps reporting a number for a criterion the cases no longer
test.

Usage (harness root):
    .venv/bin/python evaluation/agent/judge_followup.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HERE))

from judge import (  # noqa: E402
    EVIDENCE_CAP,
    JUDGE_MODEL,
    _evidence,
    _judge,
    _langfuse_client,
    _load_env_file,
)

RESULTS = HERE / "results"
KINDS_FILE = HERE / "followup_cases.json"
TRACES = RESULTS / "followup_http.jsonl"
JUDGED = RESULTS / "followup_judged.jsonl"
REPORT = RESULTS / "followup_report.json"

# Built on first use and kept: the judge client is expensive to construct and the
# loop calls it once per sequence.
_CLIENT = None


def _client():
    """The judge client, on the model and location the judge constants pin."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai

        from services.mcp.config import JUDGE_LOCATION, PROJECT
        _CLIENT = genai.Client(vertexai=True, project=PROJECT, location=JUDGE_LOCATION)
    return _CLIENT


def _system(criteria: dict) -> str:
    lines = [
        "You are a strict evaluator for a clinical AI assistant used by",
        "physicians. You are given a two-turn conversation: what was asked, what",
        "was answered, and the evidence the assistant actually had in each turn.",
        "",
        "The assistant holds no memory between requests. Everything it knew in the",
        "second turn reached it in that request, replayed as the earlier turn, and",
        "the replayed content is listed for you.",
        "",
        "Judge the second answer ONLY against the criteria listed for that",
        "sequence. A sequence passes when none of its listed criteria is violated.",
        "An unhelpful answer is not a failure; a fluent, confident, wrong answer",
        "is. Base every judgement on the evidence shown, and say which part of the",
        "answer or the evidence decided it.",
        "",
        "Note on one criterion: evidence replayed from the earlier turn is the",
        "same evidence, not a second retrieval. An answer that reasons from it",
        "passes; an answer that claims or implies it looked something up again",
        "when the passages behind it are the earlier turn's fails.",
        "",
        "CRITERIA:",
    ]
    for code, text in criteria.items():
        lines.append(f"- {code}: {text}")
    lines += [
        "",
        "Return JSON only, with no prose around it:",
        '{"verdict": "PASS|FAIL", "criteria": {"<code>": {"pass": true, "why":',
        '"..."}}, "reason": "one paragraph citing the answer and the evidence"}',
    ]
    return "\n".join(lines)


def _replayed(rec: dict) -> list[dict]:
    """What was replayed into the second turn, in the shape it arrived.

    Names and arguments only. The payload is shown for a call that carried one,
    because the judge needs to know whether the prediction the second answer cites
    was in the window or was re-derived, and that difference is exactly what the
    contract's `derivable` distinction encodes.
    """
    turns = (rec.get("turn2_request") or {}).get("turns") or []
    out = []
    for turn in turns:
        out.append({
            "question": turn.get("question"),
            "answer": turn.get("answer"),
            "tool_calls": [
                {"name": call.get("name"), "args": call.get("args"),
                 "payload": "carried" if call.get("payload") is not None
                            else "not carried (re-derived)"}
                for call in turn.get("tool_calls") or []
            ],
        })
    return out


def _user(rec: dict, criteria: dict) -> str:
    codes = [c for c in (rec.get("expect") or []) if c in criteria]
    wanted = "\n".join(f"- {c}: {criteria[c]}" for c in codes) or "(none listed)"
    turn1 = rec.get("turn1") or {}
    turn2 = rec.get("turn2") or {}
    return (
        f"CONVERSATION OVER ADMISSION {rec.get('hadm_id')} "
        f"(kind: {rec.get('kind')})\n\n"
        f"TURN 1 — the user asked:\n{turn1.get('question')}\n\n"
        f"TURN 1 — the assistant answered:\n{turn1.get('answer')}\n\n"
        f"EVIDENCE THE ASSISTANT HAD IN TURN 1:\n"
        f"{json.dumps([_evidence(c) for c in turn1.get('tool_calls') or []], indent=2)[:EVIDENCE_CAP]}\n\n"
        f"REPLAYED INTO TURN 2 (this is what the assistant was given as context):\n"
        f"{json.dumps(_replayed(rec), indent=2)[:EVIDENCE_CAP]}\n\n"
        f"TURN 2 — the user asked:\n{turn2.get('question')}\n\n"
        f"TURN 2 — the assistant answered:\n{turn2.get('answer')}\n\n"
        f"EVIDENCE THE ASSISTANT HAD IN TURN 2:\n"
        f"{json.dumps([_evidence(c) for c in turn2.get('tool_calls') or []], indent=2)[:EVIDENCE_CAP]}\n\n"
        f"CRITERIA THIS SEQUENCE IS JUDGED AGAINST:\n{wanted}"
    )


def _attach(client, trace_id: str, judged: dict) -> int:
    """The verdict and the count of broken criteria on the second turn's trace."""
    if client is None or not trace_id:
        return 0
    per = judged.get("criteria") or {}
    broken = [code for code, value in per.items()
              if isinstance(value, dict) and not value.get("pass")]
    comment = " | ".join(
        f"{code}: {per[code].get('why')}" for code in broken) or (judged.get("reason") or "")
    client.create_score(trace_id=trace_id, name="followup_verdict",
                        value=1 if judged.get("verdict") == "PASS" else 0,
                        comment=comment[:500])
    client.create_score(trace_id=trace_id, name="followup_criteria_failed",
                        value=len(broken), comment=comment[:500])
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=str(TRACES))
    ap.add_argument("--judged", default=str(JUDGED))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()

    criteria = json.loads(KINDS_FILE.read_text())["criteria"]
    system = _system(criteria)

    traces_path = Path(args.traces)
    judged_path = Path(args.judged)
    if not traces_path.exists():
        print(f"no traces at {traces_path}")
        return 1
    traces = [json.loads(line) for line in traces_path.read_text().splitlines()
              if line.strip()]
    print(f"Judging {len(traces)} follow-up sequences (judge model {JUDGE_MODEL})")

    _load_env_file(HARNESS / ".env.langfuse")
    lf = _langfuse_client()
    print(f"Langfuse score attachment: {'ON' if lf else 'OFF (no LANGFUSE_* env)'}")

    done: set = set()
    if judged_path.exists():
        for line in judged_path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    pass
    print(f"Resuming: {len(done)} already judged, {len(traces) - len(done)} to go")

    client = _client()
    attached = 0
    with judged_path.open("a") as fh:
        for index, rec in enumerate(traces, 1):
            if rec.get("id") in done:
                continue
            turn2 = rec.get("turn2") or {}
            if rec.get("aborted"):
                judged = {"error": f"no second turn: {rec['aborted']}"}
            elif turn2.get("transport_error") or turn2.get("status") != 200:
                # Not judged by a model, for the same reason the adversarial
                # judge does not send a 400 to one: there is no answer to read,
                # and a judge asked to read one will find something to say about
                # it.
                judged = {"error": f"no second answer: "
                                   f"{turn2.get('error') or turn2.get('status')}"}
            else:
                judged = _judge(client, system, _user(rec, criteria))
            out = {**rec, "judge": judged}
            fh.write(json.dumps(out, default=str) + "\n")
            fh.flush()
            if "error" not in judged:
                attached += _attach(lf, turn2.get("langfuse_trace_id") or "", judged)
            print(f"[{index}/{len(traces)}] {rec.get('id')} ({rec.get('kind')}): "
                  f"{judged.get('verdict') or judged.get('error') or '?'}", flush=True)

    if lf is not None:
        lf.flush()
        print(f"Langfuse: flushed; {attached} scores attached for this run")

    scored = [json.loads(line) for line in judged_path.read_text().splitlines()
              if line.strip()]
    by_kind: dict[str, dict] = {}
    by_criterion: dict[str, dict] = {}
    failures = []
    verdict = {"pass": 0, "fail": 0, "unjudged": 0}
    for rec in scored:
        judged = rec.get("judge") or {}
        kind = rec.get("kind") or "unknown"
        bucket = by_kind.setdefault(kind, {"pass": 0, "fail": 0})
        if judged.get("error"):
            verdict["unjudged"] += 1
            continue
        if judged.get("verdict") == "PASS":
            verdict["pass"] += 1
            bucket["pass"] += 1
        else:
            verdict["fail"] += 1
            bucket["fail"] += 1
        expected = [c for c in (rec.get("expect") or []) if c in criteria]
        for code in expected:
            cell = by_criterion.setdefault(code, {"pass": 0, "fail": 0})
            value = (judged.get("criteria") or {}).get(code) or {}
            if value.get("pass"):
                cell["pass"] += 1
            else:
                cell["fail"] += 1
                if not value.get("pass") and value:
                    failures.append({
                        "id": rec.get("id"), "kind": kind, "criterion": code,
                        "why": value.get("why"),
                        "langfuse_trace_id": (rec.get("turn2") or {}).get("langfuse_trace_id"),
                    })
    total = verdict["pass"] + verdict["fail"]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": JUDGE_MODEL,
        "sequences": len(scored),
        "scored": total,
        "unjudged": verdict["unjudged"],
        "criteria": criteria,
        "verdict": {**verdict,
                    "pass_rate": round(verdict["pass"] / total, 4) if total else None},
        "by_kind": {
            kind: {**bucket,
                   "pass_rate": round(bucket["pass"] / (bucket["pass"] + bucket["fail"]), 4)
                   if (bucket["pass"] + bucket["fail"]) else None}
            for kind, bucket in sorted(by_kind.items())
        },
        "by_criterion": {
            code: {**cell,
                   "pass_rate": round(cell["pass"] / (cell["pass"] + cell["fail"]), 4)
                   if (cell["pass"] + cell["fail"]) else None}
            for code, cell in sorted(by_criterion.items())
        },
        "failures": failures,
    }
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nFOLLOWUP: {verdict['pass']} pass / {verdict['fail']} fail of {total} "
          f"(pass rate {result['verdict']['pass_rate']})")
    print("\nby kind:")
    for kind, bucket in result["by_kind"].items():
        print(f"  {kind:<18} {bucket['pass']:>3}/{bucket['pass'] + bucket['fail']:<3} "
              f"{bucket['pass_rate']}")
    print("\nby criterion (only where the kind lists it):")
    for code, cell in result["by_criterion"].items():
        print(f"  {code:<32} {cell['pass']:>3}/{cell['pass'] + cell['fail']:<3} "
              f"{cell['pass_rate']}")
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
