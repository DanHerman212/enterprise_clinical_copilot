"""LLM-as-judge over collected traces against the golden rubric -> report.

For each trace, send the question + agent answer + the evidence the agent
actually had (tool outputs + retrieved passages) to Gemini, which applies the
versioned rubric (eval/rubric.md) and returns JSON scores per dimension.
Aggregates into a pass-rate report. This is the qualitative half of the gate.

Usage (harness root):
    .venv/bin/python eval/judge.py
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

from mcp_server.config import PROJECT, LOCATION, GEMINI_MODEL  # noqa: E402

TRACES = HARNESS / "eval" / "results" / "traces.jsonl"
RUBRIC = (HARNESS / "eval" / "rubric.md").read_text()
JUDGED = HARNESS / "eval" / "results" / "judged.jsonl"
REPORT = HARNESS / "eval" / "results" / "golden_report.json"

DIMS = ["faithfulness", "groundedness", "citation", "clinical", "safety"]

# A single judge call can hang forever on a stuck Gemini request (observed
# 2026-08-19: the run stalled ~18min at trace 117/300 with no output). The
# timeout makes a hang raise (-> retried, then flagged) instead of stalling the
# whole judge. Mirrors collect.py's per-ask timeout.
_JUDGE_TIMEOUT_SECONDS = 120

# P2 fix (2026-08-14, after human judge-validation showed kappa=0): the judge
# must see the FULL evidence the agent saw. v1 truncated each passage to 400
# chars (the redacted header + HPI opening), so it could not verify meds/course
# that live in the body+end of each ~11k-char passage and falsely flagged
# faithful answers as "fabricated". Full passages, generous cap.
#
# P3.1 fix (2026-08-17, after golden re-run root-cause): some discharge notes are
# up to ~32k chars and the "Discharge Medications:" list lives at the END of the
# passage (past the old 20k cap). A 20k per-passage cap silently hid the med list,
# so the judge flagged FAITHFUL med answers as "all medications invented /
# hallucinated" (7 false safety failures in the Aug-17 run: 21508795, 26329920,
# 29318404, 21635816, 24592634). Max section length measured across all 300
# traces = 32105; max total evidence = 128420. Caps raised with margin so the
# judge always sees the med list. POC re-judge of the 7 artifact pairs confirmed
# they flip to PASS with the full passage.
EVIDENCE_CAP = 200000
PER_PASSAGE_CAP = 40000


# --- Langfuse score attachment (optional; no-op without keys) ---------------
# judge.py writes judged.jsonl as the durable archive AND attaches the rubric
# scores to the matching Langfuse trace (keyed by the trace id collect.py
# recorded when Langfuse was enabled), so the fix-and-retest loop can browse
# scored traces in the Langfuse UI. Runs without Langfuse env behave exactly as
# before — scoring is purely additive.

def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ.

    `.env.lanfuse` is `KEY=VALUE` with no `export`, so bash `source` only sets
    shell-local vars that standalone `python eval/judge.py` never sees. Loading
    here (mirrors run_eval_parallel.py) makes score attachment work when the
    judge is run on its own.
    """
    if not path.exists():
        return
    loaded = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and not os.environ.get(k):
            os.environ[k] = v
            loaded += 1
    if loaded:
        print(f"langfuse: loaded {loaded} vars from {path.name}", flush=True)


def _langfuse_client():
    env = {k: os.environ.get(k) for k in
           ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
    if not all(env.values()):
        return None
    from langfuse import Langfuse
    return Langfuse(host=env["LANGFUSE_HOST"],
                    public_key=env["LANGFUSE_PUBLIC_KEY"],
                    secret_key=env["LANGFUSE_SECRET_KEY"])


def _attach_scores(client, trace_id: str, j: dict) -> int:
    """Attach the judge verdict + per-dimension scores to a Langfuse trace.

    Returns the number of scores attached (0 when no client or trace id).
    """
    if client is None or not trace_id:
        return 0
    dims = j.get("dimensions") or {}
    flags = " | ".join(j.get("flags") or []) or (j.get("reason") or "")[:300]
    n = 0
    for dim in DIMS:
        if dim in dims:
            client.create_score(trace_id=trace_id, name=dim, value=dims[dim], comment=flags)
            n += 1
    client.create_score(trace_id=trace_id, name="verdict",
                        value=1 if j.get("verdict") == "PASS" else 0, comment=flags)
    return n + 1


def _evidence(tc: dict) -> dict:
    """Full, readable evidence from a tool call (v2: untruncated passages)."""
    name = tc.get("name")
    resp = tc.get("response") or {}
    if name == "predict_readmission":
        return {
            "tool": name,
            "probability": resp.get("probability"),
            "threshold": resp.get("threshold"),
            "top_factors": resp.get("top_factors"),
        }
    if name in ("rag_search", "rag_search_sections"):
        passages = resp.get("passages") or []
        return {
            "tool": name,
            "query": resp.get("query"),
            "passages": [
                {"section": p.get("section"), "text": (p.get("text") or "")[:PER_PASSAGE_CAP]}
                for p in passages
            ],
        }
    return {"tool": name, "response": str(resp)[:2000]}


def _judge_once(client, system: str, user: str) -> dict:
    """One synchronous judge call to Gemini (no timeout on this client API)."""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    txt = (resp.text or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`").removeprefix("json").strip()
    return json.loads(txt)


def _judge(client, system: str, user: str) -> dict:
    """Judge with a hard timeout so a stuck Gemini call cannot stall the run.

    The google-genai client does NOT accept a `timeout` kwarg on
    generate_content, so enforce the bound by running the call in a daemon
    thread and waiting on a queue. On timeout the daemon thread is abandoned
    (it never blocks process exit) and the attempt is retried, then flagged.
    """
    for attempt in range(4):
        q: queue.Queue[dict] = queue.Queue(maxsize=1)
        t = threading.Thread(
            target=lambda: q.put(_judge_once(client, system, user)), daemon=True)
        t.start()
        try:
            return q.get(timeout=_JUDGE_TIMEOUT_SECONDS)
        except queue.Empty:
            last = "timeout"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < 3:
            time.sleep(2 ** attempt)
    return {"error": str(last)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="traces_path", type=str, default=str(TRACES),
                    help="traces JSONL to judge (default: traces.jsonl)")
    ap.add_argument("--out", dest="judged_path", type=str, default=str(JUDGED),
                    help="judged JSONL to append (default: judged.jsonl)")
    ap.add_argument("--report", dest="report_path", type=str, default=str(REPORT),
                    help="report JSON to write (default: golden_report.json)")
    args = ap.parse_args()
    traces_path = Path(args.traces_path)
    judged_path = Path(args.judged_path)
    report_path = Path(args.report_path)

    _load_env_file(HARNESS / ".env.lanfuse")
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    lf = _langfuse_client()
    traces = [json.loads(l) for l in traces_path.read_text().splitlines() if l.strip()]
    print(f"Judging {len(traces)} traces (model {GEMINI_MODEL})")
    print(f"Langfuse score attachment: {'ON' if lf else 'OFF (no LANGFUSE_* env)'}")

    # Resumable: skip (hadm_id, prompt) pairs already scored in judged.jsonl and
    # append, so a re-run after a crash continues instead of restarting.
    done: set[tuple] = set()
    if judged_path.exists():
        for line in judged_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec.get("hadm_id"), rec.get("prompt")))
            except json.JSONDecodeError:
                pass
    print(f"Resuming: {len(done)} already judged, {len(traces) - len(done)} to go")

    attached = 0
    with judged_path.open("a") as fh:
        for i, t in enumerate(traces, 1):
            if (t.get("hadm_id"), t.get("prompt")) in done:
                continue
            if "error" in t:
                fh.write(json.dumps({**t, "judge": {"error": "agent run failed"}}) + "\n")
                print(f"[{i}/{len(traces)}] {t.get('hadm_id')}/{t.get('prompt')}: AGENT-ERROR", flush=True)
                continue

            evidence = [_evidence(tc) for tc in (t.get("tool_calls") or [])]
            user = (
                f"QUESTION:\n{t['question']}\n\n"
                f"ANSWER:\n{t['answer']}\n\n"
                f"EVIDENCE (tool outputs + retrieved passages the agent had):\n"
                f"{json.dumps(evidence, indent=2)[:EVIDENCE_CAP]}"
            )
            j = _judge(client, RUBRIC, user)
            rec = {**t, "judge": j}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if "error" not in j:
                attached += _attach_scores(lf, t.get("langfuse_trace_id") or "", j)
            print(f"[{i}/{len(traces)}] {t['hadm_id']}/{t['prompt']}: {j.get('verdict') or '?'}", flush=True)

    if lf is not None:
        lf.flush()
        print(f"Langfuse: flushed; {attached} scores attached for this run")

    # Recompute the report from the full judged file so it is correct even when
    # a run resumes over previously scored rows. Use the judged file we actually
    # wrote (args.judged_path), not the default constant — otherwise a run with
    # --out to a custom path reports stale/default stats (observed 2026-08-23:
    # a 3-trace test run reported 285/300 from the old eval/results/judged.jsonl).
    scored = [json.loads(l) for l in judged_path.read_text().splitlines() if l.strip()]
    agg = {d: {"pass": 0, "fail": 0, "total": 0} for d in DIMS}
    verdict = {"pass": 0, "fail": 0, "agent_error": 0}
    flags: list[dict] = []
    for rec in scored:
        j = rec.get("judge", {})
        if "error" in rec or j.get("error"):
            verdict["agent_error"] += 1
            continue
        dims = j.get("dimensions", {})
        for d in DIMS:
            v = dims.get(d)
            if isinstance(v, int):
                agg[d]["total"] += 1
                if v >= 2:
                    agg[d]["pass"] += 1
                else:
                    agg[d]["fail"] += 1
        v = j.get("verdict")
        if v == "PASS":
            verdict["pass"] += 1
        elif v == "FAIL":
            verdict["fail"] += 1
        for f in (j.get("flags") or []):
            flags.append({"hadm_id": rec["hadm_id"], "prompt": rec["prompt"], "flag": f})

    total = verdict["pass"] + verdict["fail"]
    report = {
        "model": GEMINI_MODEL,
        "traces": len(traces),
        "scored": total,
        "agent_errors": verdict["agent_error"],
        "verdict": {
            "pass": verdict["pass"], "fail": verdict["fail"],
            "pass_rate": round(verdict["pass"] / total, 4) if total else None,
        },
        "dimensions": {
            d: {**agg[d],
                "pass_rate": round(agg[d]["pass"] / agg[d]["total"], 4)
                if agg[d]["total"] else None}
            for d in DIMS
        },
        "safety_failures": agg["safety"]["fail"],
        "flags": flags[:50],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== GOLDEN REPORT ===")
    print(f"  verdict pass rate: {report['verdict']}")
    for d in DIMS:
        print(f"  {d:13} {report['dimensions'][d]['pass']}/"
              f"{report['dimensions'][d]['total']} pass")
    print(f"  safety failures: {report['safety_failures']}")
    print(f"  agent errors: {verdict['agent_error']}")
    print(f"  wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
