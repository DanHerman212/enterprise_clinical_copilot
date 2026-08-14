"""LLM-as-judge over collected traces against the golden rubric -> report.

For each trace, send the question + agent answer + the evidence the agent
actually had (tool outputs + retrieved passages) to Gemini, which applies the
versioned rubric (eval/rubric.md) and returns JSON scores per dimension.
Aggregates into a pass-rate report. This is the qualitative half of the gate.

Usage (harness root):
    .venv/bin/python eval/judge.py
"""

import json
import sys
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

# P2 fix (2026-08-14, after human judge-validation showed kappa=0): the judge
# must see the FULL evidence the agent saw. v1 truncated each passage to 400
# chars (the redacted header + HPI opening), so it could not verify meds/course
# that live in the body+end of each ~11k-char passage and falsely flagged
# faithful answers as "fabricated". Full passages, generous cap.
EVIDENCE_CAP = 120000
PER_PASSAGE_CAP = 20000


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


def _judge(client, system: str, user: str) -> dict:
    for attempt in range(4):
        try:
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
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt)
            else:
                return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    traces = [json.loads(l) for l in TRACES.read_text().splitlines() if l.strip()]
    print(f"Judging {len(traces)} traces (model {GEMINI_MODEL})")

    # Resumable: skip (hadm_id, prompt) pairs already scored in judged.jsonl and
    # append, so a re-run after a crash continues instead of restarting.
    done: set[tuple] = set()
    if JUDGED.exists():
        for line in JUDGED.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec.get("hadm_id"), rec.get("prompt")))
            except json.JSONDecodeError:
                pass
    print(f"Resuming: {len(done)} already judged, {len(traces) - len(done)} to go")

    with JUDGED.open("a") as fh:
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
            print(f"[{i}/{len(traces)}] {t['hadm_id']}/{t['prompt']}: {j.get('verdict') or '?'}", flush=True)

    # Recompute the report from the full judged file so it is correct even when
    # a run resumes over previously scored rows.
    scored = [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]
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
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== GOLDEN REPORT ===")
    print(f"  verdict pass rate: {report['verdict']}")
    for d in DIMS:
        print(f"  {d:13} {report['dimensions'][d]['pass']}/"
              f"{report['dimensions'][d]['total']} pass")
    print(f"  safety failures: {report['safety_failures']}")
    print(f"  agent errors: {verdict['agent_error']}")
    print(f"  wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
