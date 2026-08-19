"""Run the agent on the golden sample (3 prompts each) -> traces.jsonl.

Uses the same local agent graph + MCP toolbox as Tier 2 (stdio transport),
hitting the live endpoints. Captures the answer + tool_calls (the evidence)
per (hadm_id, prompt) as the durable JSONL trace archive.

Usage (harness root):
    .venv/bin/python eval/collect.py                 # full sample x 3 prompts
    .venv/bin/python eval/collect.py --max-cases 3   # pilot (3 patients x 3)
    .venv/bin/python eval/collect.py --prompt risk   # one prompt type only
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS / "agent"))

from agent.graph import ask, final_text  # noqa: E402
from agent.mcp_client import toolbox  # noqa: E402

SAMPLE = HARNESS / "eval" / "results" / "golden_sample.json"
OUT = HARNESS / "eval" / "results" / "traces.jsonl"

PROMPTS = {
    "risk": lambda h: f"What is the 30-day readmission risk for admission {h}?",
    "meds": lambda h: f"What medications were they discharged on? For admission {h}.",
    "summarize": lambda h: f"Summarize the recent discharge notes. For admission {h}.",
}


# A single agent run can hang forever on a stuck Vertex/Gemini call (observed
# 2026-08-19: the full run stalled ~55min at trace 210/300 with no output). The
# timeout makes a hang raise (-> retried, then flagged) instead of stalling the
# whole collect.
_ASK_TIMEOUT_SECONDS = 180


def _run(question: str, retries: int = 2) -> dict:
    async def go():
        async with toolbox() as box:
            return await ask(box, question)

    last = None
    for attempt in range(retries + 1):
        try:
            return asyncio.run(asyncio.wait_for(go(), timeout=_ASK_TIMEOUT_SECONDS))
        except Exception as e:  # transient transport/Vertex errors + hangs
            last = e
            print(f"    (attempt {attempt + 1} failed: {type(e).__name__})",
                  flush=True)
    raise last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--prompt", choices=["risk", "meds", "summarize", "all"],
                    default="all")
    args = ap.parse_args()

    sample = json.loads(SAMPLE.read_text())["patients"]
    if args.max_cases:
        sample = sample[: args.max_cases]
    prompts = list(PROMPTS) if args.prompt == "all" else [args.prompt]

    total = len(sample) * len(prompts)
    print(f"Running {len(sample)} cases x {len(prompts)} prompts = {total} agent runs")

    # Resumable: skip (hadm_id, prompt) pairs already traced so a restart after
    # a crash or hang continues instead of redoing everything (append-only,
    # mirroring judge.py's resume).
    done: set[tuple] = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec.get("hadm_id"), rec.get("prompt")))
            except json.JSONDecodeError:
                pass
    print(f"Resuming: {len(done)} already traced, {total - len(done)} to go")

    completed = len(done)
    with OUT.open("a") as fh:
        for patient in sample:
            for ptype in prompts:
                if (patient["hadm_id"], ptype) in done:
                    continue
                q = PROMPTS[ptype](patient["hadm_id"])
                try:
                    state = _run(q)
                    rec = {
                        "hadm_id": patient["hadm_id"],
                        "prompt": ptype,
                        "question": q,
                        "answer": final_text(state),
                        "tool_calls": state["tool_calls"],
                        "probability": patient["probability"],
                        "band": patient.get("band"),
                    }
                    # Keep the Langfuse trace id (when Langfuse is enabled) so
                    # judge.py can attach rubric scores to the right trace.
                    if state.get("langfuse_trace_id"):
                        rec["langfuse_trace_id"] = state["langfuse_trace_id"]
                except Exception as e:  # keep going; judge flags it later
                    rec = {"hadm_id": patient["hadm_id"], "prompt": ptype,
                           "error": f"{type(e).__name__}: {e}"}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                completed += 1
                status = "ok" if "error" not in rec else "ERROR"
                print(f"[{completed}/{total}] {patient['hadm_id']}/{ptype}: {status}",
                      flush=True)

    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
