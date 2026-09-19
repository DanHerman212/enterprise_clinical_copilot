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
from datetime import datetime, timezone
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
# Where the run's artifacts go, and where the archive already is. This used to
# be `HARNESS/"eval"/"results"`, computed from a `parents[1]` that pointed at
# `evaluation/` once this file moved up a level — so a run wrote
# `evaluation/eval/results/traces.jsonl`, a path that does not exist, is not
# ignored by .gitignore, and would have committed clinical text into the
# repository. The archive's real home is beside the golden sample it reads, and
# `.gitignore` already covers it (2026-09-18).
RESULTS = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.agent.graph import ask, final_text  # noqa: E402
from services.agent.mcp_client import toolbox  # noqa: E402
from prompt_set import PROMPT_NAMES, question_for  # noqa: E402

SAMPLE = RESULTS / "golden_sample.json"
OUT = RESULTS / "traces.jsonl"

# The wording lives in `prompt_set.py` because a second collector sends the same
# questions against the deployed service, and two copies of a prompt are two
# prompts: the pass rates they produce are not comparable, which is a claim
# nobody would notice was false until the numbers disagreed.


# A single agent run can hang forever on a stuck Vertex/Gemini call (observed
# 2026-08-19: the full run stalled ~55min at trace 210/300 with no output). The
# timeout makes a hang raise (-> retried, then flagged) instead of stalling the
# whole collect.
_ASK_TIMEOUT_SECONDS = 180


async def _run_async(question: str, retries: int = 2, *, name: str | None = None,
                     tags: list[str] | None = None) -> dict:
    """Run one question on the shared event loop.

    A single persistent loop (asyncio.run(main()) at the bottom) is used for the
    whole run: per-question asyncio.run() churn left the Google GenAI clients'
    aclose() tasks stranded after each loop teardown ("Event loop is closed"),
    which accumulated across many questions and could block the loop, stalling
    the run (observed 2026-08-23 at 60/324). With one loop, wait_for can cancel
    a hung call instead of the loop being wedged during teardown.
    """
    async def go():
        async with toolbox() as box:
            return await ask(box, question, name=name, tags=tags)

    last = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(go(), timeout=_ASK_TIMEOUT_SECONDS)
        except Exception as e:  # transient transport/Vertex errors + hangs
            last = e
            print(f"    (attempt {attempt + 1} failed: {type(e).__name__})",
                  flush=True)
    raise last


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--prompt", choices=["risk", "meds", "summarize", "all"],
                    default="all")
    ap.add_argument("--sample", type=str, default=str(SAMPLE),
                    help="golden sample JSON path (default: 100-MIMIC golden_sample.json)")
    ap.add_argument("--out", type=str, default=str(OUT),
                    help="traces JSONL output, appended (default: traces.jsonl)")
    args = ap.parse_args()

    out = Path(args.out)
    sample = json.loads(Path(args.sample).read_text())["patients"]
    if args.max_cases:
        sample = sample[: args.max_cases]
    prompts = list(PROMPT_NAMES) if args.prompt == "all" else [args.prompt]

    total = len(sample) * len(prompts)
    print(f"Running {len(sample)} cases x {len(prompts)} prompts = {total} agent runs")

    # Resumable: skip (hadm_id, prompt) pairs already traced so a restart after
    # a crash or hang continues instead of redoing everything (append-only,
    # mirroring judge.py's resume).
    done: set[tuple] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec.get("hadm_id"), rec.get("prompt")))
            except json.JSONDecodeError:
                pass
    print(f"Resuming: {len(done)} already traced, {total - len(done)} to go")

    completed = len(done)
    with out.open("a") as fh:
        for patient in sample:
            for ptype in prompts:
                if (patient["hadm_id"], ptype) in done:
                    continue
                q = question_for(ptype, patient["hadm_id"])
                hadm = patient["hadm_id"]
                try:
                    state = await _run_async(
                        q,
                        name=f"eval.{ptype}",
                        tags=["eval", "hybrid-108", f"hadm:{hadm}"],
                    )
                    rec = {
                        "hadm_id": patient["hadm_id"],
                        "prompt": ptype,
                        "question": q,
                        "answer": final_text(state),
                        "tool_calls": state["tool_calls"],
                        "probability": patient["probability"],
                        "band": patient.get("band"),
                        # When the record was written. The archive has a stated
                        # lifetime (layer 8 retention decision) and a lifetime
                        # cannot be enforced on a record that does not say how
                        # old it is: without this field the only options were to
                        # keep everything or to delete by file, and the file is
                        # the whole archive.
                        "at": datetime.now(timezone.utc).isoformat(),
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

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
