"""Programmatic eval runner for the RAG agent eval (hybrid-108).

Fixes the failure mode from 2026-08-23: a broken serving config (e.g.
DISCHARGE_TABLE pointing at the real MIMIC table while the deployed index is
built from readmission.hybrid_notes) silently produced `missing_text` /
zero-passage traces for ~2/3 of a 324-question run before anyone noticed.

The pipeline is gated so that class of failure dies in ~30 seconds instead of
after a 90-minute run:

  1. PREFLIGHT  — run a small smoke set through the real agent path and assert
                  retrieval is healthy (no missing_text, low zero-passage).
                  Exit non-zero BEFORE the long run if the config is broken.
  2. COLLECT    — launch eval/collect.py (resumable, append-only JSONL).
  3. WATCHDOG   — poll the traces JSONL; if the rolling retrieval-failure rate
                  stays above a threshold for several consecutive polls, kill
                  the run and report. Do not let a corrupted run finish.
  4. FINAL GATE — after collect finishes, print a full health breakdown and
                  exit non-zero if overall retrieval failures exceed the
                  threshold (the caller should NOT judge a bad run).

Usage (harness root):
    .venv/bin/python eval/run_eval.py \
        --sample eval/results/golden_sample_hybrid_108.json \
        --out   eval/results/traces_hybrid_108.jsonl \
        --preflight-n 6 \
        --max-fail-rate 0.30 \
        --watch-interval 30 --watch-consecutive 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS / "agent"))

from agent.graph import ask  # noqa: E402
from agent.mcp_client import toolbox  # noqa: E402


# --------------------------------------------------------------------------
# Trace classification (shared with analyze_problematic.py)
# --------------------------------------------------------------------------
def classify_trace(rec: dict) -> dict:
    """Return the problem flags for one trace record."""
    out = {"missing_text": False, "zero_passage": False, "unknown_patient": False}
    for tc in rec.get("tool_calls", []):
        resp = tc.get("response", {})
        if not isinstance(resp, dict):
            continue
        e = resp.get("error")
        if e == "missing_text":
            out["missing_text"] = True
        if e == "unknown_patient":
            out["unknown_patient"] = True
        if tc.get("name") in ("rag_search", "rag_search_sections") and (
            resp.get("returned", 0) == 0 and not resp.get("error")
        ):
            out["zero_passage"] = True
    return out


def retrieval_failed(flags: dict) -> bool:
    """The retrieval-specific failures we gate on (unknown_patient is NOT
    a retrieval failure — it is the predict path and is gated separately)."""
    return flags["missing_text"] or flags["zero_passage"]


def read_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {"n": 0, "retrieval_fail": 0, "missing_text": 0,
                "zero_passage": 0, "unknown_patient": 0, "retrieval_fail_rate": 0.0}
    mt = zp = up = rf = 0
    for r in recs:
        f = classify_trace(r)
        mt += int(f["missing_text"])
        zp += int(f["zero_passage"])
        up += int(f["unknown_patient"])
        rf += int(retrieval_failed(f))
    return {"n": n, "retrieval_fail": rf, "missing_text": mt,
            "zero_passage": zp, "unknown_patient": up,
            "retrieval_fail_rate": rf / n}


def print_summary(s: dict, label: str = "summary") -> None:
    print(f"[{label}] n={s['n']} retrieval_fail={s['retrieval_fail']} "
          f"({s['retrieval_fail_rate']:.0%}) "
          f"missing_text={s['missing_text']} zero_passage={s['zero_passage']} "
          f"unknown_patient={s['unknown_patient']}", flush=True)


# --------------------------------------------------------------------------
# Preflight: smoke-test the real agent path before the long run
# --------------------------------------------------------------------------
async def _run_one(question: str, *, name: str, tags: list[str]) -> dict:
    async with toolbox() as box:
        return await ask(box, question, name=name, tags=tags)


def preflight(sample_path: Path, n: int, max_fail_rate: float) -> int:
    sample = json.loads(sample_path.read_text())
    patients = sample["patients"]
    # Ask a retrieval question (meds) for the first n patients; meds goes
    # through rag_search_sections, which is exactly the path that broke.
    cases = [(p["hadm_id"], f"What medications were they discharged on? "
                            f"For admission {p['hadm_id']}.") for p in patients[:n]]
    print(f"PREFLIGHT: {len(cases)} smoke questions through the real agent path …",
          flush=True)

    flags = {"missing_text": 0, "zero_passage": 0, "retrieval_fail": 0}
    for i, (hadm, q) in enumerate(cases, 1):
        state = asyncio.run(_run_one(q, name=f"eval.preflight",
                                     tags=["eval", "preflight", f"hadm:{hadm}"]))
        tc = state.get("tool_calls", [])
        f = {"missing_text": False, "zero_passage": False, "unknown_patient": False}
        for c in tc:
            resp = c.get("response", {})
            if not isinstance(resp, dict):
                continue
            if resp.get("error") == "missing_text":
                f["missing_text"] = True
            if c.get("name") in ("rag_search", "rag_search_sections") and (
                resp.get("returned", 0) == 0 and not resp.get("error")
            ):
                f["zero_passage"] = True
        for k, v in f.items():
            if k != "unknown_patient":
                flags[k] += int(v)
        print(f"  [{i}/{len(cases)}] hadm={hadm} "
              f"tools={[c.get('name') for c in tc]} "
              f"missing_text={f['missing_text']} zero_passage={f['zero_passage']}",
              flush=True)

    rate = (flags["missing_text"] + flags["zero_passage"]) / len(cases)
    print(f"PREFLIGHT: retrieval-failure rate = {rate:.0%} "
          f"(threshold {max_fail_rate:.0%})", flush=True)
    if rate > max_fail_rate:
        print("PREFLIGHT: FAIL — serving config is broken (missing_text / "
              "zero-passage too high). NOT starting the run. Fix DISCHARGE_TABLE "
              "/ index before re-running.", flush=True)
        return 1
    print("PREFLIGHT: PASS — retrieval is healthy. Starting the run.", flush=True)
    return 0


# --------------------------------------------------------------------------
# Watchdog
# --------------------------------------------------------------------------
def watch(out_path: Path, max_fail_rate: float, interval: int,
          consecutive: int) -> int:
    """Poll the traces file; abort if the rolling failure rate stays high.

    Returns 0 if the run is healthy enough to continue, 1 if it should be
    killed. Caller keeps calling while the collect process is alive.
    """
    recs = read_traces(out_path)
    if not recs:
        return 0  # nothing to judge yet
    s = summarize(recs)
    print_summary(s, "watch")
    # Fail only when the *overall* run has enough data and the failure rate is
    # structurally broken (not one bad question at the start).
    if s["n"] >= 20 and s["retrieval_fail_rate"] > max_fail_rate:
        print("WATCHDOG: retrieval-failure rate above threshold — "
              "run is corrupted. Aborting.", flush=True)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=str, default=str(
        HARNESS / "eval" / "results" / "golden_sample_hybrid_108.json"))
    ap.add_argument("--out", type=str, default=str(
        HARNESS / "eval" / "results" / "traces_hybrid_108.jsonl"))
    ap.add_argument("--preflight-n", type=int, default=6)
    ap.add_argument("--max-fail-rate", type=float, default=0.30)
    ap.add_argument("--watch-interval", type=int, default=30)
    ap.add_argument("--watch-consecutive", type=int, default=3)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--collect-args", nargs="*", default=[],
                    help="extra args forwarded to eval/collect.py")
    args = ap.parse_args()

    sample_path = Path(args.sample)
    out_path = Path(args.out)
    if not sample_path.exists():
        print(f"sample not found: {sample_path}", flush=True)
        return 2

    # 1. PREFLIGHT
    if not args.skip_preflight:
        rc = preflight(sample_path, args.preflight_n, args.max_fail_rate)
        if rc != 0:
            return rc
    else:
        print("PREFLIGHT: skipped (--skip-preflight)", flush=True)

    # 2. COLLECT (resumable)
    cmd = [sys.executable, str(HARNESS / "eval" / "collect.py"),
           "--sample", str(sample_path), "--out", str(out_path)]
    cmd += args.collect_args
    print("COLLECT:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)

    def _abort(signum, frame):
        print("run_eval: interrupted — stopping collect", flush=True)
        proc.terminate()
        proc.wait()
        raise SystemExit(130)
    signal.signal(signal.SIGINT, _abort)
    signal.signal(signal.SIGTERM, _abort)

    # 3. WATCHDOG
    high_count = 0
    while proc.poll() is None:
        time.sleep(args.watch_interval)
        if watch(out_path, args.max_fail_rate, args.watch_interval,
                 args.watch_consecutive) == 1:
            high_count += 1
            if high_count >= args.watch_consecutive:
                print("WATCHDOG: consecutive failures — killing collect",
                      flush=True)
                proc.terminate()
                proc.wait()
                return 1
        else:
            high_count = 0

    rc = proc.returncode
    print(f"COLLECT: finished with exit code {rc}", flush=True)

    # 4. FINAL GATE
    recs = read_traces(out_path)
    s = summarize(recs)
    print_summary(s, "final")
    if s["n"] == 0:
        print("FINAL GATE: no traces written — run produced nothing", flush=True)
        return 1
    if s["retrieval_fail_rate"] > args.max_fail_rate:
        print(f"FINAL GATE: FAIL — retrieval-failure rate {s['retrieval_fail_rate']:.0%} "
              f"> threshold {args.max_fail_rate:.0%}. Do NOT judge this run.",
              flush=True)
        return 1
    print(f"FINAL GATE: PASS — {s['n']} traces, retrieval-failure rate "
          f"{s['retrieval_fail_rate']:.0%} within threshold.", flush=True)
    print("NEXT: run the judge (eval/judge.py) on the traces.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
