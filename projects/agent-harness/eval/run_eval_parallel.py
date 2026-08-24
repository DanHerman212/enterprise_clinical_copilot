"""run_eval_parallel.py — run the hybrid-108 eval with 4 parallel workers, ONE script.

Fixes the two failure modes from 2026-08-23:
  1. Langfuse disabled in workers — .env.lanfuse is `KEY=VALUE` with no `export`,
     so `source` only set shell vars and children never saw them. This script
     loads the file into os.environ so every spawned collect.py inherits it and
     every trace lands in Langfuse.
  2. Sloppy multi-terminal launching — this single script shards the sample,
     spawns 4 collect.py workers (one per shard, fresh out files), waits for all
     to finish, then merges and prints a quality summary.

Usage (harness root):
    .venv/bin/python eval/run_eval_parallel.py \
        --sample eval/results/golden_sample_hybrid_108.json \
        --out-prefix eval/results/traces_hybrid_108_fresh \
        --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (so subprocess children inherit them).

    This is the Langfuse fix: `.env.lanfuse` has no `export`, so bash `source`
    left the vars shell-local. Loading here means Popen'd workers see them.
    """
    loaded = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k:
            os.environ[k] = v
            loaded += 1
    print(f"env: loaded {loaded} vars from {path.name} "
          f"(langfuse_enabled={bool(os.environ.get('LANGFUSE_PUBLIC_KEY'))})",
          flush=True)


def make_shards(sample_path: Path, n_shards: int, out_prefix: Path) -> list[Path]:
    """Round-robin split the sample into n_shards sample files."""
    doc = json.loads(sample_path.read_text())
    patients = doc["patients"]
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, p in enumerate(patients):
        shards[i % n_shards].append(p)
    paths = []
    for i, sh in enumerate(shards):
        p = out_prefix.with_name(f"{out_prefix.stem}.s{i}.json")
        json.dump({"seed": doc.get("seed"), "n": len(sh), "patients": sh},
                  p.open("w"))
        paths.append(p)
        print(f"shard {i}: {len(sh)} patients -> {p.name}", flush=True)
    return paths


def quality_summary(files: list[Path]) -> dict:
    from collections import Counter
    n = missing = zero = empty = no_langfuse = 0
    prompts = Counter()
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            prompts[r.get("prompt")] += 1
            a = str(r.get("answer") or "")
            if "error" in r:
                missing += 0
            if "missing_text" in a:
                missing += 1
            if "no supporting" in a.lower() or "no relevant" in a.lower():
                zero += 1
            if not a.strip():
                empty += 1
            if not r.get("langfuse_trace_id"):
                no_langfuse += 1
    return {"n": n, "by_prompt": dict(prompts), "missing_text": missing,
            "zero_passage": zero, "empty_answer": empty, "no_langfuse": no_langfuse}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=str(HARNESS / "eval" / "results"
                                            / "golden_sample_hybrid_108.json"))
    ap.add_argument("--out-prefix", default=str(HARNESS / "eval" / "results"
                                                / "traces_hybrid_108_fresh"))
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()

    sample_path = Path(args.sample)
    out_prefix = Path(args.out_prefix)
    if not sample_path.exists():
        print(f"sample not found: {sample_path}", flush=True)
        return 2

    _load_env_file(HARNESS / ".env.lanfuse")
    shard_samples = make_shards(sample_path, args.shards, out_prefix)

    procs = []
    out_files = []
    for i, shard_sample in enumerate(shard_samples):
        out_file = out_prefix.with_name(f"{out_prefix.stem}.s{i}.jsonl")
        # fresh out file (start from scratch)
        out_file.write_text("")
        out_files.append(out_file)
        cmd = [sys.executable, str(HARNESS / "eval" / "collect.py"),
               "--sample", str(shard_sample), "--out", str(out_file)]
        print(f"spawn worker {i}: {' '.join(cmd)}", flush=True)
        procs.append((i, subprocess.Popen(cmd)))

    def _abort(signum, frame):
        print("run_eval_parallel: interrupted — stopping all workers", flush=True)
        for _, p in procs:
            p.terminate()
        for _, p in procs:
            p.wait()
        raise SystemExit(130)
    signal.signal(signal.SIGINT, _abort)
    signal.signal(signal.SIGTERM, _abort)

    # Poll until all workers exit; report progress periodically.
    last_report = 0.0
    while True:
        done = all(p.poll() is not None for _, p in procs)
        if time.time() - last_report >= 60:
            counts = [len(f.read_text().splitlines()) if f.exists() else 0
                      for f in out_files]
            print(f"progress: {counts}  total={sum(counts)}", flush=True)
            last_report = time.time()
        if done:
            break
        time.sleep(5)

    codes = {i: p.returncode for i, p in procs}
    print(f"workers finished: {codes}", flush=True)

    q = quality_summary(out_files)
    print(f"\n=== FINAL QUALITY (n={q['n']}) ===")
    print(f"by_prompt: {q['by_prompt']}")
    print(f"missing_text={q['missing_text']} zero_passage={q['zero_passage']} "
          f"empty_answer={q['empty_answer']} no_langfuse={q['no_langfuse']}")
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
