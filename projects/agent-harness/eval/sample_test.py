"""Stratify the golden sample from the TEST split (RAG-covered, indexed).

The demo split cannot host the narrative eval: the RAG index was built from the
test-split note cache only (`fetch_note_cache.py WHERE split_name='test'`), so
every `rag_search` on a demo-split admission returns zero passages and the
meds/summarize prompts degrade to "no passages found" instead of exercising
retrieval + citation. The test split is the indexed corpus, so this builds the
final 100-patient sample there.

No re-scoring: `quant.py` already scored the full 49,103 test rows into
`holdout_scored.jsonl`. This reads that, bins by the same calibrated bands as
the demo sampler (low <0.10 x20, borderline 0.10-0.20 x40, high >=0.20 x40,
SEED=20260814), and live-probes each candidate against the deployed RAG index
so only admissions with >=1 indexed passage are eligible. Every row in the
final sample is therefore guaranteed to exercise real retrieval + citation.

Usage (harness root):
    .venv/bin/python eval/sample_test.py
"""

import asyncio
import json
import random
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS / "mcp_server"))

RESULTS = HARNESS / "eval" / "results"
SCORED = RESULTS / "holdout_scored.jsonl"
OUT = RESULTS / "golden_sample.json"
SEED = 20260814
PROBE_CONCURRENCY = 16

# Same calibrated bands as the demo sampler (sample.py).
BANDS = [
    ("low", 0.0, 0.10, 20),
    ("borderline", 0.10, 0.20, 40),
    ("high", 0.20, 1.01, 40),
]

PROBE_QUERY = "discharge medications and hospital course"


def _load_scored() -> list[dict]:
    rows = [
        json.loads(line)
        for line in SCORED.read_text().splitlines()
        if line.strip()
    ]
    return [{"hadm_id": int(r["hadm_id"]), "probability": float(r["probability"]),
             "readmission_30d": int(r["readmission_30d"])} for r in rows]


async def _probe(sem: asyncio.Semaphore, hadm_id: int) -> bool:
    """True if the deployed RAG index returns >=1 passage for this admission."""
    from mcp_server.tools.rag_search import rag_search  # deferred import

    async with sem:
        try:
            resp = await rag_search(hadm_id, PROBE_QUERY, top_k=1)
            return bool(resp.get("returned"))
        except Exception as e:  # transient endpoint/embedding errors
            print(f"    probe {hadm_id} failed ({type(e).__name__}): {e}", flush=True)
            return False


async def _covered_set(hadm_ids: list[int]) -> dict[int, bool]:
    sem = asyncio.Semaphore(PROBE_CONCURRENCY)
    results = await asyncio.gather(*[_probe(sem, h) for h in hadm_ids])
    return dict(zip(hadm_ids, results))


def _threshold() -> float:
    m = json.loads((RESULTS / "quant_metrics.json").read_text())
    return float(m.get("threshold", 0.12))


async def main() -> int:
    rows = _load_scored()
    threshold = _threshold()
    print(f"Test split (scored): {len(rows)} rows, threshold {threshold}")

    rng = random.Random(SEED)
    chosen: list[dict] = []
    for name, lo, hi, want in BANDS:
        cand = [r for r in rows if lo <= r["probability"] < hi]
        rng.shuffle(cand)
        print(f"\n{name:11} [{lo:.2f},{hi:.2f})  avail={len(cand):4d}  want={want}")

        covered: dict[int, bool] = {}
        taken = 0
        probe_count = 0
        for batch_start in range(0, len(cand), PROBE_CONCURRENCY * 8):
            batch = cand[batch_start : batch_start + PROBE_CONCURRENCY * 8]
            covered.update(await _covered_set([r["hadm_id"] for r in batch]))
            probe_count += len(batch)
            for r in batch:
                if taken >= want:
                    break
                if covered[r["hadm_id"]]:
                    r["band"] = name
                    chosen.append(r)
                    taken += 1
            print(f"    probed {probe_count:4d} ... covered-so-far {sum(covered.values()):4d}, taken {taken}")
            if taken >= want:
                break
        flag = "" if taken == want else f"  ** only {taken} covered in this band **"
        print(f"  -> took {taken}/{want}{flag}")

    def _band(p):
        return next(b[0] for b in BANDS if b[1] <= p < b[2])

    sample = [
        {
            "hadm_id": int(r["hadm_id"]),
            "probability": r["probability"],
            "threshold": threshold,
            "readmission_30d": int(r["readmission_30d"]),
            "band": _band(r["probability"]),
            "rag_verified": True,
        }
        for r in sorted(chosen, key=lambda r: r["probability"])
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"seed": SEED, "n": len(sample), "source": "test-split (RAG-covered)",
                    "patients": sample}, indent=2) + "\n"
    )
    print(f"\nWrote golden sample: {len(sample)} patients (all RAG-verified) -> {OUT}")
    print(f"Band mix: " + ", ".join(
        f"{b[0]}={sum(1 for s in sample if s['band'] == b[0])}" for b in BANDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
