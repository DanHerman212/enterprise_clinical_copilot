"""capture_synthetic_rag_fixtures — capture live synthetic rag passages.

Task 5c: once the synthetic index is deployed, run the REAL rag_search against
it for the demo chip queries on a primary synthetic patient and store the
responses as rag_*.json fixtures, so fixture mode cites the same passages the
live agent would retrieve. Provenance is marked SYNTHETIC.

The site's fixtures.py keys captured rag by query and gates on hadm_id, so the
primary patient's chips get real citations and everyone else falls back to the
honest empty — matching the live behavior.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/capture_synthetic_rag_fixtures.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = HARNESS_ROOT / "data" / "demo_fixtures"
SITE_FIXTURES = Path(os.environ.get(
    "SITE_FIXTURES",
    HARNESS_ROOT.parents[1].parent / "danielmherman" / "demo" / "data" / "demo_fixtures",
))
PROJECT = "trim-icon-498815-a0"

# The synthetic high-risk patient: every chip returns passages.
PRIMARY_HADM = int(os.environ.get("PRIMARY_HADM", "90000017"))

# The exact queries the site's fixture chips issue (fixtures.py _CHIP_QUERY).
CHIP_QUERIES = {
    "risk": "sepsis and elevated lactate on broad-spectrum antibiotics",
    "meds": "medications",
    "summarize": "summarize the hospital course and discharge diagnosis",
}


async def main() -> int:
    # Point rag_search at the synthetic notes table so passage text resolves
    # from synthetic data (never real MIMIC).
    os.environ["DISCHARGE_TABLE"] = f"{PROJECT}.readmission.synthetic_notes"
    sys.path.insert(0, str(HARNESS_ROOT))
    from mcp_server.tools.rag_search import rag_search  # noqa: E402

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for key, query in CHIP_QUERIES.items():
        res = await rag_search(PRIMARY_HADM, query, top_k=5)
        if res.get("error"):
            print(f"ERROR {key}: {res['error']}")
            return 1
        payload = {
            **res,
            "provenance": f"SYNTHETIC — live rag_search on the synthetic index "
                          f"({PRIMARY_HADM}, 2026-08-20)",
        }
        name = f"rag_{key}_{PRIMARY_HADM}.json"
        (OUT_DIR / name).write_text(json.dumps(payload, indent=2))
        written.append(name)
        print(f"{key}: returned={payload['returned']} -> {name}")
        for p in payload["passages"]:
            print(f"    [{p['score']:.4f}] {p['section']:24s} {p['id']}")

    if SITE_FIXTURES:
        SITE_FIXTURES.mkdir(parents=True, exist_ok=True)
        for name in written:
            (SITE_FIXTURES / name).write_text((OUT_DIR / name).read_text())
        print(f"copied {len(written)} fixtures -> {SITE_FIXTURES}")

    print(f"\nwrote {len(written)} fixtures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
