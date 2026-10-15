"""capture_synthetic_literature_fixtures — capture live PubMed records for the demo.

Milestone 4B (§3). The Literature Risk Matrix reads across 6-12 records, and fixture
mode needs those records on disk: the demo must not depend on NCBI being reachable,
and it must not break mid-rehearsal because a shared IP hit the rate limit. So the
records the matrix is built from are captured once and read from disk afterwards.

Unlike the rag fixtures this script sits beside, these are keyed by QUERY alone and
carry no admission. The literature tool has no `hadm_id` in its signature on purpose —
the same clinical question returns the same articles whoever the patient is — so
keying a capture by patient would claim a specificity the tool does not have.

Provenance is recorded inside each fixture and marked LIVE, with its date, so a reader
can tell captured public evidence from computed synthetic evidence. The rag fixtures
are marked SYNTHETIC because they come from the synthetic index; this is real PubMed.

Usage (from the agent repository root):
  NCBI_EMAIL=$(git config user.email) .venv/bin/python \
      scripts/demo/capture_synthetic_literature_fixtures.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = HARNESS_ROOT / "data" / "demo_fixtures"
SITE_FIXTURES = Path(os.environ.get(
    "SITE_FIXTURES",
    HARNESS_ROOT.parent / "danielmherman" / "demo" / "data" / "demo_fixtures",
))

# The demo cohort's conditions (§3). One capture per condition the matrix has to
# cover, so `/pubmed` on any of the three demo patients returns records about the
# condition that patient actually has.
QUERIES = {
    "chf_readmission": "congestive heart failure 30-day readmission risk",
    "copd_exacerbation": (
        "chronic obstructive pulmonary disease acute exacerbation readmission"
    ),
    "lumbar_stenosis": (
        "postoperative complications lumbar spinal stenosis readmission"
    ),
}

# The matrix is a column of records to compare, and the tool's ceiling is ten. Asking
# for the ceiling rather than the default three is the point of a matrix: three
# records do not make a table.
MAX_RESULTS = 10


async def main() -> int:
    if not os.environ.get("NCBI_EMAIL"):
        # NCBI asks for a real contact address and rate-limits anonymous traffic
        # harder. Refusing here is better than capturing a fixture that turns out to
        # be a rate-limit notice in a file named after a clinical question.
        print("NCBI_EMAIL is not set. Set it to a real contact address and re-run:\n"
              "  NCBI_EMAIL=$(git config user.email) .venv/bin/python "
              "scripts/demo/capture_synthetic_literature_fixtures.py")
        return 2

    sys.path.insert(0, str(HARNESS_ROOT))
    from services.mcp.tools.literature import search_literature  # noqa: E402

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for key, query in QUERIES.items():
        result = await search_literature(query, MAX_RESULTS)
        if result.get("error"):
            # A degraded capture is worse than no capture: it would sit on disk
            # looking like evidence and be a rate limit or a missing record. Stop
            # and say which one it was.
            print(f"ERROR {key}: {result.get('error')} — {result.get('message', '')}")
            return 1
        payload = {
            **result,
            "provenance": (
                "LIVE PubMed capture (NCBI E-utilities) 2026-10-15 — "
                f"{result['returned']} record(s) for {query!r}. "
                "Public literature; no patient data."
            ),
        }
        name = f"literature_{key}.json"
        (OUT_DIR / name).write_text(json.dumps(payload, indent=2))
        written.append(name)
        print(f"{key}: returned={payload['returned']} -> {name}")
        for article in payload["articles"]:
            print(f"    {str(article.get('pmid', '')):>9}  "
                  f"{str(article.get('journal', ''))[:26]:26}  "
                  f"{str(article.get('title', ''))[:54]}")

    if SITE_FIXTURES:
        SITE_FIXTURES.mkdir(parents=True, exist_ok=True)
        for name in written:
            (SITE_FIXTURES / name).write_text((OUT_DIR / name).read_text())
        print(f"copied {len(written)} fixtures -> {SITE_FIXTURES}")

    print(f"\nwrote {len(written)} fixtures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
