"""drive_live_chips.py — exercise the live predict + RAG tools on hybrid patients.

Calls the real MCP tools (which hit the deployed predict endpoint + Vector
Search + BigQuery text fetch) for a few hybrid patients and chip intents, so
the full user journey is confirmed against real note text.

Usage (from projects/agent-harness):
  FEATURE_TABLE=... DISCHARGE_TABLE=... ../../.venv/bin/python scripts/drive_live_chips.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.tools.predict import predict_readmission  # noqa: E402
from mcp_server.tools.rag_search import rag_search  # noqa: E402

PATIENTS = (90000001, 90000009, 90000017)  # low / borderline / high
QUERIES = ("medications", "summarize the hospital course")


async def main() -> None:
    for hid in PATIENTS:
        r = await predict_readmission(hid)
        print(f"predict {hid}: prob={r['probability']:.4f} "
              f"decision={r['decision']} src={r['feature_source']}")
        for q in QUERIES:
            rr = await rag_search(hid, q, top_k=2)
            print(f"  rag '{q}': returned={rr['returned']}")
            for p in rr["passages"][:2]:
                print(f"    {p['id']} | {p['section']} | {p['score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
