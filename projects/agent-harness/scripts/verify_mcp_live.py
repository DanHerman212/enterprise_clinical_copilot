"""verify_mcp_live.py — exercise the deployed MCP server's tools directly.

One-shot verification that the live mcp-server (Cloud Run) serves SYNTHETIC
data end-to-end: predict_readmission + rag_search on synthetic patient
90000017.

Token: taken from MCP_ID_TOKEN if set, else `gcloud auth print-identity-token
--audiences=$MCP_URL` (ADC user creds cannot mint ID tokens).

Usage:
  MCP_URL=<url> .venv/bin/python scripts/verify_mcp_live.py
"""

import asyncio
import json
import os
import subprocess
import sys

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _token(url: str) -> str:
    if os.environ.get("MCP_ID_TOKEN"):
        return os.environ["MCP_ID_TOKEN"]
    r = subprocess.run(
        ["gcloud", "auth", "print-identity-token", "--audiences=" + url],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return r.stdout.strip()


async def main() -> None:
    url = os.environ["MCP_URL"].rstrip("/")
    token = _token(url)

    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as hc:
        async with streamable_http_client(f"{url}/mcp", http_client=hc) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                tools = await s.list_tools()
                print("tools:", [t.name for t in tools.tools])

                r = await s.call_tool("predict_readmission", {"hadm_id": 90000017})
                d = json.loads(r.content[0].text)
                print(f"predict 90000017: prob={d.get('probability')} "
                      f"decision={d.get('decision')} src={d.get('feature_source')}")

                r = await s.call_tool(
                    "rag_search", {"hadm_id": 90000017, "query": "medications", "top_k": 3}
                )
                d = json.loads(r.content[0].text)
                print(f"rag 90000017 meds: returned={d.get('returned')}")
                for p in d.get("passages", [])[:3]:
                    print(f"    {p['id']}  score={p['score']}")


if __name__ == "__main__":
    asyncio.run(main())
