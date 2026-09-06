"""Ask the agent a question, over stdio or the deployed service.

    .venv/bin/python -m services.agent.run "What is the readmission risk for admission 20924467?"
    .venv/bin/python -m services.agent.run --trace "Is admission 1 high risk?"
    .venv/bin/python -m services.agent.run --transport http --url https://mcp-server-...run.app

Run from services.
"""

import argparse
import asyncio
import json
import sys

from services.agent.graph import ask, final_text
from services.agent.mcp_client import MCP_TRANSPORT, MCP_URL, toolbox
from services.mcp.config import GEMINI_MODEL

DEFAULT_QUESTION = "What is the readmission risk for admission 20924467?"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--transport", default=MCP_TRANSPORT, choices=["stdio", "http"])
    parser.add_argument("--url", default=MCP_URL, help="Cloud Run service URL for http")
    parser.add_argument("--trace", action="store_true", help="print every tool call")
    args = parser.parse_args()

    async with toolbox(transport=args.transport, url=args.url) as box:
        print(f"transport: {args.transport}")
        print(f"tools: {box.names}")
        print(f"model: {args.model}")
        print(f"\n> {args.question}\n")

        state = await ask(box, args.question, model=args.model)

        if args.trace:
            for call in state["tool_calls"]:
                print(f"[tool] {call['name']}({call['args']})")
                print(json.dumps(call["response"], indent=2)[:800])
                print()

        print(final_text(state))

        if not state["tool_calls"]:
            print("\nWARNING: the agent answered without calling any tool.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
