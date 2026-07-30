"""Ask the local agent a question over stdio.

    .venv/bin/python -m agent.run "What is the readmission risk for admission 20924467?"
    .venv/bin/python -m agent.run --trace "Is admission 1 high risk?"

Run from projects/agent-harness.
"""

import argparse
import asyncio
import json
import sys

from agent.graph import ask, final_text
from agent.mcp_client import stdio_toolbox
from mcp_server.config import GEMINI_MODEL

DEFAULT_QUESTION = "What is the readmission risk for admission 20924467?"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--trace", action="store_true", help="print every tool call")
    args = parser.parse_args()

    async with stdio_toolbox(python=sys.executable) as toolbox:
        print(f"tools: {toolbox.names}")
        print(f"model: {args.model}")
        print(f"\n> {args.question}\n")

        state = await ask(toolbox, args.question, model=args.model)

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
