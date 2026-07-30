"""Transport-agnostic MCP server.

    python -m mcp_server.server --transport stdio          # local / Claude Desktop
    python -m mcp_server.server --transport http --port 8080   # Cloud Run

The transport is a flag, not a fork in the code: the same server object serves
both, so the local path and the deployed path cannot drift apart.

NOTE: under stdio the transport *is* stdout. Anything printed to stdout
corrupts the JSON-RPC stream, so diagnostics must go to stderr.
"""

import argparse
import sys

from mcp.server import MCPServer

from .tools import predict_readmission

server = MCPServer(
    name="readmission",
    version="0.1.0",
    instructions=(
        "Tools for the MIMIC-IV 30-day readmission model. Use "
        "predict_readmission to score a hospital admission and get the "
        "feature attributions behind the score."
    ),
)
server.add_tool(predict_readmission)


def main() -> None:
    parser = argparse.ArgumentParser(description="Readmission MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP transport only")
    parser.add_argument("--host", default="0.0.0.0",
                        help="HTTP transport only; Cloud Run requires 0.0.0.0")
    args = parser.parse_args()

    if args.transport == "stdio":
        server.run("stdio")
    else:
        print(f"Serving MCP over HTTP on {args.host}:{args.port}/mcp", file=sys.stderr)
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
