"""Transport-agnostic MCP server.

    python -m mcp_server.server --transport stdio          # local / Claude Desktop
    python -m mcp_server.server --transport http --port 8080   # Cloud Run

The transport is a flag, not a fork in the code: the same server object serves
both, so the local path and the deployed path cannot drift apart.

NOTE: under stdio the transport *is* stdout. Anything printed to stdout
corrupts the JSON-RPC stream, so diagnostics must go to stderr.
"""

import argparse
import os
import sys

from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import LOCATION, PROJECT
from .features import FEATURE_SOURCE
from .tools import predict_readmission, rag_search, rag_search_sections

server = MCPServer(
    name="readmission",
    version="0.1.0",
    instructions=(
        "Tools for the MIMIC-IV 30-day readmission model. Use "
        "predict_readmission to score a hospital admission and get the "
        "feature attributions behind the score. Use rag_search to retrieve "
        "cited passages from a patient's discharge notes for a free-text "
        "question. Use rag_search_sections to retrieve one cited passage per "
        "major discharge-note section for summarization."
    ),
)
server.add_tool(predict_readmission)
server.add_tool(rag_search)
server.add_tool(rag_search_sections)


@server.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness only — deliberately shallow.

    A deep check would call Vertex and BigQuery, turning every probe into
    billed API calls on a service that scales to zero. If the process is up,
    the tool's own structured errors report dependency failures per request.
    """
    return JSONResponse({
        "status": "ok",
        "project": PROJECT,
        "location": LOCATION,
        "feature_source": FEATURE_SOURCE,
    })


# Defense-in-depth (ECC-25): on Cloud Run (K_SERVICE is platform-injected) the
# IAM front end forwards the caller's Authorization header; a request without
# one means the service was deployed --allow-unauthenticated by mistake and
# would otherwise expose the dataset AND billable Vertex/BigQuery calls.
# Local runs (no K_SERVICE) are unaffected.
_REQUIRE_AUTH_HEADER = bool(os.environ.get("K_SERVICE")) and (
    os.environ.get("ALLOW_UNAUTHENTICATED", "").lower() != "true"
)


def _require_auth_header(app):
    """Reject non-/health HTTP requests that carry no Authorization header."""
    async def guarded(scope, receive, send):
        if scope["type"] == "http" and scope.get("path") != "/health":
            names = {name.lower() for name, _ in scope.get("headers") or []}
            if b"authorization" not in names:
                response = JSONResponse(
                    {"error": "unauthenticated",
                     "message": "This service requires an identity token."},
                    status_code=401,
                )
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return guarded


def main() -> None:
    parser = argparse.ArgumentParser(description="Readmission MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    # Cloud Run injects PORT and expects the container to honour it.
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)),
                        help="HTTP transport only")
    parser.add_argument("--host", default="0.0.0.0",
                        help="HTTP transport only; Cloud Run requires 0.0.0.0")
    parser.add_argument("--stateful", action="store_true",
                        help="keep MCP sessions in memory (single instance only)")
    args = parser.parse_args()

    if args.transport == "stdio":
        server.run("stdio")
    else:
        print(f"Serving MCP over HTTP on {args.host}:{args.port}/mcp", file=sys.stderr)
        # Stateless by default. Streamable-HTTP sessions live in the memory of
        # one instance; behind a Cloud Run load balancer a follow-up request
        # can land on a different instance and fail to find its session. This
        # tool holds no per-session state, so there is nothing to lose.
        app = server.streamable_http_app(
            stateless_http=not args.stateful,
            host=args.host,
        )
        if _REQUIRE_AUTH_HEADER:
            app = _require_auth_header(app)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
