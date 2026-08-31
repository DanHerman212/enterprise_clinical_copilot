"""HTTP surface for the agent, for Django to proxy to.

Starlette rather than FastAPI: the MCP SDK already brings Starlette in, and the
API is two routes. Adding FastAPI for that is a dependency for nothing.

A fresh MCP session is opened per request rather than held open for the life of
the instance. The MCP server runs with `stateless_http=True` (§8) precisely
because streamable-HTTP sessions are per-instance and Cloud Run's load balancer
may route a follow-up request elsewhere, so a long-lived client session buys
nothing and breaks when an instance is recycled.
"""

import asyncio
import logging
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent.a2ui import risk_card_from_tool_calls
from agent.graph import ask, final_text
from agent.guardrail import guard_answer
from agent.mcp_client import MCP_TRANSPORT, MCP_URL, toolbox
from mcp_server.config import GEMINI_MODEL, LOCATION, PROJECT

logger = logging.getLogger(__name__)

# The service is IAM-private, but a bounded input is still the caller's contract
# rather than an assumption about it.
MAX_QUESTION_CHARS = 2000

# Hard deadline on one question, just under the site's 120s proxy timeout so
# the caller gets a structured 504 instead of a dropped connection — and a
# runaway graph cannot keep billing after the caller is gone (ECC-02).
ASK_TIMEOUT_SECONDS = float(os.environ.get("ASK_TIMEOUT_SECONDS", "110"))

# Defense-in-depth (ECC-07): on Cloud Run (K_SERVICE is platform-injected) the
# IAM front end forwards the caller's Authorization header; a request without
# one means the service was deployed --allow-unauthenticated by mistake.
# Local runs (no K_SERVICE) are unaffected.
REQUIRE_AUTH_HEADER = bool(os.environ.get("K_SERVICE")) and (
    os.environ.get("ALLOW_UNAUTHENTICATED", "").lower() != "true"
)


async def health(request: Request) -> JSONResponse:
    """Shallow by design: no Vertex, no MCP, no BigQuery.

    A deep check would bill on every probe of a scale-to-zero service and would
    mark the container unhealthy whenever a dependency blipped.
    """
    return JSONResponse(
        {
            "status": "ok",
            "project": PROJECT,
            "location": LOCATION,
            "model": GEMINI_MODEL,
            "mcp_transport": MCP_TRANSPORT,
            "mcp_url": MCP_URL or None,
        }
    )


async def ask_route(request: Request) -> JSONResponse:
    if REQUIRE_AUTH_HEADER and "authorization" not in request.headers:
        return JSONResponse(
            {"error": "unauthenticated",
             "message": "This service requires an identity token."},
            status_code=401,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        return JSONResponse(
            {"error": "invalid_request", "message": "'question' must be a non-empty string"},
            status_code=400,
        )
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {
                "error": "question_too_long",
                "message": f"Limit is {MAX_QUESTION_CHARS} characters.",
            },
            status_code=413,
        )

    try:
        async with asyncio.timeout(ASK_TIMEOUT_SECONDS):
            async with toolbox() as box:
                state = await ask(box, question)
    except TimeoutError:
        logger.error("agent /ask timed out after %.0fs", ASK_TIMEOUT_SECONDS)
        return JSONResponse(
            {"error": "timeout",
             "message": f"The agent did not answer within {ASK_TIMEOUT_SECONDS:.0f}s."},
            status_code=504,
        )
    except Exception as exc:
        # Never let an infrastructure failure surface as a plausible answer.
        # The MCP SDK raises asyncio.ExceptionGroup when a transport task
        # fails; unwrap it so the real cause is logged and returned instead of
        # hiding behind "unhandled errors in a TaskGroup".
        cause = exc
        if isinstance(exc, BaseExceptionGroup):
            detail = " | ".join(str(e) for e in exc.exceptions)
            cause = RuntimeError(f"{type(exc).__name__}: {detail}")
            cause.__cause__ = exc  # keep the group as the logged root cause
        logger.error("agent /ask failed", exc_info=cause)
        return JSONResponse(
            {"error": "agent_failed", "message": f"{type(cause).__name__}: {cause}"},
            status_code=502,
        )

    # The A2UI envelope is composed here rather than in the browser so the
    # rendering contract is testable in Python and versioned with the agent.
    # It is None when the run answered without predicting, which the caller
    # must treat as "show the prose" rather than "show an empty card".
    card = risk_card_from_tool_calls(state["tool_calls"])

    # Deterministic post-hoc guardrails (P4): the LLM proposes, code disposes.
    # The served answer is the guarded one; flags are returned for observability
    # (they surface in Langfuse) and so a client can choose to render a note.
    text = final_text(state)
    guarded = guard_answer(text, state["tool_calls"])

    return JSONResponse(
        {
            "question": question,
            "answer": guarded["answer"],
            "guardrail_flags": guarded["flags"],
            "tool_calls": state["tool_calls"],
            "a2ui": card,
            "model": GEMINI_MODEL,
            "mcp_transport": MCP_TRANSPORT,
        }
    )


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/ask", ask_route, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    # Cloud Run injects PORT. Do not hardcode it.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
