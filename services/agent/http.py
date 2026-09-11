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
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from services.agent.a2ui import compose_presentation
from services.agent.contracts import (
    AgentRequestError,
    AgentResponseError,
    parse_agent_request,
    validate_agent_success,
)
from services.agent.graph import ask, final_text
from services.agent.guardrail import guard_answer
from services.agent.mcp_client import MCP_TRANSPORT, toolbox
from services.mcp.config import GEMINI_MODEL
from services.mcp.runtime import requires_cloud_run_auth, timeout_chain

logger = logging.getLogger(__name__)

# The service is IAM-private, but a bounded input is still the caller's contract
# rather than an assumption about it.
MAX_QUESTION_CHARS = 2000

# Hard deadline on one question, just under the site's 120s proxy timeout so
# the caller gets a structured 504 instead of a dropped connection — and a
# runaway graph cannot keep billing after the caller is gone (ECC-02).
_, ASK_TIMEOUT_SECONDS = timeout_chain()

# Defense-in-depth (ECC-07): local and Cloud Run services share this decision.
REQUIRE_AUTH_HEADER = requires_cloud_run_auth()


async def health(request: Request) -> JSONResponse:
    """Shallow by design: no Vertex, no MCP, no BigQuery.

    A deep check would bill on every probe of a scale-to-zero service and would
    mark the container unhealthy whenever a dependency blipped.

    No project/region/MCP URL (ECC-06): the route is unauthenticated at the
    app layer, and internal topology must not leak to a direct caller.
    """
    return JSONResponse(
        {
            "status": "ok",
            "model": GEMINI_MODEL,
            "mcp_transport": MCP_TRANSPORT,
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

    try:
        question = parse_agent_request(
            body, max_question_chars=MAX_QUESTION_CHARS
        )["question"]
    except AgentRequestError as exc:
        return JSONResponse(
            {"error": exc.code, "message": exc.message},
            status_code=exc.status_code,
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
        # fails; unwrap it so the real cause is logged instead of hiding
        # behind "unhandled errors in a TaskGroup".
        cause = exc
        if isinstance(exc, BaseExceptionGroup):
            detail = " | ".join(str(e) for e in exc.exceptions)
            cause = RuntimeError(f"{type(exc).__name__}: {detail}")
            cause.__cause__ = exc  # keep the group as the logged root cause
        # Detail stays server-side (ECC-06): exception text routinely embeds
        # the private MCP URL, IAM/audience detail and table names. The caller
        # gets a stable code + correlation id that pairs with the log line.
        correlation_id = uuid.uuid4().hex[:12]
        logger.error("agent /ask failed [%s]", correlation_id, exc_info=cause)
        return JSONResponse(
            {"error": "agent_failed",
             "message": "The agent failed to answer. Please retry.",
             "correlation_id": correlation_id},
            status_code=502,
        )

    # Deterministic post-hoc guardrails (P4): the LLM proposes, code disposes.
    # The served answer is the guarded one; flags are returned for observability
    # (they surface in Langfuse) and so a client can choose to render a note.
    text = final_text(state)
    if not text:
        # Typically MAX_TOKENS spent entirely on thinking — the model returns
        # empty text and raises nothing. A stale fragment must not ship as the
        # answer (ECC-12).
        logger.error("agent produced no final answer text")
        return JSONResponse(
            {"error": "answer_unavailable",
             "message": "The agent did not produce an answer. Please retry."},
            status_code=502,
        )
    guarded = guard_answer(text, state["tool_calls"])

    # tool_calls are trimmed to what the site's canvas composition reads
    # (name + response) — the raw arguments never need to reach the browser
    # (ECC-08). Guardrails above ran on the full records.
    trimmed_calls = [
        {"name": tc["name"], "response": tc["response"]}
        for tc in state["tool_calls"]
    ]

    # The presentation contract is composed HERE, not in the BFF: citation
    # renumbering, source resolution, and the A2UI canvas are evidence
    # semantics — they belong to the layer that ran the guardrails and saw the
    # tool evidence. Django passes them through; the browser renders them.
    presentation = compose_presentation(question, guarded["answer"], trimmed_calls)

    payload = {
        "question": question,
        "answer": presentation["answer"],
        "guardrail_flags": guarded["flags"],
        "tool_calls": trimmed_calls,
        "a2ui": presentation["a2ui"],
        "sources": presentation["sources"],
        "model": GEMINI_MODEL,
        "mcp_transport": MCP_TRANSPORT,
    }
    try:
        validate_agent_success(payload)
    except AgentResponseError:
        logger.error("agent produced a payload outside its success contract")
        return JSONResponse(
            {"error": "answer_unavailable",
             "message": "The agent did not produce an answer. Please retry."},
            status_code=502,
        )
    return JSONResponse(payload)


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
