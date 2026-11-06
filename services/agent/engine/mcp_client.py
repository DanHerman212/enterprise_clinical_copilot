"""Hand-rolled MCP -> Gemini tool adapter.

from __future__ import annotations

Why hand-rolled rather than `langchain-mcp-adapters` (probed 2026-07-30, v0.3.1):
that package declares `mcp>=1.24.0` with **no upper bound**, so pip installs it
happily next to our `mcp==2.0.0` — then it fails at import, because
`tools.py` imports `mcp.server.fastmcp` (removed in 2.0) and reads
`tool.inputSchema` (renamed to `input_schema`). Its `sessions.py` *has* been
migrated, so the package is half-ported. The only alternative was pinning
`mcp<2.0`, which would invalidate the server built in §6-§8 and the deployed
image. The adapter surface we actually need is this file.

Transport is selected by environment so local development stays on stdio:
`MCP_TRANSPORT=stdio` (default) or `http` with `MCP_URL` pointing at the Cloud
Run service. `toolbox()` picks; the graph never knows which it got.
"""

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from services.agent.routing.admission_context import PREFLIGHT_TOOLS
import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from services.agent.postprocess import citations
from services.agent.postprocess import evidence_state as session_evidence
from services.agent.routing import routing
from shared.labels import humanize
from services.agent.telemetry import observability, stages
from services.agent.telemetry.stages import emit as _emit
from shared.errors import AgentResponseError
from typing import Any, Callable, Sequence, TypedDict

logger = logging.getLogger(__name__)

from services.mcp.contracts import ToolContractError, validate_tool_result
from services.mcp.runtime import timeout_chain

_LOG = logging.getLogger(__name__)

#: The repository root — the directory `services` and `shared` are importable from.
#:
#: Three levels up from this file: `engine` → `agent` → `services` → the root. It was
#: `parents[2]` (the `services` directory) while this module lived directly under
#: `services/agent/`, and the constant was not re-read when the module moved into
#: `engine/` (Phase 7). The child's `cwd` is what makes `python -m services.mcp.server`
#: importable, so with the wrong value the stdio transport died before the session
#: initialised — twice, in two different ways: "No module named 'services'" from inside
#: `services/`, then, once PYTHONPATH papered over that, `services/mcp/server.py`
#: shadowing the installed `mcp` SDK and importing itself.
#:
#: Measured 2026-11-02 while wiring Stage Gate 2's live verification. Nothing caught it
#: earlier because no test and no deployed route starts the stdio child: the demo runs
#: over `http`, and the live gate scripts call the retrieval layer directly.
REPO_ROOT = Path(__file__).resolve().parents[3]

MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")  # "stdio" | "http"

# The one tool whose arguments leave our system: a search string goes to NCBI.
# Named rather than derived, because what makes it special is where its arguments
# GO, and a second tool that talks to a third party should acquire the same
# treatment deliberately rather than by inheriting it.
EXTERNAL_TOOL = routing.EXTERNAL_TOOL


def _external_argument_refusal(name: str, arguments: dict, tool) -> dict | None:
    """Refuse a call that would carry a patient identifier to a third party.

    The SDK validates a payload against the tool's DECLARED arguments by dropping
    the ones it does not declare. So a caller that passes an admission alongside
    the search terms does not get an error — the identifier disappears in transit,
    and the call is made as though it had never been there. Verified live
    2026-10-12: `search_literature` with `hadm_id` in the payload returned a
    normal result.

    That silence is the problem. "No identifier leaves the service" is then true
    by accident of a validation library rather than by anything this service
    decided, and the same silence would cover a future tool that forwards its
    arguments somewhere less forgiving.

    Scoped to the external tool on purpose: an undeclared argument to a local
    retrieval tool is a harmless extra key today, and refusing every one of those
    would turn a model's formatting habit into a failed answer.
    """
    if name != EXTERNAL_TOOL:
        return None
    declared = sorted((tool.input_schema or {}).get("properties") or {})
    undeclared = sorted(set(arguments or {}) - set(declared))
    if not undeclared:
        return None
    return {
        "error": "undeclared_argument_refused",
        "message": (
            f"The search was not run: the call carried {undeclared}, which this "
            f"tool does not take and which is dropped in transit rather than "
            f"rejected. It takes only {declared}. Call it again with search "
            f"terms and nothing else — no admission, no patient details."
        ),
    }
MCP_URL = os.environ.get("MCP_URL", "")

# Gemini's Schema is a subset of JSON Schema. Pydantic emits keys it rejects
# ("title", "$schema", "additionalProperties", "default"), and an unknown key is
# a 400 at generate time, not at declaration time. Strip to the allowed set.
_ALLOWED_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum",
    "properties", "required", "items", "minimum", "maximum",
}

#: Whether the model-facing object schema is CLOSED.
#:
#: `additionalProperties: false` is the JSON-Schema way to say "these are the only keys",
#: and whether it reaches Gemini depends on this flag because the SDK's Schema message has
#: no such field: sending it is a 400 at generate time. The probe that decides is
#: `tests/agent/test_remediation_2026_11_06.py::test_the_model_schema_can_carry_a_closed_object`
#: plus one live call; until that call is made, the honest state is "the schema does not
#: claim to be closed", and the argument is withheld by omission instead — which the
#: inspectable half of the same fix does: the description no longer names it and the hub
#: forbids writing it.
_CLOSE_MODEL_SCHEMA = False


def _clean_schema(node: Any) -> Any:
    """Recursively drop schema keys Gemini does not accept."""
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == "items":
            cleaned[key] = _clean_schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _model_schema(name: str, schema: Any) -> Any:
    """The schema the MODEL is shown: no admission id, because the session already has it.

    A tool that takes one admission describes its subject, and the subject of a turn is
    fixed by the request — never by the model. Advertising `hadm_id` asked the model to
    generate a number it had no way to know and every reason to guess: measured
    2026-11-05, a probe that named a real admission read another patient's discharge
    summary during a turn about a different one, and the only thing standing in the way
    was a refusal of the attempt after it had been made.

    So the argument is removed from the copy the model reads. `MCPToolbox.call` still
    receives it — `_execute_tool_calls` injects the session's admission — so the tool's
    own signature, its validation and `_external_argument_refusal` are untouched. The
    tool still declares `hadm_id`; only the model is not told about it.

    Structured as "does this tool declare the argument" rather than as a list of tool
    names, so a patient-scoped tool added tomorrow is covered without a second registry
    (`routing` owns the names, and `search_literature` declares no admission at all).

    A tool left with no properties becomes `{"type": "object", "properties": {}}` with no
    `required` list, which is the shape LangChain and Gemini accept for a call that takes
    no arguments — `rag_search_sections`, `discharge_medications` and
    `predict_readmission` are exactly that, once the admission is gone.
    """
    cleaned = _clean_schema(schema)
    if not isinstance(cleaned, dict):
        return cleaned
    properties = cleaned.get("properties")
    if not isinstance(properties, dict) or "hadm_id" not in properties:
        return _closed(cleaned)
    remaining = {key: value for key, value in properties.items() if key != "hadm_id"}
    if not remaining:
        return _closed({"type": "object", "properties": {}})
    bound = {key: value for key, value in cleaned.items() if key != "required"}
    bound["properties"] = remaining
    required = [entry for entry in (cleaned.get("required") or [])
                if entry != "hadm_id"]
    if required:
        bound["required"] = required
    return _closed(bound)


def _closed(schema: Any) -> Any:
    """The same schema, declaring that no other keys are accepted.

    Applied on EVERY branch, including the tool left with no properties at all — that is
    the branch the model reads for `discharge_medications`, and a schema that says
    `{"type": "object", "properties": {}}` and nothing else announces no constraint,
    which is what let a decoder emit `{"hadm_id": null}` for a tool that takes no
    arguments.

    Off by default (`_CLOSE_MODEL_SCHEMA`) because the SDK's Schema message has no such
    field: whether Gemini accepts it is one live call away and unmeasured, and a schema
    keyword that 400s would break every bound tool at once. See the note on the flag.
    """
    if not _CLOSE_MODEL_SCHEMA or not isinstance(schema, dict):
        return schema
    if schema.get("type") != "object":
        return schema
    return {**schema, "additionalProperties": False}


#: The argument-and-description hygiene that does NOT depend on the API's schema subset.
#:
#: A tool description is prose in the prompt, and a description that says "Requires
#: hadm_id" tells the model the argument exists after the schema stopped advertising it.
#: The sentence is removed rather than rewritten: this function may not author clinical or
#: contract prose, and a rewritten sentence is a second claim about the tool.
_WITHHELD_ARGUMENT = "hadm_id"


def _model_description(text: str) -> str:
    """A tool's description without the sentences that name the withheld argument."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = [sentence for sentence in sentences
            if _WITHHELD_ARGUMENT not in sentence.lower()]
    return " ".join(kept).strip()


def _unwrap(payload: dict) -> dict:
    """Unwrap the single-key envelope a union return type produces.

    A tool annotated `-> Result | Error` is advertised as returning `{"result": …}`, and the
    SDK puts that envelope in the structured payload. Only an envelope whose value is itself
    an object is unwrapped: the `{"result": "<text>"}` fallback below hands back a string,
    and unwrapping that would leave the caller with something that is not a payload at all.
    """
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    return payload


def _checked(name: str, payload: dict) -> dict:
    """Check a tool result against its contract before anything downstream acts on it.

    This is the point where a result arrives from outside the process, so it is checked
    rather than trusted. A payload outside its contract becomes a structured error the model
    can report, which is the shape every other failure already takes here.
    """
    try:
        return validate_tool_result(name, payload)
    except ToolContractError:
        _LOG.error("MCP tool %s returned a payload outside its contract", name)
        return {
            "error": "invalid_tool_response",
            "message": f"The {name} tool returned a payload outside its contract.",
        }


def _payload(result: Any) -> dict:
    """Normalise a CallToolResult into the dict the tool returned."""
    if getattr(result, "structured_content", None):
        return _unwrap(result.structured_content)
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return _unwrap(json.loads(text))
            except json.JSONDecodeError:
                return {"result": text}
    return {}


#: The failures a retry can help with. `TimeoutError` and `ConnectionError` are `OSError`
#: subclasses, and anyio — which the `mcp` transport is built on — reports a broken or
#: closed stream as a `RuntimeError` subclass, so these roots cover the transport surface
#: without naming SDK internals a dependency bump would invalidate. A bare `Exception`
#: would also catch a programming error and retry it three times.
_TRANSIENT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    OSError, RuntimeError, EOFError,
)


def _transient_transport(exc: BaseException) -> bool:
    """Whether a failure is the transport misbehaving rather than the call being wrong.

    `BaseExceptionGroup` is the shape anyio reports a task-group failure in, and it is not
    an `OSError` or a `RuntimeError`, so a stream that closed mid-call arrived here as an
    opaque group and was classified as "everything else": zero retries, one log line, and a
    failed turn. The group is unwrapped: if every leaf is a transport failure the call can
    be made again; if any leaf is not, this is a bug and is not retried.

    A group holding a cancellation is never transient. `CancelledError` is a
    `BaseException`, it is the caller's decision, and retrying it would outlive the request
    that asked for it.
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    leaves: list[BaseException] = []

    def collect(error: BaseException) -> None:
        nested = getattr(error, "exceptions", None)
        if nested:
            for inner in nested:
                collect(inner)
        else:
            leaves.append(error)

    collect(exc)
    return bool(leaves) and all(
        isinstance(leaf, (*_TRANSIENT_TRANSPORT_ERRORS, BaseExceptionGroup))
        for leaf in leaves
    )

#: How much of the backoff delay may be added as jitter, as a fraction of it. Bounded so
#: the ladder still fits inside `call_timeout_seconds` and the caller's `ask` deadline.
_JITTER_FRACTION = 0.25


@dataclass
class MCPToolbox:
    """The MCP tools of one session, exposed as Gemini function declarations."""

    session: ClientSession
    # Must stay below the site's 120s upstream deadline and the agent server's
    # ASK_TIMEOUT_SECONDS — a tool that outlives its caller is pure spend. The
    # ordering against the model leg is enforced by `timeout_chain`.
    call_timeout_seconds: float = timeout_chain().tool
    _tools: dict[str, Any] = field(default_factory=dict)
    #: How many times ONE call is attempted before it is reported as failed.
    #:
    #: Three, because the failures this exists for are bursts rather than outages: a
    #: Cloud Run instance that just scaled, a stream reset between hops, a read that
    #: timed out at the transport while the tool was still inside its own deadline. A
    #: call that failed three times across a second and a half is a real failure and
    #: the model is told so — inventing a number is the outcome this must not have.
    max_attempts: int = 3
    #: The first backoff, doubled per attempt. Deliberately small: the whole retry
    #: ladder has to fit inside `call_timeout_seconds` and the caller's `ask` deadline,
    #: or a retry budget becomes a timeout instead of a recovery.
    base_delay_seconds: float = 0.25

    async def load(self) -> "MCPToolbox":
        result = await self.session.list_tools()
        self._tools = {tool.name: tool for tool in result.tools}
        if not self._tools:
            raise RuntimeError("MCP server advertised no tools")
        return self

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def model_arguments(self, name: str, arguments: dict) -> dict:
        """`arguments` without the one argument this tool's schema withheld from the model.

        The mirror of `_model_schema`: the schema the model is shown has no `hadm_id`, and
        this is the other half of that promise — an argument the model was never offered is
        not carried into the call. The admission is the session's to supply, and a call
        that arrives carrying one is a model repeating a number it should not have had.

        Narrow on purpose. It removes `hadm_id` and stops there: the question is whether
        the model supplied an argument the schema hid, not whether every key it wrote is
        declared. A tool whose schema advertises no properties still receives whatever the
        model sent, because dropping an argument a tool needs is a worse failure than
        tolerating one it ignores — `_external_argument_refusal` is the rule for the tool
        where an undeclared key actually leaves the building.

        A tool that declares no admission at all is returned untouched: nothing was
        withheld from the model, so there is nothing to take back.
        """
        tool = self._tools.get(name)
        if tool is None:
            return arguments
        declared = (tool.input_schema or {}).get("properties") or {}
        if "hadm_id" not in declared:
            return arguments
        schema = _model_schema(name, tool.input_schema)
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(properties, dict) and "hadm_id" in properties:
            return arguments
        return {key: value for key, value in (arguments or {}).items()
                if key != "hadm_id"}

    @property
    def gemini_tool(self) -> types.Tool:
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=_model_description(tool.description or ""),
                    parameters=_model_schema(tool.name, tool.input_schema),
                )
                for tool in self._tools.values()
                if tool.name not in PREFLIGHT_TOOLS
            ]
        )

    async def call(self, name: str, arguments: dict) -> dict:
        """Execute a tool. Transport failures become structured errors.

        The model is instructed to report errors rather than invent a number,
        so an error must reach it as data, not as an exception that collapses
        the graph. What reaches it is a sentence: a transport's exception text
        and the SDK's own refusal detail — pydantic validation messages carry a
        documentation URL and the offending value — are diagnostics, and they go
        to the log (ECC-21 / gap 4).
        """
        if name not in self._tools:
            return {"error": "unknown_tool", "message": f"No MCP tool named {name!r}"}

        refusal = _external_argument_refusal(name, arguments, self._tools[name])
        if refusal is not None:
            # The argument NAMES, never their values: a value here is exactly the
            # kind of thing that must not reach a log line.
            _LOG.error("MCP tool %s refused: undeclared argument(s) in the call", name)
            return refusal
        result = await self._call_with_backoff(name, arguments)
        if result is None:
            return {
                "error": "tool_call_failed",
                "message": "The tool call failed before it returned a result.",
            }

        payload = _payload(result)
        if getattr(result, "is_error", False) and "error" not in payload:
            # The tool returned no structured error of its own, so this is the
            # SDK refusing the call: invalid arguments, or the call raising
            # before the tool could shape a failure. Both read the same to a
            # caller, and the difference is in the log.
            _LOG.error("MCP tool %s was refused before it ran: %s", name, payload)
            payload = {
                "error": "tool_call_failed",
                "message": "The tool refused the call before it ran.",
            }
        return _checked(name, payload)

    async def _call_with_backoff(self, name: str, arguments: dict) -> Any | None:
        """One transport call, retried while the failure looks transient.

        Returns the SDK's result, or `None` once the attempts are spent. `None` rather
        than an exception because the caller's contract is a dict the model can report —
        an exception here would collapse the graph and lose the answer.

        Two classes of failure, and only one is retried:

          * **transient** — a timeout, a refused or reset connection, a stream the `mcp`
            transport reports as a `RuntimeError`. Nothing was decided; the same call may
            well succeed. `TimeoutError` and `ConnectionError` are `OSError` subclasses,
            and anyio's broken/closed resource errors descend from `RuntimeError`, so the
            two roots below cover the transport surface without a list of SDK internals
            that a dependency bump would invalidate.
          * **everything else** — a bug, a schema the SDK rejected, a call that cannot
            work. Retried zero times, logged once, reported to the model.

        `asyncio.CancelledError` is a `BaseException` and is re-raised first regardless,
        because a cancelled turn is the caller's decision and retrying it would outlive
        the request that asked for it.

        The delay doubles per attempt and carries bounded jitter: two turns that hit the
        same overloaded instance should not retry in lockstep. `base_delay_seconds=0.0`
        makes the ladder a no-op, which is what a unit test wants.
        """
        attempts = max(1, self.max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return await self.session.call_tool(
                    name, arguments, read_timeout_seconds=self.call_timeout_seconds
                )
            except asyncio.CancelledError:
                raise
            except BaseExceptionGroup as group:
                if not _transient_transport(group):
                    _LOG.error("MCP tool %s call failed in a task group", name,
                               exc_info=group)
                    raise
                if attempt >= attempts:
                    _LOG.error(
                        "MCP tool %s call failed after %d attempt(s)", name, attempts,
                        exc_info=group,
                    )
                    return None
                delay = self.base_delay_seconds * (2 ** (attempt - 1))
                if delay > 0:
                    delay += random.uniform(0.0, delay * _JITTER_FRACTION)
                    await asyncio.sleep(delay)
                _LOG.warning(
                    "MCP tool %s attempt %d/%d failed in a transport group; "
                    "retrying in %.2fs", name, attempt, attempts, delay,
                )
            except _TRANSIENT_TRANSPORT_ERRORS as exc:
                if attempt >= attempts:
                    _LOG.error(
                        "MCP tool %s call failed after %d attempt(s)", name, attempts,
                        exc_info=exc,
                    )
                    return None
                delay = self.base_delay_seconds * (2 ** (attempt - 1))
                if delay > 0:
                    delay += random.uniform(0.0, delay * _JITTER_FRACTION)
                    await asyncio.sleep(delay)
                _LOG.warning(
                    "MCP tool %s attempt %d/%d failed transiently; retrying in %.2fs: %s",
                    name, attempt, attempts, delay, exc,
                )
            except Exception as exc:
                _LOG.error("MCP tool %s call failed", name, exc_info=exc)
                return None
        return None


def _mcp_child_environment() -> dict[str, str]:
    """The environment the MCP server subprocess starts with.

    `StdioServerParameters` without `env` passes the SDK's own default allow-list to
    the child — on this platform six keys: HOME, LOGNAME, PATH, SHELL, TERM, USER.
    That is the right default for a server someone else wrote. Ours is this package's
    second entry point and reads its configuration out of the environment, so the
    filter was quietly wrong for it.

    Quietly is the problem. Nearly every setting the server reads has a pinned or
    corpus-document fallback, so prediction and retrieval kept working while
    `NCBI_EMAIL` — the one credential with no fallback — went missing, and the
    literature tool reported itself degraded rather than broken. A search that could
    have run was announced to the clinician as "temporarily unavailable" in the only
    environment a developer can watch. Confirmed 2026-10-12 by printing the child's
    environment: NCBI_EMAIL absent, six keys present.

    So the child inherits the parent's environment. The parent has already loaded
    `.env`, and the child is the same code in the same trust domain; a key missing
    here should mean a misconfigured parent, not a differently-shaped child.
    """
    return dict(os.environ)


@contextlib.asynccontextmanager
async def stdio_toolbox(python: str | None = None):
    """Run the MCP server as a subprocess and yield a loaded toolbox.

    No network and no auth, so a failure in §10 can only be agent-shaped.
    """
    params = StdioServerParameters(
        command=python or sys.executable,
        args=["-m", "services.mcp.server", "--transport", "stdio"],
        cwd=str(REPO_ROOT),
        env=_mcp_child_environment(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield await MCPToolbox(session).load()


def id_token(audience: str) -> str:
    """Mint an ID token for a private Cloud Run service.

    The audience must be the **service URL**, not the /mcp path. A mismatched
    audience produces a 401 identical to a missing IAM binding, and the
    audience is the more common cause.

    On Cloud Run the metadata server mints this in-process. Locally, ADC is a
    *user* credential, which cannot mint an ID token for an arbitrary audience.
    Check the credential type first rather than letting fetch_id_token fail:
    its failure path probes the GCE metadata server and stalls for seconds
    before raising DefaultCredentialsError, on every local run.
    """
    import google.auth
    import google.auth.transport.requests
    import google.oauth2.credentials
    import google.oauth2.id_token

    credentials, _ = google.auth.default()

    if not isinstance(credentials, google.oauth2.credentials.Credentials):
        # Service account or metadata server: in-process, no fork.
        request = google.auth.transport.requests.Request()
        return google.oauth2.id_token.fetch_id_token(request, audience)

    # Local user credential. NOTE: this forks. Keep it early — once
    # google-cloud-aiplatform has opened gRPC channels, fork+exec can deadlock
    # in gRPC's atfork handler. See the gsutil hang in mlops.
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return result.stdout.strip()


# ID tokens live one hour; refresh with headroom. Cached per audience so a
# fresh per-request toolbox does not mint (or fork gcloud for) a new token on
# every /ask.
_TOKEN_TTL_SECONDS = 45 * 60
_token_cache: dict[str, tuple[str, float]] = {}


def _cached_id_token(audience: str) -> str:
    token, fresh_until = _token_cache.get(audience, ("", 0.0))
    if time.monotonic() >= fresh_until:
        token = id_token(audience)
        _token_cache[audience] = (token, time.monotonic() + _TOKEN_TTL_SECONDS)
    return token


@contextlib.asynccontextmanager
async def http_toolbox(url: str | None = None):
    """Connect to the deployed MCP server over authenticated streamable HTTP."""
    base = (url or MCP_URL).rstrip("/")
    if not base:
        raise RuntimeError("MCP_URL is not set; required when MCP_TRANSPORT=http")

    # Token minting is blocking I/O (metadata server or a gcloud fork); off the
    # event loop so concurrent requests aren't stalled behind it (ECC-09).
    token = await asyncio.to_thread(_cached_id_token, base)
    headers = {"Authorization": f"Bearer {token}"}

    # SDK 2.0 takes an http_client, not headers — and it must be httpx2, which
    # the SDK vendors alongside the ordinary httpx other libraries pull in.
    # Transport timeout sits just above the per-call read timeout so the tool
    # deadline fires first with a structured error.
    #
    # Every exit below is wrapped, because the failure this exists for happens in an
    # `__aexit__` and not in a call. Measured 2026-11-06, twice: a lab turn died with
    # `ExceptionGroup: unhandled errors in a TaskGroup` raised by the HTTP stream's
    # teardown after the tool result had already been captured and used. The turn's answer
    # was thrown away by the closing of a socket. A session that tears down badly AFTER the
    # body finished has nothing left to report: the caller already has what it asked for,
    # so the teardown is logged and the body's result stands. A teardown that happens while
    # the body is failing is not swallowed — `finished` is only set when the body returned
    # normally, and anything else is re-raised untouched, including cancellation.
    finished = False
    try:
        async with httpx2.AsyncClient(headers=headers, timeout=110) as http_client:
            async with streamable_http_client(f"{base}/mcp", http_client=http_client) as (
                read,
                write,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    box = await MCPToolbox(session).load()
                    try:
                        yield box
                    except BaseException:
                        raise
                    else:
                        finished = True
    except BaseExceptionGroup as group:
        if not finished:
            raise
        _LOG.warning(
            "MCP session teardown failed after the turn completed (%s); "
            "discarding the teardown error, the result is already returned",
            type(group).__name__,
        )


@contextlib.asynccontextmanager
async def toolbox(transport: str | None = None, url: str | None = None):
    """Transport-agnostic entry point. The graph never knows which it got."""
    chosen = (transport or MCP_TRANSPORT).lower()
    if chosen == "stdio":
        async with stdio_toolbox() as box:
            yield box
    elif chosen == "http":
        async with http_toolbox(url) as box:
            yield box
    else:
        raise ValueError(f"Unknown MCP transport {chosen!r}; expected 'stdio' or 'http'")


# ===========================================================================
# The tool-execution seam, moved down from `graph.py`, `deterministic.py`
# and `ingress/contracts.py` (Phase 7).
#
# The executor produces `RecordedToolCall`s and both the reasoning loop and
# the command path drive it, so it cannot live in either of them: that was
# the `graph` <-> `deterministic` cycle. The record type and its validator
# come with it, which is also what stops the engine importing ingress.
# ===========================================================================

MAX_PASSAGE_CHARS = 4000



MAX_TOOL_CALLS_PER_TURN = 5



#: How many times ONE tool may run in one turn.
#:
#: The turn budget above is a total, and a total cannot tell a turn that read five
#: different things from a turn that asked one tool five questions. The second is what
#: happened: a literature question on 2026-10-30
#: (`079be88f…`) made one `rag_search` and four `search_literature` calls with
#: reworded queries, and four refinements of one search are a loop that the model
#: describes as thoroughness. Two is the allowance for a genuine second attempt — the
#: first search that comes back empty or off-target — and no more.
#:
#: A per-tool cap rather than a literature-specific one: `get_lab_trends` and
#: `rag_search` are already held to one call each by `ONCE_PER_TURN`, and the tool that
#: is not capped is the one whose arguments are free text, which is every tool that can
#: be re-asked in different words.
PER_TOOL_CALL_LIMIT = 2



ONCE_PER_TURN: frozenset[str] = routing.ONCE_PER_TURN_TOOLS



_DELIMITER_RE = re.compile(r"<\s*/?\s*tool_result", re.IGNORECASE)



class _MCPTool(BaseTool):
    """One MCP tool, exposed to the model as a LangChain tool.

    `_arun` calls through to `MCPToolbox.call`, which returns a plain dict
    (already normalised / JSON-safe, and errors are structured `{"error": ...}`
    payloads — never exceptions). Returning the dict as the tool's content
    keeps that error contract intact for the agent's prompts.py guardrails.
    """

    name: str
    description: str
    box: MCPToolbox

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError("MCP tools are async-only")

    async def _arun(self, **kwargs: Any) -> Any:
        return await self.box.call(self.name, dict(kwargs))



def _escape_delimiter(text: str) -> str:
    """Neutralise the wrapper's delimiter inside the text the wrapper carries."""
    if _DELIMITER_RE.search(text):
        logger.warning(
            "tool result carries the wrapper's own delimiter; escaping it "
            "rather than letting it close the block"
        )
    return _DELIMITER_RE.sub(lambda match: "&lt;" + match.group(0)[1:], text)



def _call_key(name: str, arguments: dict[str, Any]) -> str:
    """A stable identity for one call, so a repeat can be recognised.

    Sorted keys rather than the dict itself: two calls carrying the same arguments in a
    different order are the same call, and comparing them as written would let a model
    defeat the check by reordering its own JSON.
    """
    return f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"


def _with_session_admission(name: str, arguments: dict[str, Any],
                            hadm_id: int | None) -> dict[str, Any]:
    """One call's arguments, with the session's admission where the tool expects it.

    The model is no longer shown `hadm_id` (`_model_schema`), so a patient-scoped call
    arrives without one and is completed here. Three properties this holds:

      * a tool that publishes records rather than chart evidence — `search_literature` —
        takes no admission and is never given one;
      * an admission the model DID write is left exactly as written, so the refusal in
        `_outbound_patient_identifier` can refuse it rather than have it replaced by the
        correct value;
      * a turn with no admission injects nothing, and the guard refuses the call. Filling
        in `None` would send a call the tool cannot answer and blame the tool.
    """
    if hadm_id is None or name in routing.LITERATURE_TOOLS:
        return arguments
    if arguments.get("hadm_id") is not None:
        return arguments
    return {**arguments, "hadm_id": hadm_id}



def _passage_total(payload: Any) -> int:
    """How many citation numbers a payload consumes.

    One definition, in `citations`, because two layers count and they must agree: this one
    advances the running number the model is shown, and `citations.citable_passages`
    builds the array a marker resolves against.
    """
    return citations.passage_total(payload)



def _render_passage(passage: dict[str, Any], *, keep_id: bool = False) -> dict[str, Any]:
    """Return the passage as the model will see it, or refuse it visibly.

    `id` is DROPPED from the model's copy. It is a machine name — `MT-2616-DS_discharge_
    medications_2` — and its trailing digit is a LOCAL position inside one tool result.
    Beside a global `^[n]` it is a second, disagreeing index: the trailing digit counts
    positions INSIDE one tool result, so the second medication passage is `2` by its id
    and `^[3]` in the thread that shows it. The prompt already forbids writing
    the id into an answer, so the model had no use for it and one way to be misled by it.
    `keep_id` exists for the tests that assert what the tool returned is what is recorded;
    nothing on the model path sets it.
    """
    rendered = dict(passage)
    if not keep_id:
        rendered.pop("id", None)
    text = passage.get("text")
    if not isinstance(text, str) or len(text) <= MAX_PASSAGE_CHARS:
        return rendered
    logger.warning(
        "refusing passage %s: %d characters, over the %d-character ceiling",
        passage.get("id"), len(text), MAX_PASSAGE_CHARS,
    )
    return {**rendered, "text": (
        f"[refused: {len(text)} characters, over the "
        f"{MAX_PASSAGE_CHARS}-character ceiling for one passage]"
    )}



def _render_tool_result(payload: Any, *, offset: int = 0,
                        ref_ids: Sequence[int] | None = None) -> str:
    """Render a tool result as the body the wrapper hands the model.

    The projection is built here rather than applied to the payload itself: the
    same payload is what the execution record, the caller and the browser's
    citation lookup use, and those should carry what the tool actually returned.
    A refused passage keeps its slot, with the refusal in place of its text, so
    the positions every citation refers to do not shift.

    `offset` is how many citable units the model has already been shown in this
    conversation. Citation numbers are global across tool results and across the
    replayed turns, because that is the array the citation layer resolves against
    (`citations.citable_passages` concatenates the same units in the same order).
    Restarting at 1 per result meant `^[1]` named the first passage of whichever result
    the reader happened to look at (TF-10, 2026-10-23).

    `ref_ids` is that numbering when the session record already decided it. The record
    assigned a position to every unit it ingested and reports which, in payload order
    (`IngestResult.slot_ids`), so a passage the conversation retrieved in an earlier turn
    is stamped with the position it already has rather than with a second one counted
    from `offset`. Given them, `offset` is not consulted.
    """
    if isinstance(payload, str):
        return _escape_delimiter(payload)
    # Numbering is `citations`': it owns what a citable unit is (`citable_passages`) and
    # it is the module that will later resolve or delete a marker, so it also stamps the
    # numbers. `dress` is the shaping only this layer knows about — dropping the machine
    # id, refusing an over-long passage — and it runs before the numbers are attached.
    rendered = citations.render_numbered(payload, offset, dress=_render_passage,
                                         ref_ids=ref_ids)
    return _escape_delimiter(json.dumps(rendered, ensure_ascii=False))



def _tools(toolbox: MCPToolbox,
           allowed: frozenset[str] | None = None) -> list[BaseTool]:
    """Build one LangChain tool per MCP tool, sharing the toolbox.

    Each MCP `input_schema` is declared to the model as the tool's
    `args_schema` (Gemini-safe subset), so `bind_tools` advertises the real
    parameter names and types instead of leaving the model to guess them from
    prose (ECC-10) — which is also what made the old kwargs-flatten heuristic
    necessary.

    `allowed` is the free-text boundary (UX Phase 1B): the reasoning loop is
    handed only these, so a tool that belongs to a command is not merely refused,
    it is absent. `None` means every tool the server advertises.
    """
    return [
        _MCPTool(
            name=name,
            description=_model_description(tool.description or ""),
            args_schema=_model_schema(name, tool.input_schema),
            box=toolbox,
        )
        for name, tool in toolbox._tools.items()
        if allowed is None or name in allowed
    ]



class RecordedToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    response: dict[str, Any]



def validate_recorded_tool_call(payload: Any) -> RecordedToolCall:
    """Validate the internal trace record before it enters agent state."""
    if not isinstance(payload, dict):
        raise AgentResponseError("Recorded tool call is malformed.")
    if not isinstance(payload.get("name"), str):
        raise AgentResponseError("Recorded tool call name is invalid.")
    if not isinstance(payload.get("args"), dict):
        raise AgentResponseError("Recorded tool call arguments are invalid.")
    if not isinstance(payload.get("response"), dict):
        raise AgentResponseError("Recorded tool call response is invalid.")
    return payload



def _is_number(value: Any) -> bool:
    """A real number, rather than a bool wearing one's type."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)



def _presented_factor(factor: Any) -> Any:
    """One risk driver, named and directed, with the patient's own value beside it.

    Rebuilt field by field rather than copied and edited, because the field that has to
    go is the one a `{**factor}` spread would keep: the contribution magnitude. It is not
    withheld for want of a use — the widget draws it as a bar length — but because it is
    not this turn's to state, and a raw contribution is quotable.

    Three fields at most: the humanized feature name, which is the name both the prose and
    the bars use; the direction the model attached to it; and the patient's own value for
    that feature, at the precision the tool publishes it with.

    The value is passed through as it arrives rather than re-floated. The tool decides the
    precision (`prediction._clinical_value` — two decimals, integers as integers), and
    re-floating it here would print a whole number as `42.0` and hand a re-derived form of
    it to a prompt whose rule is to quote the served value exactly. What the model reads
    and what the guardrail verifies against are then the same number.

    The value used to be left out of here, on the argument that it was "already stated in
    the `<admission_metadata>` block". That was true of the eleven fields the manifest
    renders — age, admission type, prior utilisation — and false of every clinical input
    the booster is scored on. `rdw_max`, the eight laboratory anchors, the procedure and
    medication counts and `oncology_flag` are attributed by the model and stated by
    nothing: not the manifest, not a note passage, and not this payload. Asked what in the
    chart explained an RDW driver, the agent searched the notes for a feature name no
    discharge summary writes, found nothing, and answered — correctly, of the three things
    it had been given — that the chart did not document the value (reported 2026-11-02).

    Stated rather than omitted, and stated as the record's number. `prior_inpatient_days`
    arrives as 16.0417 because that is the encoded row; the prompt asks for it verbatim and
    `guardrail.normalise_factor_values` rewrites it to the number a clinician would write,
    which is the same rule the manifest's own values have always been under.
    """
    if not isinstance(factor, dict):
        return factor
    presented: dict[str, Any] = {
        "feature": humanize(str(factor.get("feature", ""))),
    }
    if isinstance(factor.get("direction"), str):
        presented["direction"] = factor["direction"]
    value = factor.get("value")
    if _is_number(value):
        presented["value"] = value
    return presented


def _presented_prediction(name: str, payload: Any) -> Any:
    """The prediction result as the interface shows it, for the model to read.

    The model has never seen a feature key, and the keys are not words a clinician reads:
    the widget draws "red cell distribution width (RDW)" where the result says `rdw_max`.
    The mapping already exists (`feature_labels.py`) and the interface uses it, so the
    substitution is made here rather than requested from the model — a translation the
    model performs is a translation the model can invent, and the prose and the bars have
    to name a factor the same way.

    `humanize`, not `label_for`. Both read the same curated map, so they agree on every
    feature it names; they part only in the fallback, where `label_for` hands back the raw
    key and `humanize` title-cases it. The widget draws `humanize`, so the prose has to be
    shown `humanize` too — otherwise a feature the map has not caught yet reads as
    "Rdw Max" on its bar and `rdw_max` in the sentence describing it. One factor, one name,
    whichever name the map happens to know.

    The numbers are STATED, not handed over. This projection used to pass the served
    floats through and ask the prompt to render them in the right form; the prompt asked
    three times and the leak survived all three, because a float the model is holding is
    a float it can paste. So the copy the model reads carries `probability_display` and
    `threshold_display` — the score and the threshold written the way the widget prints
    them — and carries no `probability`, `threshold` or `base_value` at all.

    Withheld rather than rounded. A display string is the same number written as the
    reader will see it written, which is a different thing from a rounded float the model
    then has to print a digit of: this projection used to pre-round, and given 0.33 the
    model wrote "33.0%" beside a widget drawing 33.2%, which is why the rounding was
    removed in the first place. Given "33.2%" there is no digit to lose.

    The drivers keep their names, their direction and the patient's own value for the
    feature; what they lose is the magnitude. The narrative says which factors led, what
    this admission recorded for them, and what they mean; the widget draws how much. A raw
    contribution was quotable too, and a number the guardrail does not recognise as the
    score is stripped and shown as a safety swap — so withholding it also removes a way
    for a good answer to be visibly withdrawn from a reader mid-sentence.

    The value is stated here because for a clinical feature this is the only place that
    could state it: `direction` is measured against a population baseline, so it is not
    evidence that the patient has the factor, and the manifest block carries the
    administrative row rather than the scored laboratory inputs. A record whose
    `<admission_metadata>` reads `recent_ed_visits: 0` can still be told the factor
    increased its estimate, and with the value beside the direction the prose can say both
    things about one factor without guessing at either.

    `EHR_FORMATTING` names both display strings, and the guardrail still checks the prose
    against the SERVED value, because it reads the recorded call and not this copy. It
    accepts the display strings: one decimal of a percentage is exactly half a unit in
    its own last place, which is the tolerance `verify_risk_numbers` allows.

    Applied to the model's copy only. The recorded call keeps what the service
    returned, so the execution record and the caller's payload still say what the
    tool said.
    """
    if name != "predict_readmission" or not isinstance(payload, dict):
        return payload
    projected = dict(payload)
    # The two numbers a clinician can quote, in the form the interface writes them.
    for field, display in (("probability", "probability_display"),
                           ("threshold", "threshold_display")):
        value = payload.get(field)
        if _is_number(value):
            projected[display] = f"{float(value) * 100:.1f}%"

    # ...and the floats they were derived from, removed rather than described as
    # unquotable. A prompt rule is a request; a key that is not there is not.
    for field in ("probability", "threshold", "base_value"):
        projected.pop(field, None)

    factors = payload.get("top_factors")
    if isinstance(factors, list):
        projected["top_factors"] = [_presented_factor(factor) for factor in factors]
    return projected




def _presented_for_model(name: str, payload: Any) -> Any:
    """The tool result in the form the MODEL reads. Never the form the caller records.

    One tool needs a translation the model must not perform itself: the estimate, whose
    floats are replaced by the display strings the widget prints and whose contribution
    magnitudes are withheld. Everything else — note passages, medication lists, lab
    series — passes through exactly as the tool returned it.

    The lab series used to be rewritten here, carrying an ordinal
    `_abnormality_rank` computed from how far each analyte sat outside its reference
    interval. That is gone: the panel reaches the model as the tool wrote it, and the
    prompt asks for the comparison in prose. Nothing is cross-referenced against the
    patient either, so the same call renders the same way whichever route asked for it.
    """
    return _presented_prediction(name, payload)



def _presented_call(call: dict) -> dict:
    """The call as the model was shown it, for building a session record.

    The record renderer prints what it holds into the model's window, so it must hold
    what the model is allowed to read. `_presented_for_model` is what withholds the
    estimate's floats and the driver magnitudes; a record built from the raw response
    would hand both back on the next model round, which is the leak the projection exists
    to close.

    The record's own DELIMITERS are not handled here. `evidence_state.render_for_prompt`
    escapes them at the point it prints them, which is the only place that can cover every
    value it interpolates — including the section names it takes from the model's own
    arguments, which never pass through a payload projection at all.
    """
    return {
        **call,
        "response": _record_text(
            _presented_for_model(str(call.get("name") or ""), call.get("response"))),
    }


def _record_text(payload: Any) -> Any:
    """A payload as the session record may hold it, for ingestion only.

    One shaping, and it has to happen before ingestion because the record IS the model's
    copy: an over-long passage is refused in place, exactly as `_render_passage` refuses it
    for a live result, so a whole-note chunk cannot put its full text into the window on
    every round. The refusal keeps the passage's slot, so positions do not shift; `id` is
    kept here because the record's identity for a note passage is the passage id.

    Applied to the copy the record holds, never to the copy `recorded` stores or the one a
    live `ToolMessage` carries: rewriting the tool's own result would make the record
    disagree with what the tool said. A payload that is not a dict is left alone, which is
    also what `ingest_tool_output` does with it.
    """
    if not isinstance(payload, dict):
        return payload
    passages = payload.get("passages")
    if not isinstance(passages, list):
        return payload
    return {**payload, "passages": [
        _render_passage(passage, keep_id=True) if isinstance(passage, dict)
        else passage
        for passage in passages
    ]}


async def _execute_tool_calls(
    toolbox: MCPToolbox, calls: list[dict],
    on_event: Callable[[dict[str, Any]], None] | None = None,
    project: Callable[[str, Any], Any] | None = None,
    guard: Callable[[str, dict], dict | None] | None = None,
    already_called: list[dict] | None = None,
    traced: bool = False,
    order: Callable[[str, dict], dict] | None = None,
    cite_offset: int = 0,
    used_up: frozenset[str] = frozenset(),
    evidence_state: session_evidence.SessionEvidenceState | None = None,
    hadm_id: int | None = None,
    attempted: int = 0,
) -> tuple[list[BaseMessage], list[RecordedToolCall]]:
    """Run one turn's tool calls, refusing those beyond the per-turn budget.

    `project` rewrites what the MODEL is shown for a call, and nothing else: the
    recorded call, the execution record, the caller's payload and the browser's
    citation lookup all carry what the tool actually returned. It exists so a
    deterministic command can hand the formatting turn the same field names the
    interface draws with, without that substitution becoming a lie about what the
    tool said.

    `guard` may refuse a call before it is dispatched, returning the error payload
    the model should see instead. A refused call announces nothing: the stages
    describe work that happened, and this is work that did not.

    `already_called` is the calls THIS TURN has already made whose results are in the
    transcript, and passing it is what makes the budget a per-turn budget. The reasoning
    loop invokes this once per iteration with the calls of a single model turn, so the
    `index` below is 0 every time — and an index-based budget therefore never tripped. A
    turn made eight calls against a budget of five, looped until the graph's recursion
    limit, and settled as an EMPTY ANSWER rather than a failure (observed live 2026-10-16,
    twice, on a lab question).

    `used_up` is the second line of the `ONCE_PER_TURN` cap. Not binding a spent tool is
    the first, and it is the one that holds; this catches the case the budget comments
    already name — a model that emits a call for a tool that is absent from its request.
    Such a call is refused rather than run, and announces no stage (rule 3).

    `evidence_state` is the session record, and passing it is what makes the numbering the
    RECORD's rather than a running total. Each payload is ingested before it is rendered,
    so `_render_tool_result` can stamp the positions the record actually gave the units:
    a passage this conversation retrieved in an earlier turn keeps its position instead of
    being handed a second one. `numbered` then follows the record's high-water mark rather
    than a payload sum, so the two cannot drift apart even when a call is deduplicated.

    `hadm_id` is the session's admission, and it is what makes the model's job a smaller
    one. The schema the model reads carries no `hadm_id` (`_model_schema`), so a call
    arrives without one; it is added here, after the guard has seen what the model
    actually wrote. Order matters and is the whole design: the guard must see a
    HALLUCINATED admission — that is the only way the attempt is refused rather than
    quietly replaced by the correct one — and the tool must receive a complete argument
    list. So `guard` gets the model's arguments verbatim, and `toolbox.call` gets them with
    the session's admission filled in.

    `attempted` is how many calls this turn has already ASKED for, refusals included, and
    it is what bounds the loop now that `recorded` holds only executed calls. The two
    numbers answer two different questions: `recorded` is evidence, and a refused call is
    not evidence; the attempt count is the budget, and a refused call still cost a round.
    """
    messages: list[BaseMessage] = []
    recorded: list[RecordedToolCall] = []
    prior = list(already_called or [])
    numbered = cite_offset
    done = {_call_key(str(call.get("name")), dict(call.get("args") or {}))
            for call in prior}
    # Grows as the round executes, for the tools capped at one call: a single model turn
    # can ask for the same tool twice, and the cap is a property of the turn rather than
    # of one iteration of the loop around it.
    spent = set(used_up)
    # Calls already made per tool, this turn's own and the ones `already_called` reported.
    # A dict rather than a counter of the batch: the allowance belongs to the turn, and a
    # single model turn can emit the same tool twice.
    per_tool: dict[str, int] = {}
    for made in prior:
        made_name = str(made.get("name"))
        per_tool[made_name] = per_tool.get(made_name, 0) + 1
    for index, call in enumerate(calls):
        name = str(call["name"])
        # What the MODEL wrote, kept beside the complete arguments because the guard has
        # to see it: an admission the model invented must be refused, not overwritten.
        asked_for = dict(call.get("args") or {})
        # Projected onto what the model was SHOWN before the session completes it. The
        # schema hides the admission and this is the other half of that: `asked_for` stays
        # untouched, because the cross-patient check below reads what the model WROTE and
        # an attempt to name another admission must be refused rather than corrected.
        #
        # Gated on the session's own admission, because projection is only safe where
        # something will put the admission back. The command path builds its calls from
        # `routing.tool_arguments` and passes no session admission here — its arguments
        # already carry one, and a filter with nothing to re-add would strip the admission
        # out of every command's tool call (measured immediately: `/ehr` dispatched
        # `rag_search_sections({})` and the tool refused it).
        #
        # Named `argument_filter` rather than `project`: `project` is this function's own
        # parameter, the projector that shapes a RESULT for the model, and a local of the
        # same name silently replaced it — every tool result was then projected onto an
        # argument schema and reached the record as an empty payload.
        argument_filter = getattr(toolbox, "model_arguments", None)
        offered = (argument_filter(name, asked_for)
                   if argument_filter is not None and hadm_id is not None
                   else asked_for)
        arguments = _with_session_admission(name, offered, hadm_id)
        # Keyed on the complete arguments, because `prior` holds recorded calls and a
        # recorded call carries the admission that was dispatched. Keying on what the
        # model wrote would make the same call look new every round.
        key = _call_key(name, arguments)
        # Attempts, not executions: a refused call cost the model a round either way, and a
        # budget that only counted dispatches let a model re-ask a capped tool for ever.
        spent_count = max(len(prior), attempted) + index
        # Whether this call was turned away rather than run. A refusal must not consume a
        # capped tool's allowance: the model gets one honest call, not that call minus the
        # attempts nothing was ever dispatched for.
        refused = True
        if call["name"] in spent:
            payload: dict[str, Any] = {
                "error": "tool_already_spent",
                "message": (
                    f"{call['name']} has already been used for this question and will "
                    "not run again. Its result is above: answer from it, and say what it "
                    "does and does not cover."
                ),
            }
        elif spent_count >= MAX_TOOL_CALLS_PER_TURN:
            # The message says what to DO, not only what was refused. A refusal a model
            # cannot act on is read as a transient failure and retried, which is the loop
            # this line exists to end.
            payload = {
                "error": "tool_call_limit",
                "message": (
                    f"This turn has already made {spent_count} tool calls and its budget is "
                    f"{MAX_TOOL_CALLS_PER_TURN}. Answer from the results you have, or say "
                    "plainly what you could not determine."
                ),
            }
        elif per_tool.get(name, 0) >= PER_TOOL_CALL_LIMIT:
            # The total budget could still be unspent here, which is why this is its own
            # refusal: the turn has room and this tool does not, and telling the model it
            # is out of budget would send it looking for a different tool to loop.
            payload = {
                "error": "tool_call_limit_per_tool",
                "message": (
                    f"{name} has already been called {per_tool.get(name, 0)} times for "
                    f"this question and its limit is {PER_TOOL_CALL_LIMIT}. Its results "
                    "are above: answer from them, and if they do not cover the question, "
                    "say plainly what you could not determine rather than searching "
                    "again in different words."
                ),
            }
        elif key in done:
            # A repeat is a loop, not diligence. The result is already above this call in
            # the transcript, so returning it again gives the model nothing new to read.
            payload = {
                "error": "repeated_tool_call",
                "message": (
                    f"{call['name']} has already been called in this turn with these "
                    "exact arguments, and its result is above. Use that result, or call "
                    "with different arguments — do not call it again unchanged."
                ),
            }
        else:
            payload = guard(call["name"], asked_for) if guard is not None else None
            if payload is None:
                refused = False
                # Emitted HERE, one line above the call it describes: the stage is
                # derived from an execution that is about to happen, not from a
                # script of what should happen. A refused call (the branches
                # above) announces nothing, because nothing was called.
                _emit(on_event, stages.tool_event(call["name"]))
                # Spanned ONLY when the caller has an outer span open — the command
                # path, which opens `command.<wire>` around the whole turn. In the
                # reasoning loop the trace belongs to LangChain's callback handler and
                # a span opened here does not inherit it: Langfuse gave each tool its
                # own root and the list filled with `tool.rag_search` traces sharing
                # one timestamp. Observed live 2026-10-17.
                if traced:
                    # The span carries the arguments as its input and the payload as
                    # its output. A tool span with neither is a row saying a tool ran,
                    # which is the one thing the reader already knew from the name.
                    with observability.observation(
                        f"tool.{call['name']}", input=arguments,
                    ) as tool_span:
                        payload = await toolbox.call(call["name"], arguments)
                        observability.record_span_output(
                            tool_span, payload,
                            is_error=bool(
                                isinstance(payload, dict) and payload.get("error")
                            ),
                        )
                else:
                    payload = await toolbox.call(call["name"], arguments)
                # What came back, in the call's own terms. Emitted here because this
                # is where the payload exists, and only for a call that actually
                # ran: a refused call announces nothing, result included (rule 3).
                _emit(on_event, stages.result_event(call["name"], payload))
        done.add(key)
        if not refused:
            per_tool[name] = per_tool.get(name, 0) + 1
            if name in ONCE_PER_TURN and routing.spends_turn_allowance(
                    name, arguments):
                spent.add(name)
        # `order` rewrites the payload BEFORE the model is shown it and before it is
        # recorded, so one order governs both. Doing it afterwards reordered the
        # citation layer while the formatter kept reading retrieval order, and the
        # markers stayed crossed in the other direction (TF-7, measured 2026-10-23).
        if order is not None and isinstance(payload, dict):
            payload = order(call["name"], payload)
        shown = project(call["name"], payload) if project is not None else payload
        # Only a call that actually DISPATCHED is evidence. A pre-dispatch refusal is a
        # message to the model and nothing else: the tool never ran, so there is no result
        # to record, no position to assign, no evidence for a guardrail to weigh and
        # nothing for a later turn to replay.
        #
        # Recording it was not harmless. `a2ui.turn_scope` badges a turn `patient_chart`
        # from `args.hadm_id` alone, so a REFUSED cross-patient attempt badged the turn as
        # a chart read; `context_grounded` read the same call on replay; the PMID
        # allowance counted a refused literature search; and the evaluator's
        # `turn_tool_calls` check scored the attempt as a disclosure that never happened
        # (`dev_t7_03`; audit §2.2). The budget does not regress either — it counts
        # attempts now, not recordings (`attempted`, below).
        executed = not refused
        if executed:
            recorded.append(validate_recorded_tool_call({
                "name": call["name"], "args": arguments, "response": payload,
            }))
        # Ingested BEFORE the copy the model reads is built, because the record is what
        # decides the numbers — and ingested as `shown`, because the record is rendered
        # back into the model's window and must not hold anything the projection
        # withheld. `recorded` keeps the tool's own payload; only the record the model
        # reads is projected.
        slot_ids: tuple[int, ...] = ()
        if evidence_state is not None and executed:
            slot_ids = evidence_state.ingest_tool_output(
                str(call["name"]), arguments, _record_text(shown),
            ).slot_ids
        # The record's positions and the renderer walk the same array now, so the length
        # check is a statement of that rather than a hope: a projection that added or
        # dropped a passage would fall back to the running number instead of stamping a
        # position onto the wrong passage.
        use_ref_ids = (
            slot_ids
            if slot_ids and len(slot_ids) == citations.passage_total(shown)
            else None
        )
        body = _render_tool_result(shown, offset=numbered, ref_ids=use_ref_ids)
        numbered = (
            evidence_state.position_count()
            if evidence_state is not None
            else numbered + _passage_total(shown)
        )
        messages.append(
            ToolMessage(
                content=(
                    f'<tool_result name="{call["name"]}">\n{body}\n</tool_result>'
                ),
                tool_call_id=call.get("id", ""),
                name=call["name"],
            )
        )
    return messages, recorded

