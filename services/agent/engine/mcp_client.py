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
from services.agent.presentation.domain import humanize
from services.agent.telemetry import observability, stages
from services.agent.telemetry.stages import emit as _emit
from shared.errors import AgentResponseError
from typing import Any, Callable, TypedDict

logger = logging.getLogger(__name__)

from services.mcp.contracts import ToolContractError, validate_tool_result
from services.mcp.runtime import timeout_chain

_LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")  # "stdio" | "http"

# The one tool whose arguments leave our system: a search string goes to NCBI.
# Named rather than derived, because what makes it special is where its arguments
# GO, and a second tool that talks to a third party should acquire the same
# treatment deliberately rather than by inheriting it.
EXTERNAL_TOOL = "search_literature"


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


@dataclass
class MCPToolbox:
    """The MCP tools of one session, exposed as Gemini function declarations."""

    session: ClientSession
    # Must stay below the site's 120s upstream deadline and the agent server's
    # ASK_TIMEOUT_SECONDS — a tool that outlives its caller is pure spend. The
    # ordering against the model leg is enforced by `timeout_chain`.
    call_timeout_seconds: float = timeout_chain().tool
    _tools: dict[str, Any] = field(default_factory=dict)

    async def load(self) -> "MCPToolbox":
        result = await self.session.list_tools()
        self._tools = {tool.name: tool for tool in result.tools}
        if not self._tools:
            raise RuntimeError("MCP server advertised no tools")
        return self

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def gemini_tool(self) -> types.Tool:
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=(tool.description or "").strip(),
                    parameters=_clean_schema(tool.input_schema),
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
        try:
            result = await self.session.call_tool(
                name, arguments, read_timeout_seconds=self.call_timeout_seconds
            )
        except Exception as exc:
            _LOG.error("MCP tool %s call failed", name, exc_info=exc)
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
    async with httpx2.AsyncClient(headers=headers, timeout=110) as http_client:
        async with streamable_http_client(f"{base}/mcp", http_client=http_client) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield await MCPToolbox(session).load()


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



ONCE_PER_TURN: frozenset[str] = frozenset({
    "rag_search", "rag_search_sections", "get_lab_trends", "discharge_medications",
    # The estimate is a fact about the admission, not about the request, so a second
    # call in the same turn can only return what the first one returned. It is here
    # because `/ehr` already runs it as its supplementary call: a free-text turn that
    # could bind it a second time would spend a call of the five on a score already in
    # the transcript, and re-render the driver bars beside an answer that did not ask
    # for them.
    "predict_readmission",
})



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



def _passage_total(payload: Any) -> int:
    """How many citation numbers a payload consumes.

    One definition, in `citations`, because two layers count and they must agree: this one
    advances the running number the model is shown, and `citations.citable_passages`
    builds the array a marker resolves against.
    """
    return citations.passage_total(payload)



def _render_passage(passage: dict[str, Any]) -> dict[str, Any]:
    """Return the passage as the model will see it, or refuse it visibly."""
    text = passage.get("text")
    if not isinstance(text, str) or len(text) <= MAX_PASSAGE_CHARS:
        return passage
    logger.warning(
        "refusing passage %s: %d characters, over the %d-character ceiling",
        passage.get("id"), len(text), MAX_PASSAGE_CHARS,
    )
    return {**passage, "text": (
        f"[refused: {len(text)} characters, over the "
        f"{MAX_PASSAGE_CHARS}-character ceiling for one passage]"
    )}



def _note_modality(section: Any) -> str:
    """The modality label for one note passage, as the model reads it.

    The two labels are the vocabulary SYSTEM_PROMPT's citation rule is written in, so they
    are built here and nowhere else: a second spelling of either would be a rule the model
    cannot match against what it was shown.
    """
    name = str(section or "").strip() or "unrecorded section"
    return f"CLINICAL NOTE SECTION — {name}"


def _lab_modality(trend: dict[str, Any]) -> str:
    """The modality label for one analyte's series."""
    name = str(trend.get("label") or trend.get("key") or "").strip() or "laboratory value"
    return f"STRUCTURED LAB TREND — {name}"


def _render_tool_result(payload: Any, *, offset: int = 0) -> str:
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
    """
    if isinstance(payload, str):
        return _escape_delimiter(payload)
    rendered = payload
    if isinstance(payload, dict):
        passages = payload.get("passages")
        if isinstance(passages, list):
            # Each passage carries the number its citation marker must use. The model was
            # previously handed an unlabelled array and had to count positions to emit
            # `^[n]`, and on a four-section summary it counted wrong: `^[3]` went to the
            # discharge instructions and `^[4]` to the discharge diagnosis, both of which
            # begin with "discharge" (TF-7, measured 2026-10-23). The number is now given
            # rather than inferred, and the positions the citation layer resolves against
            # are the same numbers.
            #
            # Each also carries `cite_modality`, which says what KIND of evidence the
            # number points at. The model can see the difference between a note passage
            # and a laboratory series in the payload's own shape, but not in the number:
            # `^[4]` and `^[6]` look alike on the page, and on Gate 2 `dev_t4_02` cited a
            # laboratory trend index onto a narrative claim about transfusion and
            # medication management (2026-10-27). Labelling the number is what lets the
            # citation rule in SYSTEM_PROMPT be checked rather than remembered.
            rendered = dict(payload, passages=[
                ({**(_render_passage(passage) if isinstance(passage, dict) else {}),
                  "cite": offset + index,
                  "cite_modality": _note_modality(
                      passage.get("section") if isinstance(passage, dict) else None)})
                 if isinstance(passage, dict) else passage
                for index, passage in enumerate(passages, start=1)
            ])
        else:
            # A laboratory panel is the second kind of citable evidence (Phase E,
            # 2026-10-26). Without a number the trajectory was not something the model
            # could cite, and a question about an analyte therefore had no citable answer
            # in front of it: the loop searched the note for a citable passage and
            # appended a second paragraph drawn from it. One analyte, one number.
            trends = payload.get("trends")
            if isinstance(trends, list):
                rendered = dict(payload, trends=[
                    ({**trend, "cite": offset + index,
                      "cite_modality": _lab_modality(trend)}
                     if isinstance(trend, dict) else trend)
                    for index, trend in enumerate(trends, start=1)
                ])
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
            description=(tool.description or "").strip(),
            args_schema=_clean_schema(tool.input_schema),
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
    """One risk driver, named and directed, with no magnitude to quote.

    Rebuilt field by field rather than copied and edited, because the field that has to
    go is the one a `{**factor}` spread would keep. The magnitude is not withheld for
    want of a use — the widget draws it as a bar length — but because it is not this
    turn's to state.
    """
    if not isinstance(factor, dict):
        return factor
    return {
        "feature": humanize(str(factor.get("feature", ""))),
        **({"direction": factor["direction"]}
           if isinstance(factor.get("direction"), str) else {}),
    }



def _presented_prediction(name: str, payload: Any) -> Any:
    """The prediction result as the interface shows it, for the model to read.

    The model has never seen a feature key, and the keys are not words a clinician
    reads: the widget draws "red cell distribution width (RDW)" where the result says
    `rdw_max`. The mapping already exists (`feature_labels.py`) and the interface uses
    it, so the substitution is made here rather than requested from the model — a
    translation the model performs is a translation the model can invent, and the prose
    and the bars have to name a factor the same way.

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
    reader will see it written, which is a different thing from a rounded float the
    model then has to print a digit of: this projection used to pre-round, and given
    0.33 the model wrote "33.0%" beside a widget drawing 33.2%, which is why the
    rounding was removed in the first place. Given "33.2%" there is no digit to lose.

    The drivers keep their names and their direction and lose their magnitudes. The
    narrative says which factors led and what they mean; the widget draws how much. A
    raw contribution was quotable too, and a number the guardrail does not recognise as
    the score is stripped and shown as a safety swap — so withholding it also removes a
    way for a good answer to be visibly withdrawn from a reader mid-sentence.

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


def _excursion(low: Any, high: Any, value: Any) -> float | None:
    """How far outside a reference interval a value sits, in range-widths.

    Zero inside the interval, positive outside, and normalised by the interval's own
    width — a plain distance ranks a wide interval above a narrow one however abnormal
    the reading is. `followups._excursion` is the same rule for the chip text; this is
    the copy the model reads, and the two are the same number because they are the same
    definition rather than because they agree.
    """
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (low, high, value)):
        return None
    width = max(float(high) - float(low), 1e-9)
    if value > high:
        return round((float(value) - float(high)) / width, 3)
    if value < low:
        return round((float(low) - float(value)) / width, 3)
    return 0.0


def _presented_labs(payload: Any) -> Any:
    """Lab trends as the model may see them: every timepoint annotated with `excursion`.

    Why this is computed here rather than by the model. `dev_t3_03` asked which value was
    "furthest outside its reference range"; haemoglobin was `low`, and so were the red
    cells and the RDW, so the per-timepoint `flag` could not rank them and the model
    picked one. The prompt forbids re-deriving the ranking from the digits — correctly,
    because a digit the model is holding is a digit it can misread — so the ranking has
    to be computed before the model sees the panel. This is that computation.

    Applied to the model's copy only. The recorded call keeps what the tool returned, so
    the execution record and the caller's payload still say exactly what the series said.
    """
    if not isinstance(payload, dict):
        return payload
    trends = payload.get("trends")
    if not isinstance(trends, list):
        return payload
    presented = dict(payload)
    rows = []
    for trend in trends:
        if not isinstance(trend, dict):
            rows.append(trend)
            continue
        row = dict(trend)
        low, high = trend.get("reference_low"), trend.get("reference_high")
        points = []
        peak = 0.0
        for point in trend.get("series") or []:
            if not isinstance(point, dict):
                points.append(point)
                continue
            excursion = _excursion(low, high, point.get("value"))
            annotated = dict(point)
            if excursion is not None:
                annotated["excursion"] = excursion
                peak = max(peak, excursion)
            points.append(annotated)
        row["series"] = points
        row["excursion_max"] = round(peak, 3)
        rows.append(row)
    presented["trends"] = rows
    return presented


def _presented_for_model(name: str, payload: Any) -> Any:
    """The tool result in the form the MODEL reads. Never the form the caller records.

    Two tools need a translation the model must not perform itself: the estimate, whose
    floats are replaced by the display strings the widget prints, and the lab series,
    whose ranking is computed rather than inferred. Everything else passes through
    untouched.
    """
    if name == "get_lab_trends":
        return _presented_labs(payload)
    return _presented_prediction(name, payload)



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
    for index, call in enumerate(calls):
        arguments = dict(call.get("args") or {})
        key = _call_key(str(call.get("name")), arguments)
        spent_count = len(prior) + index
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
            payload = guard(call["name"], arguments) if guard is not None else None
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
        if not refused and call["name"] in ONCE_PER_TURN:
            spent.add(call["name"])
        # `order` rewrites the payload BEFORE the model is shown it and before it is
        # recorded, so one order governs both. Doing it afterwards reordered the
        # citation layer while the formatter kept reading retrieval order, and the
        # markers stayed crossed in the other direction (TF-7, measured 2026-10-23).
        if order is not None and isinstance(payload, dict):
            payload = order(call["name"], payload)
        recorded.append(validate_recorded_tool_call({
            "name": call["name"], "args": arguments, "response": payload,
        }))
        shown = project(call["name"], payload) if project is not None else payload
        body = _render_tool_result(shown, offset=numbered)
        numbered += _passage_total(shown)
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

