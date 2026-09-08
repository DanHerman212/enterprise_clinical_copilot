# Phase 0 Baseline: Runtime Contracts

**Date:** 2026-09-06

**Scope:** Current behavior of the online request path across the `enterprise_clinical_copilot` repository and the companion `danielmherman` Django repository.

**Status:** Read-only baseline. No runtime code was changed to produce this document.

## System boundary

The browser-facing request is handled by Django. Django is the public boundary and owns authentication, demo-cohort authorization, quota accounting, fixture/live mode selection, citation preparation, and A2UI composition.

The private agent service owns request validation, the LangGraph run, MCP connectivity, post-hoc answer guardrails, and the agent-facing response. The agent service does not know about Django users or demo quotas.

The current online path is:

```text
Browser
  -> Django POST /demo/a2ui/ask/
    -> fixture response OR private agent POST /ask
      -> LangGraph agent
        -> MCP client
          -> MCP server
            -> prediction and/or retrieval tool
      <- agent response
    <- Django adds citation metadata, A2UI canvas, and remaining quota
  <- browser renders the response
```

## Boundary 1: Browser to Django

**Route:** `POST /demo/a2ui/ask/`

**Owner:** `danielmherman/demo/views.py::a2ui_ask`

**Authentication:** Django `@login_required`.

### Request shape

The frontend sends JSON with one of these forms:

```json
{"hadm_id": 90000009, "chip": "risk"}
```

```json
{"hadm_id": 90000009, "question": "Why was this patient flagged?"}
```

```json
{"question": "..."}
```

The frontend sends the selected admission with chip and selected-patient free-text requests. The CSRF token is sent in the `X-CSRFToken` header.

**Frontend owner:** `danielmherman/static/js/demo_flow.js::post`, `askChip`, and `askFreeText`.

### Django validation and authorization

`_question_for` in `demo/views.py` performs the following checks before live-mode quota consumption:

1. Parse `hadm_id` as an integer when present.
2. Reject non-positive admission ids.
3. Verify that the admission exists in the Django `DemoPatient` allowlist.
4. Resolve a chip key through `demo/fixtures.py::CHIPS`.
5. Reject unknown chips.
6. Require non-empty free text when no chip is supplied.
7. Enforce `MAX_QUESTION_CHARS = 2000`.
8. Append `For admission {hadm_id}.` to selected-patient requests.

The `DemoPatient` allowlist is the application-level authorization boundary. It is checked before the request can claim quota or invoke a downstream service.

### Fixture/live branch

The branch is controlled by `DEMO_FIXTURE_MODE` in Django settings.

- **Fixture mode:** `demo/fixtures.py::fixture_ask` returns captured payloads. It is development scaffolding and is rejected when `IS_PRODUCTION` is true.
- **Live mode:** Django claims quota, calls `demo/agent_client.py::ask`, handles failures/refunds, and then composes the response from the returned tool calls.

This means the same browser endpoint has two execution paths with different downstream behavior. The response shape is intended to remain compatible between them.

## Boundary 2: Django to private agent

**Route:** `POST /ask`

**Client:** `danielmherman/demo/agent_client.py::ask`

**Server:** `enterprise_clinical_copilot/services/agent/http.py::ask_route`

### Agent request

Django sends:

```json
{"question": "Assess the 30-day readmission risk for admission 90000009."}
```

The request does not include the Django user, quota, chip key, or browser state. Those concerns terminate at Django.

The client:

- requires `DEMO_AGENT_URL`;
- limits concurrent calls with `DEMO_AGENT_MAX_CONCURRENCY`;
- mints an identity token whose audience is the service URL;
- sends the request to `{DEMO_AGENT_URL}/ask`;
- applies `DEMO_AGENT_TIMEOUT` (default 120 seconds);
- rejects malformed response bodies;
- classifies failures as zero-spend or potentially-spent through `AgentError.spent`.

The agent service requires an `Authorization` header on Cloud Run unless explicitly configured otherwise. It independently enforces a 2,000-character question limit and applies an internal 110-second request deadline by default.

### Agent success response

The current `services/agent/http.py` response is:

```json
{
  "question": "Assess the 30-day readmission risk for admission 90000009.",
  "answer": "...",
  "guardrail_flags": [],
  "tool_calls": [
    {
      "name": "predict_readmission",
      "response": {"...": "..."}
    }
  ],
  "a2ui": {"...": "..."},
  "citation_map": {"1": 1},
  "intent_sections": ["discharge_medications"],
  "model": "gemini-2.5-flash",
  "mcp_transport": "http"
}
```

The agent server deliberately removes tool-call arguments before returning to Django. The response retains only each tool name and its response payload. This is tested in `tests/agent/test_error_disclosure.py`.

`guardrail_flags`, `a2ui`, `citation_map`, and `intent_sections` are agent-generated response fields. The citation renumbering, citation map, section-intent resolution, and A2UI canvas composition all happen in the agent (`services/agent/a2ui.py` + `services/agent/citations.py`) because they are evidence semantics — they belong where the answer, the guardrails, and the retrieved evidence meet. Django passes the contract through unchanged and adds only `remaining`. Fixture mode emits the same contract by reusing the agent's own composer through a thin cross-repo adapter in `demo/fixtures.py`.

### Agent failure responses

| Condition | HTTP status | Body contract |
|---|---:|---|
| Missing authorization header on Cloud Run | 401 | `error: unauthenticated`, safe message |
| Invalid JSON | 400 | `error: invalid_json` |
| Missing or blank question | 400 | `error: invalid_request`, safe message |
| Question exceeds 2,000 characters | 413 | `error: question_too_long`, safe message |
| Agent deadline exceeded | 504 | `error: timeout`, safe message |
| Internal agent/MCP failure | 502 | `error: agent_failed`, safe message, 12-character correlation id |
| Empty final assistant text | 502 | `error: answer_unavailable`, safe message |

Internal exception detail is logged on the agent service and is not returned to the caller. The `/health` route is intentionally shallow and does not call Vertex, MCP, or BigQuery.

## Boundary 3: Agent to MCP

The agent opens a fresh MCP toolbox session per request. `services/agent/mcp_client.py::MCPToolbox` loads the server's advertised tools, converts MCP input schemas into Gemini-compatible function declarations, invokes tools, and converts tool failures into structured payloads instead of collapsing the graph.

The transport is configuration-driven:

- `stdio` for local execution;
- authenticated streamable HTTP for the deployed agent-to-MCP call.

The agent graph is explicit:

```text
START -> agent -> tools -> agent -> END
```

A model turn may end the run or emit tool calls. The tool node executes calls, records their responses, and returns control to the agent for final narration. `MAX_TOOL_CALLS_PER_TURN` and `RECURSION_LIMIT` bound execution. Langfuse tracing is optional and enabled only when its required environment variables are present.

## Boundary 4: Django response to browser

After live-mode agent success, `a2ui_ask`:

1. Detects error payloads inside `tool_calls` and refunds quota with HTTP 502.
2. Determines question intent for citation resolution.
3. Renumbers citation markers in the answer.
4. Stores a citation remapping for the browser.
5. Extracts prediction and retrieval tool responses.
6. Composes the A2UI canvas through `demo/a2ui_canvas.py`.
7. Adds `intent_sections` and the remaining quota.
8. Returns the combined JSON response.

The browser stores the response as an episode turn. It retains tool calls, passages, citation metadata, and the A2UI envelope so a prior turn can be re-rendered when a citation is selected.

## Current source-of-truth map

| Concern | Current source | Consumers |
|---|---|---|
| Django request validation and chip wording | `danielmherman/demo/views.py`, `demo/fixtures.py` | Django view and fixture path |
| Agent HTTP contract | `enterprise_clinical_copilot/services/agent/http.py` | Django agent client, agent tests |
| Agent orchestration state | `services/agent/graph.py` | Agent server, agent tests, Langfuse |
| MCP tool registration | `services/mcp/server.py` | MCP clients |
| Prediction response | `services/mcp/tools/predictionion.py` and serving code | Agent, Django canvas, tests |
| Retrieval response and isolation | `services/mcp/tools/retrieval.py` | Agent, Django canvas, tests |
| Browser response state | `danielmherman/static/js/demo_flow.js` | A2UI renderer and thread |
| A2UI composition | `danielmherman/demo/a2ui_canvas.py` and agent response path | Browser renderer |
| Agent boundary tests | `tests/agent/test_error_disclosure.py`, agent tests | Agent service |
| Django boundary tests | `danielmherman/demo/tests.py` | Django view, quota, fixture/live behavior |

## Phase 0 observations

1. The agent service and Django response are already separate conceptual contracts, even though neither is represented by a typed schema module.
2. Django validates the admission against its own `DemoPatient` table; the MCP tools validate admission ids again for tool correctness and retrieval isolation. These checks should not be merged without preserving their different responsibilities.
3. The agent and Django use different timeout values: 110 seconds internally and 120 seconds at the proxy. This is intentional and should become an explicit configuration invariant.
4. Fixture mode is useful for UI development but creates a second path through the same Django endpoint. Its compatibility with live mode is covered partly by tests and should be made explicit in Phase 1.
5. Citation handling is the most distributed response concern: the agent emits markers, Django renumbers and maps them, and browser code resolves and renders them.
6. The current tests provide good behavior coverage for authentication, quota/refund semantics, fixture/live branching, error disclosure, and tool-call trimming. The missing artifact is a single contract description tying those tests to the service boundary.

## Phase 0 exit condition

An engineer should be able to begin Phase 1 using:

- this document for the cross-repository request path;
- `demo/views.py` for Django validation and response composition;
- `demo/agent_client.py` for the proxy behavior;
- `services/agent/http.py` for the agent HTTP contract;
- `services/agent/graph.py` for orchestration;
- `services/agent/mcp_client.py` and `services/mcp/server.py` for the tool boundary;
- `test_error_disclosure.py` and `demo/tests.py` for current boundary behavior.

The next step is to turn the Django-agent contract described here into explicit validation and tests without changing the existing wire keys or deployment topology.

## Baseline validation

The focused tests were run without cloud calls:

- ECC agent boundary and spend controls: `13 passed` from `test_error_disclosure.py` and `test_spend_caps.py`.
- Django authentication, console/live ask, quota, refund, and response behavior: `15 passed` in `4.382s` from `DemoAuthTests` and `A2uiAskLiveTests`.

The Django test output includes expected error logs from tests that exercise agent failure and downstream `502` handling; the suite completed with status `OK`.
