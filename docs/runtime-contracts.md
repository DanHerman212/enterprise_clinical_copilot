# Runtime Contracts

The online system has four boundaries. Each boundary has one owner and a structured
payload. Detailed implementation remains in the owning package.

## Django to agent

`POST /ask` receives:

```json
{"question": "Assess risk for admission 90000009."}
```

The agent validates a non-empty question and a 2,000-character maximum. Success returns
`question`, `answer`, `guardrail_flags`, `tool_calls`, `a2ui`, `citation_map`,
`intent_sections`, `model`, and `mcp_transport`. Failures use stable codes for invalid
input, timeout, unavailable answer, and internal failure. See `services/agent/contracts.py`
and `services/agent/http.py`.

The presentation contract — renumbered `answer`, `citation_map`, `intent_sections`, and the
composed `a2ui` canvas — is produced by the agent (`services/agent/a2ui.py` +
`services/agent/citations.py`), because that is the layer where the answer, the guardrails,
and the retrieved evidence meet. Django validates the envelope and forwards it unchanged,
adding only the web-specific `remaining` quota. The browser renders; it does not reinterpret
citations.

## Agent internal state

LangGraph records each executed tool call as:

```json
{"name": "predict_readmission", "args": {"hadm_id": 90000009}, "response": {}}
```

Arguments remain internal to the agent trace. The HTTP response sent to Django trims each
record to `name` and `response`.

## MCP tools

The MCP service exposes three tools:

- `predict_readmission(hadm_id)`: structured features to probability, decision, and factors.
- `rag_search(hadm_id, query, top_k)`: admission-filtered vector retrieval with scored passages.
- `rag_search_sections(hadm_id)`: deterministic section passages for summary requests.

Tool failures are returned as data with `error` and `message`; they are not silently
converted into plausible clinical content. Contracts are defined in
`services/mcp/contracts.py`.

## Safety invariants

- Django checks demo admission membership before downstream spend.
- MCP retrieval applies the admission restriction at the index and re-checks note ownership.
- Tool timeout is shorter than agent timeout, which is shorter than the Django proxy timeout.
- Health checks do not call billable dependencies.
- Model probability comes from structured features and is not changed by retrieval text.
