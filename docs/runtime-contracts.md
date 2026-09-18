# Runtime Contracts

The online system has four boundaries. Each boundary has one owner and a structured
payload. Detailed implementation remains in the owning package.

## Django to agent

`POST /ask` receives intent, not prompt text:

```json
{"hadm_id": 90000009, "chip": "risk"}
```

Exactly one of `question` or `chip` is required, `hadm_id` is optional, and `turns` is the one
accepted way to carry earlier turns of a conversation (at most six, each a bounded question and
answer, optionally with the tool calls it made). The agent composes the wording, validates the
question against a 2,000-character maximum, and refuses any other field by name — a
conversation-named field is refused as a product decision made visible rather than dropped in
silence. Success returns `question`, `answer`, `guardrail_flags`, `tool_calls`, `a2ui`,
`sources`, `model`, `code_revision`, and `mcp_transport`. Failures use stable codes for invalid
input, timeout, unavailable answer, and internal failure. See `services/agent/contracts.py`
and `services/agent/http.py`.

The presentation contract — the renumbered `answer`, the resolved `sources`, and the
composed `a2ui` canvas — is produced by the agent (`services/agent/a2ui.py` +
`services/agent/citations.py`), because that is the layer where the answer, the guardrails,
and the retrieved evidence meet. Django validates the envelope and forwards it unchanged,
adding only the web-specific `remaining` quota and, when a turn produced one, the
`conversation_id` it stored the turn under. The browser renders; it does not reinterpret
citations.

## Agent internal state

LangGraph records each executed tool call as:

```json
{"name": "predict_readmission", "args": {"hadm_id": 90000009}, "response": {}}
```

The HTTP response sends each call as `name`, `args`, `response`, and `derivable` — whether the
result can be obtained again by calling the same tool with the same arguments. Nothing new
crosses the boundary: the arguments are the caller's own question and admission, and the
passages the call returned already do. A caller that stores a turn needs both fields, because a
replayed call without its arguments misstates what was asked, and a result that can be
re-derived must not be kept: retrieval returns discharge-note text, so the site stores the
prediction payloads and leaves the retrieval results to be resolved again
(`services/agent/contracts.py`, `RE_DERIVABLE_TOOLS`).

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
