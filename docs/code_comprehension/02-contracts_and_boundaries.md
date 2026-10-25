# Phase 2: Wire Contracts & Trust Boundaries Remediation

**Status:** `COMPLETE`  
**Target Layers:** Layer 1 (`01-client.md`), Layer 2 (`02-api_gateway.md`), Layer 3 (`03-agent_orchestration.md`), & Layer 5 (`05-tool_execution.md`)  
**Repositories in Scope:** `enterprise_clinical_copilot` & `danielmherman`  
**Objective:** Audit and remediate wire-contract drift, parallel/superseded presentation payloads, and request/refusal boundary alignment between the Django BFF (`danielmherman`) and the Agent Service (`enterprise_clinical_copilot`).

---

## 0. Governance Protocol for This Spoke

This phase strictly follows the Two-Step Agent Protocol defined in `00-master-alignment-plan.md`:

* **Step A — The Audit Pass (Read-Only — CURRENT STEP):**
  * Inspect the target files listed in Section 1 against the Reference Architecture (`docs/architecture/01-client.md`, `02-api_gateway.md`, `03-agent_orchestration.md`, and `05-tool_execution.md`).
  * Record exact file paths, line numbers, findings, and the proposed surgical changes in **Section 2 (Step A — Audit Findings)**.
  * **STOP.** Do NOT modify any application or test code during Step A. Await explicit Architect approval.
* **Step B — The Execution Pass (Write + Verify — LOCKED UNTIL STEP A APPROVED):**
  * After Architect approval of Step A, execute the approved changes, run all verification gates, and record the results in **Section 3 (Step B — Execution Record)**.

---

## 1. Step A Audit Scope & Questions to Answer

Inspect both repositories and document exact file and line-number findings for the following four boundary areas:

### 1.1 Dual Presentation Contract Audit (`services/agent/payload.py` vs. `services/agent/a2ui.py`)
* **Context:** `services/agent/payload.py` builds `ui_components`, `narrative_text`, and top-level `citations`, which are injected into the `/ask` response in `services/agent/turn.py` and `services/agent/http.py` and validated in `services/agent/contracts.py`. Meanwhile, `services/agent/a2ui.py` composes the `a2ui` envelope, `references`, `scope`, and `verified`.
* **Audit Tasks:**
  1. Check every reference to `ui_components`, `narrative_text`, and top-level `citations` across both `enterprise_clinical_copilot` and `danielmherman` (including `demo/views.py`, `demo/conversations.py`, `demo/fixtures.py`, `static/js/`, tests, and evaluation scripts).
  2. Determine whether `services/agent/payload.py` and its wire fields (`ui_components`, `narrative_text`, `citations`) are completely superseded by `a2ui.py` in production, or if any active runtime/eval consumer still reads them.
  3. List the exact files and line numbers that must be updated or removed to establish a single canonical presentation contract.

### 1.2 Residual `SourceCard` & Top-Level `sources` Audit (`a2ui.py`, `conversations.py`, `fixtures.py`)
* **Context:** Phase 1 removed the retired `SourceCard` widget from `static/js/a2ui_clinical_components.js`[cite: 4].
* **Audit Tasks:**
  1. Inspect `enterprise_clinical_copilot/services/agent/a2ui.py` and `danielmherman/demo/conversations.py` (`record_answer`, `turn_payloads`) to see how `sources` vs. `references` are generated, stored on `Turn` rows, and rehydrated[cite: 3].
  2. Inspect `danielmherman/demo/verify_live_citations.py` (flagged in Phase 1 Open Item 1) and `demo/fixtures.py` for any stale `SourceCard` or legacy schema assumptions[cite: 3, 4].
  3. Propose the exact cleanup so stored turns, fixtures, and live responses share one unambiguous citation/reference contract.

### 1.3 BFF-to-Agent Request Schema Parity (`demo/views.py` vs. `services/agent/contracts.py`)
* **Context:** `demo/views.py` validates browser requests against `APPROVED_REQUEST_FIELDS` and `CONVERSATION_FIELDS`, transforms the payload via `_intent_for()`, `_with_replay()`, and `_streaming_ask()`, and forwards the request to the agent via `demo/agent_client.py`[cite: 3].
* **Audit Tasks:**
  1. Compare the outbound dictionary keys built by `demo/views.py` (`_intent_for`, `_with_replay`, `_streaming_ask`) and sent by `demo/agent_client.py` against `APPROVED_REQUEST_FIELDS` and `validate_ask_request()` in `enterprise_clinical_copilot/services/agent/contracts.py`[cite: 3].
  2. Compare the browser-facing refusal rules in `demo/views.py` against `services/agent/contracts.py` and `services/agent/routing.py`[cite: 3].
  3. Identify any schema drift, missing field validations, or duplicated constants that lack cross-repo contract tests.

### 1.4 Error & Policy Refusal Vocabulary Parity (`CORE_TOOLS` & `POLICY_REFUSALS`)
* **Context:** `demo/views.py` defines `CORE_TOOLS = frozenset({"predict_readmission", "rag_search", "rag_search_sections"})` and `POLICY_REFUSALS = frozenset({"tool_call_limit", "repeated_tool_call", "patient_scope_refused", "patient_identifier_refused"})` to decide whether a turn should be refunded and turned into a 502 outage (`UNAVAILABLE`) or passed through as a valid answer[cite: 3].
* **Audit Tasks:**
  1. Cross-check `CORE_TOOLS` and `POLICY_REFUSALS` in `demo/views.py` against the exact tool names and refusal codes emitted by `enterprise_clinical_copilot/services/agent/turn.py` and `services/mcp_server/`[cite: 3].
  2. Verify whether any error codes, refusal shapes, or internal details can leak across the BFF boundary (`02-api_gateway.md` / S1-03)[cite: 3].
  3. Propose exact contract tests to lock this vocabulary on both sides.

---

## 2. Step A — Audit Findings & Proposed Surgical Plan

Read-only pass. No application or test code was modified. Line numbers are against the working tree at the time of audit.

### 2.0 Summary

| § | Question | Verdict | Severity |
| :--- | :--- | :--- | :--- |
| 1.1 | Is the Phase-1 "strict payload" a second live contract? | **Confirmed.** Three fields produced on every turn, validated as mandatory, consumed by nobody. | High |
| 1.2 | Do `sources` / `references` / `citations` overlap? | **Confirmed.** Three names carrying two payloads; one published name is unread by the client; fixtures omit the rest. | High |
| 1.3 | Does the BFF request shape match the agent's? | **Traffic is a strict subset — no drift today.** The *declared sets* diverge and nothing pins them. Duplicated literals in two repos. | Medium |
| 1.4 | Does the refusal vocabulary match? | **Codes match exactly.** `CORE_TOOLS` has drifted from the served tool set; refund policy is therefore wrong for two tools. | Medium |

---

### 2.1 Dual presentation contract (`1.1`)

**Produced on every successful turn**

| Field | Producer | Validation |
| :--- | :--- | :--- |
| `narrative_text` | `services/agent/http.py:546` (`presentation["answer"]`) | `services/agent/contracts.py:710`, mandatory unless `_draws_the_answer(payload)` |
| `citations` | `services/agent/http.py:550` (`presentation["sources"]`, same list object) | `services/agent/contracts.py:717`, must be a list of dicts |
| `ui_components` | `services/agent/http.py:551-553` → `payload.build_ui_components` (`services/agent/payload.py:323`), imported at `services/agent/http.py:42` | `services/agent/contracts.py:723-732`, component name must be a string, `props` must be an object |

`contracts.validate_agent_success` (`services/agent/contracts.py:621`) is invoked from `services/agent/http.py:583`, so all three are enforced on the live request path. The source comment at `http.py:541-545` states the intent explicitly: *"Additive on purpose … so the site, the renderer and the new components can each move on their own revision"*, and `http.py:547-549` admits `citations` is `sources` under a second name. The migration window the comment anticipates has closed: the Phase-1 frontend that consumed `ui_components` is deleted.

**Consumers**

| Consumer | References | Notes |
| :--- | :--- | :--- |
| `danielmherman` site (`demo/`, `static/js/`) | **0** | No reference to `ui_components` or `narrative_text` anywhere. |
| `evaluation/platform/task.py:50` | 1 | `response.get("a2ui") or response.get("ui_components")` — fallback only, unreachable while `a2ui` is composed on every turn. |
| `docs/ux_workshop/phase1/run_commands.py:85,97,151` | 3 | Historical harness inside the workshop record; not executed by CI. |

**Tests pinning the fields** (`grep -rc`): `tests/agent/test_agent_contract.py` 10, `tests/agent/test_command_boundary.py` 6, `tests/agent/test_progress_stream.py` 2, `tests/agent/test_freetext_boundary.py` 2, `tests/agent/test_affordance_and_meds_surface.py` 2, `tests/agent/test_golden_path_fixes.py` 1, `tests/agent/test_platform_eval.py` 1, `tests/agent/test_judge_modality_and_adaptive.py` 1. Also `services/agent/payload.py` itself (4) and `services/agent/graph.py` (2 — `ui_components` as an eval-internal concept, not the wire field).

**Finding.** The canonical presentation contract is `a2ui` + `answer` + `sources` + `references`; the three legacy fields are a second, mandatory-on-write contract with zero production readers. They cannot be deleted independently of their validators: `validate_agent_success` raises on a missing `ui_components`, so removing the payload alone turns every turn into a 500.

---

### 2.2 `sources` vs `references` vs stored `citations` (`1.2`)

**Three names, two payloads, three shapes**

| Name | Where produced | Shape |
| :--- | :--- | :--- |
| `sources` | `services/agent/a2ui.py:1233` | resolved note passages: `{cite, section, text, query}` |
| `references` | `services/agent/a2ui.py:1253` | narrative citations: `{index, source, title, label, excerpt}` — `list((narrative or {}).get("citations") or [])`, built by `note_citations` (`a2ui.py:305`) |
| `citations` (inside the canvas) | `services/agent/a2ui.py:302,609` | the `CitedNarrative` component's own field |

**Stored** (`danielmherman/demo/conversations.py`)

- `record_answer:98` — `citations=_stored_citations(result['sources'])`; `_stored_citations:267` deliberately keeps only `{cite, section, query}` and drops `text`.
- `_stored_presentation:134` — persists `{a2ui, scope, references, follow_ups, verified}`; `_presentation_parts:~163` normalises legacy bare-envelope rows.
- `turn_payloads:190` — republishes **both** `citations` and `references` on every replayed turn.

**Read**

- Client: `static/js/copilot_canvas.js:1142` — `payload.references || []`. `payload.citations` is never read anywhere in `demo/` or `static/js/`.
- `demo/test_conversation_memory.py:147` reads `Turn.citations` directly; `:230` uses a `citations` key in a stored-shape fixture.

**Fixture parity** — `demo/fixtures.py:390-408` composes fixture answers through the agent's own `compose_presentation` and returns `{question, answer, tool_calls, a2ui, sources, literature, source, model, fixture_note}`. It omits `references`, `scope`, `verified`, `follow_ups`, `command`, `routed_to`, plus all three legacy fields, while the comment above the return claims exact parity with the live response. A fixture turn therefore cannot exercise the references drawer.

**`demo/verify_live_citations.py`** — repaired in Phase 1. It reads `cited_narrative(a2ui)` for marker structure and `note_source(body)` from `body["sources"]` (`:85`); the `"unavailable"` case asserts absence of sources plus an explicit no-medication sentence, `"empty_passages"` asserts `source is None`. It is consistent with the live contract and needs no change.

**Finding.** `Turn.citations` is a third name for the same facts, in a fourth shape, published on the replay path and read only by a test. `sources` and `references` are genuinely different payloads and should both survive; `citations` is the redundant one.

---

### 2.3 BFF ↔ agent request schema parity (`1.3`)

| Set | Location | Members |
| :--- | :--- | :--- |
| `APPROVED_REQUEST_FIELDS` | `danielmherman/demo/views.py:65-68` | `question, hadm_id, chip, conversation_id, asked, stream_prose, follow_ups, intent` |
| `ACCEPTED_REQUEST_FIELDS` | `services/agent/contracts.py:169-172` | `question, hadm_id, chip, turns, stream_prose, follow_ups, run_tag, intent` |
| BFF `CONVERSATION_FIELDS` | `demo/views.py:73-76` | `history, messages, session_id, context, turns` |
| Agent `CONVERSATION_FIELDS` | `services/agent/contracts.py:190-192` | same plus `conversation_id` |

**Outbound traffic.** `_intent_for` (`views.py:276-363`) builds `{question | chip, hadm_id, intent}`; `_with_replay` (`views.py:225-241`) adds `turns`; `_streaming_ask` (`views.py:992-996`) adds `stream_prose` and `follow_ups`. `agent_client.ask` (`demo/agent_client.py:153`) posts the dict verbatim to `/ask`. Every key emitted is inside the agent's accepted set, so **no request can be rejected today**. `run_tag` is agent/eval-only — no reference in `demo/*.py`. `conversation_id` and `asked` are browser-facing names the BFF consumes before dispatch and never forwards.

**Duplicated literals with no shared source**

- `MAX_QUESTION_CHARS = 2000` in both repos — `demo/views.py:39`, `services/agent/http.py:50`.
- Chip vocabulary — `demo/fixtures.py:108` (`CHIP_NAMES`) vs `services/agent/questions.py` (`CHIP_QUESTIONS` keys).
- Intent vocabulary — `services/agent/contracts.py:185` (`REQUEST_INTENTS = {"risk_driver"}`) matches the only intent the site emits.

**Finding.** The schemas are compatible by construction, not by verification: no test in either repo asserts the relationship. A field added to `_intent_for` and not to `ACCEPTED_REQUEST_FIELDS` produces a live 400 with both test suites green. The declared sets also diverge on `CONVERSATION_FIELDS` (`conversation_id` is absent from the site's copy), which is a documentation defect rather than a runtime one.

---

### 2.4 Refusal & error vocabulary parity (`1.4`)

**Agent refusal codes and their sites**

| Code | Emitted at | In BFF `POLICY_REFUSALS` |
| :--- | :--- | :--- |
| `patient_scope_refused` | `services/agent/graph.py:423` | yes |
| `patient_identifier_refused` | `services/agent/graph.py:438` | yes |
| `tool_call_limit` | `services/agent/graph.py:504` | yes |
| `repeated_tool_call` | `services/agent/graph.py:515` | yes |

`POLICY_REFUSALS` (`demo/views.py:463-472`) contains exactly these four. The other agent codes — `agent_failed`, `invalid_json`, `invalid_tool_response`, `timeout`, `tool_call_failed`, `unauthenticated`, `undeclared_argument_refused`, `unknown_command`, `unknown_tool` — are correctly *not* policy refusals.

**`CORE_TOOLS` drift.** `demo/views.py:452` lists `{predict_readmission, rag_search, rag_search_sections}`. The MCP server registers seven tools at `services/mcp/server.py:95-101`: those three plus `discharge_medications`, `search_literature`, `get_admission_profile`, `get_lab_trends`.

**Consequences, traced through `_tools_errored` (`demo/views.py:494`)**

1. An error from `discharge_medications` or `get_lab_trends` is neither refunded nor surfaced as an outage: the credit is spent, the answer is a guarded abstention, and the UI reports success. The comment above the constant warns that a hand-maintained list drifts; it has drifted.
2. `unknown_tool`, `undeclared_argument_refused`, `invalid_tool_response` and `tool_call_failed` on a *core* tool are treated as outages (refund + 502) even where nothing downstream was called.
3. No agent error code reaches the browser; the BFF maps to its own unavailable message. The agent's `AgentRequestError` message does pass through by design (`agent_client.AgentRefused`) — that is intended guidance, not leakage.

---

### 2.5 Proposed surgical plan (Step B, on approval)

Ordered so that each step leaves both suites green. Nothing here changes behaviour that a clinician sees except where marked **DECISION**.

**A. Collapse to one presentation contract** (`enterprise_clinical_copilot`)

1. `services/agent/http.py:541-553` — delete `narrative_text`, `citations`, `ui_components` and the comment block above them.
2. `services/agent/http.py:42` — drop the `build_ui_components` import.
3. `services/agent/contracts.py:704-732` — delete the three validations and the `_draws_the_answer` narrative exemption (the answer survives in `answer` and inside the `a2ui` narrative). Keep the rest of `validate_agent_success`.
4. Delete `services/agent/payload.py` — orphaned once step 2 lands. Its only non-test caller is `http.py`.
5. Update the tests that construct or assert those fields: `tests/agent/test_agent_contract.py` (`:350-557`), `test_command_boundary.py` (`:43,714-789`), `test_affordance_and_meds_surface.py` (`:79,89`), `test_progress_stream.py` (`:292,307`), `test_freetext_boundary.py` (`:249,371`), `test_golden_path_fixes.py` (`:121`), `test_platform_eval.py`, `test_judge_modality_and_adaptive.py`.
6. `evaluation/platform/task.py:50` — drop the `ui_components` fallback; `a2ui` is unconditional.
7. Leave `docs/ux_workshop/phase1/run_commands.py` alone; it is a dated research artifact, not a consumer.

**B. One citation name on the replay path** (`danielmherman`)

8. `demo/conversations.py:190` — stop republishing `citations` in `turn_payloads`. Keep publishing `references`.
9. Keep `Turn.citations` and `_stored_citations` as stored provenance, and update the comment at `demo/models.py:324-335` to say so. **DECISION:** removing the column requires a migration and destroys audit history; recommend retaining it, unrepublished.
10. `demo/test_conversation_memory.py:147` — retarget the assertion at `presentation['references']` if step 8 lands, or keep it if the column is retained and documented as provenance.
11. `demo/fixtures.py:390-408` — add `references`, `scope`, `verified`, `follow_ups` (and `command`/`routed_to` where the live path emits them), or narrow the comment to state exactly which keys are modelled. **DECISION:** narrowing the comment is honest and cheap; extending the fixtures makes the drawer testable offline. Recommend extending, since the drawer is currently untested against fixtures.

**C. Pin request parity across repos**

12. Add one checked-in wire contract both suites read — e.g. `services/agent/wire_contract.json` listing accepted request fields, chip names, intents and `MAX_QUESTION_CHARS`; the agent asserts it matches `ACCEPTED_REQUEST_FIELDS`/`CHIP_QUESTIONS`/`REQUEST_INTENTS`, and a site test asserts the BFF's outbound key set and `MAX_QUESTION_CHARS` against the same file. **DECISION:** this adds a file and a cross-repo read path; the alternative is two hand-copied literal lists, which is what exists today and has already drifted on `CONVERSATION_FIELDS`.
13. Add `conversation_id` to the agent's `CONVERSATION_FIELDS` comment or remove it from the site's, so the two declared sets agree.

**D. Refusal vocabulary**

14. Extend `CORE_TOOLS` to the tools whose failure must be refunded: `discharge_medications`, `get_lab_trends`. **DECISION:** this is a product ruling on the refund rule, not a code fix — confirm which tools count as "the answer cannot be produced without it". `search_literature` and `get_admission_profile` are degradable and should stay out.
15. Split `_tools_errored` into "outage" (refund + 502) and "pre-dispatch policy refusal" (no refund, guidance), so `unknown_tool` / `undeclared_argument_refused` stop reading as outages.

**Audit-tooling corrections** (recorded, not acted on): this Spoke cites `services/agent/turn.py` (does not exist; the code is `services/agent/http.py`), `services/mcp_server/` (actual: `services/mcp/`), and the layer documents `02-api_gateway.md`, `03-agent_orchestration.md`, `05-tool_execution.md` (actual filenames: `02-edge.md`, `03-orchestrator.md`, `05-tools.md`).

---

## 3. Step B — Execution Record

Architect approval received for §2.5 in full, with four rulings on the `DECISION` items. Step B executed against those rulings. No behaviour visible to a clinician changed except the two refund-rule corrections in §3.5.

### 3.1 Executed Changes — Agent (`enterprise_clinical_copilot`)

| # | File | Change |
| :--- | :--- | :--- |
| 1 | `services/agent/http.py` | Removed `narrative_text`, `citations`, `ui_components` from the success payload, and the `build_ui_components` import. `a2ui`, `answer`, `sources`, `references` are unchanged. |
| 2 | `services/agent/contracts.py` | Removed the three validations and their comment block from `validate_agent_success`. `_draws_the_answer` is retained: the empty-`answer` rule for a surface-drawn turn still depends on it. |
| 3 | `services/agent/payload.py` | **Deleted** (339 lines). Orphaned once the import went; its only remaining callers were tests. It also still described `source_cards`, a component the Phase-1 frontend deletion removed. |
| 4 | `evaluation/platform/task.py` | Dropped the `ui_components` fallback; `a2ui` is composed unconditionally. |
| 5 | `tests/agent/test_agent_contract.py` | Removed the three fields from `_success_payload` and deleted the four tests that pinned them. |
| 6 | `tests/agent/test_command_boundary.py` | Removed the `payload` imports, `EXPECTED_COMPONENTS` and the five-test "payload the answer is delivered as" section. The medication test keeps its `command_payload` assertions, which are the command boundary's own. |
| 7 | `tests/agent/test_affordance_and_meds_surface.py` | Deleted the two `build_ui_components` tests with the module; the passage-resolution test beside them is the live equivalent and is unchanged. |
| 8 | `tests/agent/test_progress_stream.py` | Removed the three names from the required-key set and the two equality assertions. |
| 9 | `tests/agent/test_wire_contract_parity.py` | **New.** Declared request fields exercised one by one against the parser; unknown field refused; chip and intent vocabularies pinned; sibling-repository checks guarded by `skipUnless`. |

### 3.2 Executed Changes — Site (`danielmherman`)

| # | File | Change |
| :--- | :--- | :--- |
| 10 | `demo/conversations.py` | `turn_payloads` no longer republishes `citations`. Docstring updated to name `references` as the drawer's source and to say why the stored identities stay stored. |
| 11 | `demo/models.py` | `Turn.citations` documented as unrepublished provenance. No migration: the column and its contents are unchanged. |
| 12 | `demo/fixtures.py` | Fixture responses now carry `verified`, `scope`, `references`, `command`, `routed_to`, `guardrail_flags`, `follow_ups`, `code_revision`, `langfuse_trace_id` and `mcp_transport` (`'fixture'`). Values come from the agent's own `compose_presentation` wherever it decides them. |
| 13 | `demo/views.py` | `CORE_TOOLS` extended with `discharge_medications` and `get_lab_trends`; `search_literature` and `get_admission_profile` documented as deliberately absent. `POLICY_REFUSALS` gained `undeclared_argument_refused`. `CONVERSATION_FIELDS` comment corrected: `turns` is the one name that does anything, and `conversation_id` is absent because it never leaves this layer. |
| 14 | `demo/test_conversation_memory.py` | Added a test asserting the published payload carries `references`, not `citations`, while the stored column keeps its identities. |
| 15 | `demo/test_wire_contract.py` | **New.** Forwards from `_intent_for`, `_with_replay` and the streaming path checked against `APPROVED_REQUEST_FIELDS`; sibling-repository checks guarded by `skipUnless(sibling.exists())`. |

### 3.3 Rulings Applied

| Ruling | Applied as |
| :--- | :--- |
| §2.5 8–10: stop republishing `citations`, retain the column | Done. Test asserts both halves: `references` published, `Turn.citations` stored. |
| §2.5 11: extend `fixtures.py` | Done, to the full live key set rather than the four named keys — a partial set leaves the same class of difference the extension exists to remove. |
| §2.5 12–13: parity without a cross-repo dependency | Two self-contained contract files, one per repository, each with its sibling checks behind `skipUnless(sibling_path.exists())`. Isolated CI runners skip the cross-repo group and still run the internal one. |
| §2.5 14–15: `CORE_TOOLS` and `POLICY_REFUSALS` | Both exactly as ruled; `tool_call_failed` and `invalid_tool_response` remain refunded outages. |

### 3.4 Deliberate Non-Changes

- `docs/ux_workshop/phase1/run_commands.py` still references the retired fields. It is a dated research artifact inside the workshop record, not a consumer.
- `services/agent/graph.py:239` (`EVIDENCE_TEXT_KEYS`) and `tests/agent/test_freetext_boundary.py` mention `narrative_text` as a *tool response* key. Unrelated to the wire field; untouched.
- `Turn.citations` keeps its column and values. Removing it would need a migration and would destroy the record of what each turn resolved against.

### 3.5 Behaviour Changes Reachable by a Clinician

1. An error from `discharge_medications` or `get_lab_trends` now refunds the credit and answers 502 instead of spending the credit on an abstention.
2. `undeclared_argument_refused` no longer reads as an outage: the agent refused its own malformed call before anything downstream was contacted.
3. A fixture-mode turn renders the same badge, scope label, references drawer and command state as a live turn.

### 3.6 Verification Gates

| Gate | Result |
| :--- | :--- |
| `pytest tests` (`enterprise_clinical_copilot`) | **1349 passed, 1 skipped** (202s) |
| `python manage.py test` (`danielmherman`, `ENVIRONMENT=development`) | **243 tests OK** (54s) |
| `python manage.py check --deploy` (`ENVIRONMENT=production`) | **System check identified no issues (2 silenced)** |

Gate 3 needs the production environment populated: `GOOGLE_CLOUD_PROJECT=trim-icon-498815-a0` plus the five required vars (`CLOUD_SQL_CONNECTION_NAME`, `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `GS_BUCKET_NAME`, `DEMO_AGENT_URL`). Settings read Secret Manager unconditionally in production, so a dummy project name fails before the check runs.

Suite arithmetic: the agent lost 15 tests with the retired contract and gained 15 in `test_wire_contract_parity.py`; the site gained 10 (9 wire-contract, 1 citation-publication).
