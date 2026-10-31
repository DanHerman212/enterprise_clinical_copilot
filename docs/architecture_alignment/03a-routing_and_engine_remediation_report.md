# Phase 3 — Routing & Engine Layer Remediation

Directive: `03-routing_and_engine_layer_remediation.md`. Scope executed: `routing/`,
`engine/`, `postprocess/`, `ingress/`, `presentation/a2ui.py` (Law 4 registry import
only), `turn.py` (to wire `finish_reason` into `postprocess`). Nothing under `tests/`
was modified, added or deleted.

Result: 4 of the 6 Phase 3 gates are green. Two frozen tests contradict the directive
itself and are reported below rather than edited.

---

## 1. Change by file

| File | +/− | Change |
|---|---|---|
| `routing/routing.py` | +63 | Six canonical registries: `LITERATURE_QUESTION_TOOLS`, `RESULT_KEPT_TOOLS`, `EXTERNAL_TOOL`, `ONCE_PER_TURN_TOOLS`, `ONCE_PER_TRANSCRIPT_TOOLS`, `FULL_FETCH_UNFILTERED_TOOLS`. Each carries the reasoning from the module it was moved out of. |
| `ingress/contracts.py` | 11 | `RE_DERIVABLE_TOOLS` and `RESULT_KEPT_TOOLS` are now reads of the routing sets, re-exported under their old names so the wire contract's callers are unchanged. |
| `engine/graph.py` | 21 | `ONCE_PER_TRANSCRIPT` and `FULL_FETCH_UNFILTERED` bound to routing; `_freetext_allowed` returns `LITERATURE_QUESTION_TOOLS` for a literature turn; `replace_final_text` ×2 replaced by postprocess calls. |
| `engine/mcp_client.py` | 60 | `EXTERNAL_TOOL` and `ONCE_PER_TURN` bound to routing; new `PER_TOOL_CALL_LIMIT = 2`, enforced per tool per turn; per-tool allowance counted only for calls that actually ran. |
| `engine/deterministic.py` | 28 | `OUTPATIENT_EHR_NOTICE` and the withheld/silent branch replaced by `answer.outpatient_ehr_notice()` and `answer.empty_turn_sentence(...)`. The route now supplies facts; postprocess chooses the sentence. |
| `engine/model.py` | 3 | `silent_refusal` calls `answer.silent_turn_sentence(outpatient=...)`. |
| `postprocess/answer.py` | +82 | `apply_model_reply`, `reject_model_reply`, `reply_withheld`, `silent_turn_sentence`, `outpatient_ehr_notice`, `empty_turn_sentence`. |
| `postprocess/citations.py` | +9 | `RETRIEVAL_TOOLS`, re-exported from `routing.PASSAGE_TOOLS` for the presentation layer. |
| `postprocess/guardrail.py` | +35 | `guard_answer(..., finish_reason=None, state=None)`; a withheld reply is resolved before any clinical guard runs and returns `safety_withheld`. |
| `presentation/a2ui.py` | 8 | `rag_payload` reads `citations.RETRIEVAL_TOOLS` instead of restating three tool names. |
| `turn.py` | +6 | `finish_reason(final_message(state))` and `state` passed into `guard_answer`. |

281 insertions, 45 deletions across 11 files. No new module.

---

## 2. What each law now holds

**Law 4 — one tool registry.** `engine/graph.py`, `engine/mcp_client.py`,
`ingress/contracts.py` and `presentation/a2ui.py` each held a literal set of tool names.
All four now read `routing/`. The two gates are green:

- `test_law4_no_duplicate_tool_registry_literals`
- `test_law4_tool_registries_are_declared_once`

**Law 2 — the engine does not select prose.** The three shapes the audit named are
gone: no engine module names a fallback constant, and no engine module calls
`replace_final_text`. The decision moved to `postprocess/answer.py`; the engine holds
one line of invocation per site, which is what keeps
`test_outpatient_ehr_notice.py::test_a_formatter_that_says_nothing_leaves_a_readable_answer`
satisfied without the constant ever being named in `engine/`.

- `test_law2_the_engine_never_selects_a_fallback_sentence` — green

**Contract 1 — a withheld reply is resolved in postprocessing.** `guard_answer` takes
the provider's reason. A withheld reply discards the fragment before any guard runs —
there is nothing left to verify — and returns the withheld summary with
`safety_withheld` in the flags, or the silent-turn sentence when nothing was retrieved.
Both routes reach it now: the engine previously owned the branch, so the free-text
route had the fragment, the reason, and no decision.

- `test_a_withheld_reply_is_resolved_by_postprocess_not_the_engine` — green
- `test_the_withheld_sentence_is_served_with_its_reason_in_the_flags` — green

**Contract 2 — a literature question is scoped, and cannot loop.**
`_freetext_allowed` narrows to literature plus record retrieval for a literature turn,
and `PER_TOOL_CALL_LIMIT = 2` stops the fourth refinement of one search. The cap counts
only calls that ran, so a refusal does not consume an allowance.

- `test_a_literature_question_binds_no_lab_or_medication_tool` — green
- `test_the_literature_tool_has_its_own_call_cap` — green

---

## 3. Gates

| Gate | Result |
|---|---|
| `git diff --name-only tests/` | empty |
| `pytest tests/agent/test_layer_separation.py` | 3 failed, 24 passed |
| `pytest tests/agent/test_architecture_remediation_contracts.py` | 3 failed, 6 passed |
| `pytest tests -m "not live"` | 8 failed, 1632 passed, 1 skipped, 43 deselected |

Against the Phase 2 baseline (12 failed of 1639) six of the twelve targets are now
green. The six intended residuals are Contract 3 ×2 (canonical `EvidenceLedger`,
presentation must not renumber), Contract 4 ×1 (literature turn must not draw a lab
card) and `test_law1_*` ×3 (marker regexes in `postprocess/citations.py` and
`evaluation/`, which the directive placed outside this phase).

---

## 4. Two frozen tests contradict the directive

Both are pre-existing tests that assert the behaviour the audit found defective. They
are not weakened, skipped or edited; they are reported because the decision about which
side to change belongs to the owner of `tests/`.

**`tests/agent/test_freetext_boundary.py:165`** asserts that a literature turn binds
`FREETEXT_TOOLS | {"search_literature"}`, which includes `get_lab_trends` and
`discharge_medications`. Contract 2 asserts the same call returns
`(LITERATURE_TOOLS | PASSAGE_TOOLS) - {"discharge_medications"}`. The two sets cannot
both be right; Contract 2 is the one written to fail against un-remediated code, and its
docstring names the measurement (a five-call chart-plus-literature turn) that makes the
older expectation the defect. Lines 165–172 hold the same conflict for a driver-shaped
literature turn.

**`tests/agent/test_progress_stream.py:223`** asserts that five `search_literature`
calls in one turn produce ten stage events, i.e. that all five executed. Its docstring
states the premise directly — "a literature search takes the clinician's own words and
has no such cap, which makes it the one tool a turn can legitimately spend its budget
on" — which is the assumption Contract 2 overturns. With `PER_TOOL_CALL_LIMIT = 2`,
three of the five are refused before dispatch, so four events are emitted.

Both need the owner's decision. The minimal amendments, if approved: restate the
literature expectation in `test_freetext_boundary.py` in terms of
`routing.LITERATURE_QUESTION_TOOLS`, and re-point `test_a_refused_call_announces_nothing`
at the turn budget using calls the per-tool cap does not reach — the property it guards
(a refused call announces nothing) is unaffected.

---

## 5. Not done, by design

- Contract 3 (`EvidenceLedger`, presentation renumbering) and Contract 4 (canvas
  scoping) — Phase 5 and Phase 6.
- `test_law1_no_module_compiles_an_inline_marker_regex` needs `evaluation/` changes,
  which the directive ruled out of this phase.
- No commit. No push.
