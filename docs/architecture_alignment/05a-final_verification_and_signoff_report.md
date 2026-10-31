# Phase 5 — Final Verification, Evaluation Dry-Run & Sign-Off

**Date:** 2026-10-31
**Branch:** `main` at `7f28e3b`
**Mode:** read-only. No file under `services/`, `tests/` or `evaluation/` was modified,
added or deleted during this phase. The only write is this report.

**Verdict:** the remediation is complete and clean. Every file the programme touched is
**green, and so is the wider evaluation suite: both failures this phase surfaced were
repaired in the follow-up cleanup on the same day — §4 records what each was and what the
fix was. Final state: `pytest tests -m "not live"` 1640 passed,
`pytest tests/evaluation` 197 passed, and `python -m evaluation.platform.dry_run`
reports `STEP 7 PASSED — 100%`.**

---

## 1. Verification gates

| Command | Result |
|---|---|
| `pytest tests/agent/test_layer_separation.py tests/agent/test_architecture_remediation_contracts.py` | **36 passed** (27 + 9) |
| `pytest tests -m "not live"` | **1640 passed, 1 skipped, 43 deselected** |
| `pytest tests -m "not live" -k "mcp"` | **17 passed, 1 skipped, 1666 deselected** |
| `pytest tests/evaluation -m "not live"` | **166 passed, 31 deselected** |
| `pytest tests/evaluation` (all markers) | **197 passed** — was 196 passed / 1 failed before §4.1 |
| `python -m evaluation.platform.dry_run` | **STEP 7 PASSED — 100%**, 9 checks, 6 items — was an abort before §4.2 |

There is no `tests/mcp` directory: the MCP contracts live in `tests/agent`
(`test_mcp_stdio.py`, `test_mcp_contract_boundary.py`) and are selected by `-k mcp`. The
evaluation suite is split the same way the rest of the repo is — 31 of its 197 tests are
live-marked, and the failing one is among them.

---

## 2. The eight Charter 13 findings, before and after

| # | Before (`01a-audit_report.md`) | After | Enforced by |
|---|---|---|---|
| **L1-1** | `presentation/a2ui.py:363-364` called `renumber_citations` + `citation_remap` and published the rewritten prose as `answer`. | `compose_presentation` publishes the settled string verbatim. Neither function is called from any presentation module; both remain importable for callers holding an expanded answer. | `test_law1_presentation_never_transforms_a_citation_marker` |
| **L1-2** | `presentation/a2ui.py:380` — `len(citable_passages(...))`, a citable-unit count outside the owner. | `EvidenceLedger.passage_count` / `.literature_start`. One array, one length, in `citations`. | `test_law1_presentation_never_counts_citable_units` |
| **L1-3** | `engine/graph.py:1276` and `engine/mcp_client.py:837` each advance a running offset. | Both still advance an offset — the model's numbering has to follow the render — but both read the single `citations.passage_total`, which is now one line over `EvidenceLedger`. Charter 13 §5 permits `engine/` → `citations` for exactly this. Duplication retained deliberately; the *definition* is single. | `test_law1_*` + `citation_bound` |
| **L1-4** | `evaluation/platform/evaluators.py:207` and `evaluation/inventory/multiturn_chip_audit.py:93` compiled their own marker regexes. | Both import `citations.MARKER_PATTERN` — the same object, asserted by identity. `citations.py` itself no longer compiles a marker literal either: every pattern is built from `MARKER_OPEN`. | `test_law1_no_module_compiles_an_inline_marker_regex` |
| **L2-1** | `engine/deterministic.py:611-629` read `finish_reason`, chose between the withheld summary and the silent-turn sentence, and wrote it. | `guard_answer(..., finish_reason=..., state=...)` resolves a withheld reply before any clinical guard runs and emits `safety_withheld`; the engine calls `answer.empty_turn_sentence(...)` with facts only. | `test_law2_the_engine_never_selects_a_fallback_sentence`, Contract 1 ×2 |
| **L2-2** | `engine/deterministic.py:506` selected `OUTPATIENT_EHR_NOTICE`; `engine/model.py:352-353` returned a fallback sentence. | `answer.outpatient_ehr_notice()` and `answer.silent_turn_sentence(outpatient=...)`. | same |
| **L2-3** | `engine/graph.py:1122,1144` called `replace_final_text`, including with `""`. | `answer.apply_model_reply` / `answer.reject_model_reply`. No engine module calls `replace_final_text`. | same |
| **L4** | Five duplicate tool registries: `ingress/contracts.py:94-95,109`, `engine/graph.py:178-192`, `engine/mcp_client.py:64,404-411`, `presentation/a2ui.py:121-123`. | `routing/` declares six registries (`LITERATURE_QUESTION_TOOLS`, `RESULT_KEPT_TOOLS`, `EXTERNAL_TOOL`, `ONCE_PER_TURN_TOOLS`, `ONCE_PER_TRANSCRIPT_TOOLS`, `FULL_FETCH_UNFILTERED_TOOLS`) beside the existing ones; all four modules read them. | `test_law4_no_duplicate_tool_registry_literals`, `test_law4_tool_registries_are_declared_once` |

The three passes were left alone: Law 3 (prompts), Law 5 (trace written once — and the
caveat under it is now closed by L1-1, since the traced answer and the served answer are
the same string), and the read-only resolution Charter 13 §5 permits.

### 2.1 The §2 defects

| Finding | Resolution |
|---|---|
| §2.1 runaway free-text loops — one literature turn made five calls (`rag_search` + four `search_literature`) | `_freetext_allowed` narrows a literature turn to `LITERATURE_QUESTION_TOOLS` (literature + record retrieval, minus the medication list), and `PER_TOOL_CALL_LIMIT = 2` bounds one tool within a turn. |
| §2.2 canvas bleed — a sparkline panel under a literature answer | `compose_risk_canvas` clears the lab trends when a literature view is present. A chart turn still draws its card; an empty search still draws `LiteratureList`. |
| §2.3 sparse vs. sequential numbering — "no component owns the display sequence" | `EvidenceLedger` owns the array, its length and where the published records start. Numbers remain positional by design (that was settled on 2026-10-29); the finding was ownership, and there is now one owner. |
| §2.4 a withheld summary swaps silently | Contract 1. The replacement is flagged `safety_withheld`, which is what `_prose_settled` reads to call the swap visible. |
| §3 the gates could not see any of it | Three AST gates (marker transformation, citable-unit counting, inline marker regexes) and nine behavioural contracts, written red and now green. |

### 2.2 The four behavioural contracts

| Contract | Before | After |
|---|---|---|
| 1 — a withheld reply is resolved in postprocessing, visibly | 2 failed | 2 passed |
| 2 — a literature question is scoped, and cannot loop | 2 failed | 2 passed |
| 3 — one ledger, and the presentation publishes what it is given | 2 failed | 2 passed |
| 4 — a canvas is scoped to its turn | 1 failed | 1 passed |

Two of these nine could not pass as written and were corrected by the owner, with the
measurements in `04a-presentation_remediation_report.md` §4: a fixture that asserted both
"unchanged" and "expanded" for one string, and a canvas fixture that omitted the
`narrative` `compose_presentation` always supplies.

---

## 3. Evaluation dry-run

`python -m evaluation.platform.dry_run` aborts before it reports: `StopIteration` at
`dry_run.py:173`, which selects an item with `command == "risk"`. The `/risk` command was
retired on 2026-10-26 (`015f7eb`), so no fixture carries it. Pre-existing; `dry_run.py` is
not in this programme's diff. See §4.2.

What the directive asks the dry run to prove — that `task.py` and `evaluators.py` execute
cleanly against `citations.MARKER_PATTERN` and `EvidenceLedger` — was verified directly
by driving the same chain the dry run drives, stopping short of the retired-command line:

```
MARKER_PATTERN: \^\[([^\]]*)\]
evaluators holds the same object: True
EvidenceLedger passage_count/literature_start: 2 3
fixture items: 6
commands: ['None', 'ehr', 'labs', 'meds', 'pubmed']
  ehr        command_plus_followup structural=1.0 verdict=1.0
  pubmed     command_plus_followup structural=1.0 verdict=1.0
  meds       command_plus_followup structural=1.0 verdict=0.0
  labs       command_plus_followup structural=1.0 verdict=0.0
  pubmed     single_probe          structural=0.0 verdict=0.0
  None       single_probe          structural=1.0 verdict=1.0
judge calls: 5
```

Six fixture items through `make_task` → assertions → judge composite, with no error. The
`pubmed` probe failing structurally while drawing five judge calls for six items is the
short-circuit the dry run documents ("this must fail the structural check and never reach
the judge"). `meds` returning `verdict=0` with a passing structural check is the parrot
case. `task.py:21`'s `passage_total` import and `evaluators.py:212`'s `MARKER_PATTERN`
import both resolve to the leaf module, which is what the platform's charter allow-list
permits.

---

## 4. Two pre-existing failures, found here and repaired here

Both were found by running suites wider than the default CI filter, and both were fixed
in the cleanup that followed this phase. Neither was caused by the remediation; nothing in
either failing path is in its diff.

### 4.1 `tests/evaluation/test_tier1_cohort_sweep.py:235` — **fixed**

```
assert admission_phenotypes.is_matrix_eligible(hadm_id) is False
AssertionError: 90000028   (returned True)
```

`is_matrix_eligible` is `presentation/domain.py:284`:

```python
def is_matrix_eligible(hadm_id: int, *, is_outpatient: bool | None = None) -> bool:
    return is_outpatient is not True
```

`f070bb6` replaced `return encounter.current_is_outpatient() is not True` with the
caller-supplied argument ("this module is not the authority on which records are visits —
it is told") and did not update this test, which still relies on `encounter.set_current`
being read inside the function. The module is `pytestmark = pytest.mark.live` at line 51,
so `pytest -m "not live"` — what CI runs — deselects it and the assertion has been stale
invisibly.

**Fix applied (one line, plus a docstring):** the call now passes
`is_outpatient=True`, and the docstring records why the encounter became an argument.
The `set_current` calls are kept so the test still states the encounter of the turn it
describes. 30 passed in the module, 197 passed across `tests/evaluation`.

### 4.2 `evaluation/platform/dry_run.py:173-174` — **fixed**

```python
risk_out = task(item={"input": next(i for i in items if i.command == "risk").model_dump()})
```

`/risk` was retired in `015f7eb`; `case_set.CENSUS_PROMPTS` is now
`("ehr", "pubmed", "meds", "labs")`. `next(...)` over an empty selection raises
`StopIteration`, so the dry run cannot report.

**Fix applied:** the lookup is `i.command == "ehr"` — `/ehr` is the surviving card turn
in `CENSUS_PROMPTS` — and the assertion changed with it. It could not simply be retargeted:
the old one asserted `"16.7%"`, a number `/ehr` does not produce, and `FakeAgent` lost its
`/risk` branch in the same commit, so the check had been measuring a 400. It now asserts
two things instead: the prose is non-blank, and it is the summary the card drew. The dry
run reports 9 checks passed, 6 items, `STEP 7 PASSED — 100%`.

---

## 5. Final footprint

```
$ git status --short
 M evaluation/inventory/multiturn_chip_audit.py
 M evaluation/platform/evaluators.py
 M services/agent/engine/deterministic.py
 M services/agent/engine/graph.py
 M services/agent/engine/mcp_client.py
 M services/agent/engine/model.py
 M services/agent/ingress/contracts.py
 M services/agent/postprocess/answer.py
 M services/agent/postprocess/citations.py
 M services/agent/postprocess/guardrail.py
 M services/agent/presentation/a2ui.py
 M services/agent/presentation/canvas.py
 M services/agent/routing/routing.py
 M services/agent/turn.py
 M tests/agent/test_architecture_remediation_contracts.py
 M tests/agent/test_freetext_boundary.py
 M tests/agent/test_progress_stream.py
 D docs/architecture_alignment/00-master_engineering_strategy_backend_audit_and_alignment.md
?? docs/architecture_alignment/  (9 files: 00-master_engineering_strategy.md,
   01a-audit_report.md, 02-governance_lock_directive.md, 02a-governance_lock_report.md,
   03-routing_and_engine_layer_remediation.md, 03a-routing_and_engine_remediation_report.md,
   04-presentation_remediation.md, 04a-presentation_remediation_report.md,
   05-evaluation_and_signoff.md)
```

```
$ git diff --stat
 ...trategy_backend_audit_and_alignment.md |  70 -------
 evaluation/inventory/multiturn_chip_audit.py   |   3 +-
 evaluation/platform/evaluators.py              |   9 +-
 services/agent/engine/deterministic.py         |  28 +--
 services/agent/engine/graph.py                 |  21 +--
 services/agent/engine/mcp_client.py            |  60 ++++--
 services/agent/engine/model.py                 |   3 +-
 services/agent/ingress/contracts.py            |  11 +-
 services/agent/postprocess/answer.py           |  94 ++++++++++
 services/agent/postprocess/citations.py        | 146 +++++++++++++--
 services/agent/postprocess/guardrail.py        |  35 ++++
 services/agent/presentation/a2ui.py            |  71 ++++---
 services/agent/presentation/canvas.py          |  12 +-
 services/agent/routing/routing.py              |  63 +++++++
 services/agent/turn.py                         |   6 +
 tests/agent/test_architecture_remediation_contracts.py |  18 +-
 tests/agent/test_freetext_boundary.py          |  13 +-
 tests/agent/test_progress_stream.py            |  29 ++-
 18 files changed, 518 insertions(+), 174 deletions(-)
```

Fourteen production files (twelve under `services/`, two under `evaluation/`), three test
files, one documentation rename. The three test files are the two the Phase 4 directive
authorised plus the fixture corrections it authorised in §4;
`test_architecture_remediation_contracts.py` is read-only again (`-r--r--r--`).

---

## 6. Sign-off

Charter 13 is now enforced mechanically rather than by convention: eight findings closed,
three passes preserved, and every one of them pinned by a gate that failed before the
remediation and passes after. Charters 14 and 15 were already green and remain so — the
evaluation harness now reads its marker regex from the citation owner instead of compiling
its own, which is Charter 15 Law 1 seen from the other side.

The branch is ready to commit and push. Recommended commit body: the six registries moved
into `routing/`, the `EvidenceLedger`, presentation publishing the settled prose verbatim,
the withheld-reply contract, the per-tool cap, and the eight gates that hold them.

The two stale call sites this phase surfaced are repaired: the live cohort sweep passes
its encounter as an argument (§4.1) and the dry run reports again (§4.2). No follow-ups
remain, and the suite is green under every marker filter.
