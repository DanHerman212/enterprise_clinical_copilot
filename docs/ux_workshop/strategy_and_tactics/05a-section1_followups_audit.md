# Section 1 Audit Report — `followups.py` and Runtime Context Availability

**Parent document:** `05-post_eval_ux_augmentation.md` (Phase 05, Section 1.2)
**Status:** Complete — awaiting Architect review before Section 2
**Scope:** Read-only. No files were changed in producing this report.
**Date:** 2026-10-23

---

## 1. Call site, contract and tracing

### 1.1 Where it is invoked

One call site, not three: `services/agent/http.py:444`, inside `_compose_success`, gated on
`want_follow_ups`, which arrives as `parsed["follow_ups"] is True`. Both routes converge there —
`/ask` at `http.py:711` and `/ask/stream` at `http.py:964` — so `deterministic.py`, `graph.py`
and the tool modules never call it. Chip generation is downstream of every path and is currently
path-agnostic.

### 1.2 Arguments passed

```python
followups.suggest(
    question,
    presentation["answer"],           # guarded, post-citation answer
    model=chain.MODEL_ID,
    scope=presentation.get("scope"),  # "patient_chart" | "literature" | None
)
```

The answer passed is the one that will actually be served, not the model's first draft: a chip
built on a sentence a guardrail removed would ask about something the reader never saw.

### 1.3 Return contract

- `async suggest(...) -> list[str]`.
- At most 3 (`FOLLOW_UP_COUNT`), each ≤ 200 characters (`_MAX_QUESTION_CHARS`), whitespace
  collapsed. An over-length question is dropped rather than truncated.
- An empty answer short-circuits to `[]` before any model call.
- Every failure mode — timeout, refusal, unparseable JSON, empty list — returns `[]`.
- Timeout 8.0 s, no retries (`FOLLOW_UP_TIMEOUT_SECONDS`).
- Served as the envelope field `follow_ups` (`http.py:525`): one plain list in both the JSON body
  and the terminal SSE frame. There is no separate SSE event for chips.

### 1.4 Langfuse tracing

**None.** The generation call is `_build_llm(model).ainvoke([...])` with no `config`, no callbacks,
and `followups.py` imports neither `observability` nor the Langfuse handler. It is the only model
call in the chain that leaves no span, no input and no output in Langfuse.

The only evidence a chip request happened at all is the failure log line:

```
logger.info("follow-ups were not proposed trace=- error=%s", type(error).__name__)
```

It carries no trace id, so a missing chip row cannot be joined to the turn it belonged to.

---

## 2. In-memory data available at the call site

`_compose_success(question, state, trace, *, want_follow_ups, outpatient)` gives a generator:

- `state["tool_calls"]` — full tool responses (passages, trends, medications, risk payload)
- `state["command"]` — the wire name, or `None` for free text
- `state["command_payload"]` — `/meds` only
- `state["replayed_tool_calls"]` — earlier turns' calls, replayed into this one
- `state["manifest_answered"]` — bool
- `presentation` — answer, sources, scope, `verified`, citation map

### 2.1 Per Turn-1 path

| Turn 1 path | Chart signal already in memory | Note |
|---|---|---|
| `/ehr` | `rag_search_sections` passages (plus record-wide `rag_search` fallback), `predict_readmission` payload: probability/threshold display strings, ranked drivers | richest path — sections and risk in one turn |
| `/meds` | `discharge_medications` passages, plus `command_payload` holding the **verified** rows (name, dose, frequency) and the `dropped` rows | the verified rows are safer than re-parsing prose |
| `/labs` | `get_lab_trends`: per-analyte `series`, `reference_low`/`reference_high`, unit | the discharge point is `series[-1]` |
| `/risk` | `predict_readmission` only | **never offers chips**: the answer is prose-free by design, `presentation["answer"]` is `""`, and `suggest` returns `[]` on its first line |
| `/pubmed` | `search_literature` articles only | no chart state at all; scope routes to the literature wording |
| free text (ReAct) | whatever the loop called, plus replayed Turn-1 calls | bounded set, see §4 |

### 2.2 Gaps a deterministic engine must close

- **`hadm_id` is not available.** It is not on `state` and is not passed to `_compose_success`.
- **The pre-flight manifest is not available.** `admission_context` (encounter line, age, length of
  stay, discharge location, prior admissions) is consumed inside `graph.answer` into the prompt and
  survives only as the boolean `state["manifest_answered"]`.
- **The encounter kind arrives only as a boolean** (`outpatient`), not as data a template can read.

Two ways to close this, and the cheaper one is (a):

- **(a) Per-turn chart-state dict** assembled in `_run_chain` — `hadm_id`, manifest, and the tool
  payloads, all of which except the manifest are already in `state` — threaded into
  `_compose_success`. Pure in-memory, no new dependency.
- **(b) Cached patient-state helper** — requires a store lookup, which introduces a network
  dependency into a path that is currently pure in-memory, on a call that cannot fail the turn
  anyway.

---

## 3. Reusable entity and phenotype extraction logic

### 3.1 Promotable as-is

- **`evaluation/inventory/dev_fact_sheet.py` — `DRUG_CLASSES`:** anticoagulant, beta-blocker,
  diuretic and taper regexes, **brand-inclusive** (Coumadin, Lovenox, Toprol-XL). This is the
  version to take: the generic-only table in `cohort_provenance.py` reported anticoagulant absent
  on `90000015`, a false negative on the exact fact a trap would be built from.
- **`dev_fact_sheet.py` helpers:** `_excursion` (how far outside reference, in range-widths) is
  exactly the "which lab was worst" ranking rule; `_discharge_point` (`series[-1]`);
  `_section_map`; `_first_statement` (strips `1.` list markers).
- **`evaluation/inventory/cohort_provenance.py`:** `_medication_facts`, `_lab_facts` (abnormal
  detection against reference), `_note_facts` (section presence), `_risk_facts`.
- **`services/mcp/retrieval/sections.py`:** `SUMMARY_SECTIONS` (the canonical four),
  `KNOWN_HEADINGS`, `extract_section`. Section detection already lives in the service, not only in
  the inventory.
- **`services/agent/deterministic.py::parse_medications`:** verified medication rows with a
  source-text check. The correct source for a "how is X dosed" chip is these rows, not a regex over
  prose.
- **`services/agent/admission_phenotypes.py::phenotype_of_query`:** the existing text→phenotype cue
  vocabulary. `ADMISSION_PHENOTYPE` itself is keyed by `hadm_id` and must not be the selector under
  guardrail 1.

### 3.2 Does not exist — new extractors required

- Follow-up provider or appointment extraction (`09/21/2007`, "Dr. A", a return interval).
- Return-precaution phrasing.
- Procedure phrases ("Roux-en-Y", "percutaneous coronary intervention").
- Condition naming beyond the discharge-diagnosis sentence.

The discharge-diagnosis sentence plus the four drug classes cover most of what the dev cohort can
answer, so the gap is bounded but real.

---

## 4. Turn-2 routing compatibility

A chip is sent as `question` with `turns=[replay of Turn 1]`; `parse_agent_request` resolves it to
free text and `_run_chain` hands it to the ReAct loop.

**Allowed tools** (`routing.FREETEXT_TOOLS`, applied by `graph._freetext_allowed`):

- `rag_search`
- `rag_search_sections`
- `discharge_medications`
- `get_lab_trends`

**Two conditional additions:**

- `search_literature` **only** when `routing.asks_for_literature(question)` matches its cue list:
  literature, evidence, study/studies, trial/trials, meta analysis, systematic review, pubmed,
  guideline(s), efficacy, effectiveness, benchmark(s).
- `predict_readmission` **only** when `routing.wants_risk_driver(question, intent)` — either a
  declared `intent == "risk_driver"`, or `asks_why_risk_driver`, which requires "risk" or
  "readmission" **and** a driver word **and** a why word.

### 4.1 Constraints that bind the Section 2 templates

1. **Literature cues must be avoided** in medication and lab chips. A chip containing "evidence",
   "studies" or "guidelines" binds PubMed and can return a `LiteratureList` over a chart answer.
2. **Risk-driver chips must trip the cue test** or declare themselves. The request already supports
   a declared `intent` field (`contracts.py:465` → `routing.py:772`), used today by the driver-bar
   affordance, so a taxonomy chip can declare its intent instead of relying on phrasing.
3. **No chip may ask for the number.** Free text never produces a bare risk score — only display
   strings and driver names — so a chip asking for a score is unanswerable by construction.

Turn-1 tool calls are replayed into the Turn-2 prompt, so a chip may legitimately say "these
medications" or "that result" and be understood.
