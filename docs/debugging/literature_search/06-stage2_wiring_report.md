# Stage Gate 2 — wiring, and the closed trace boundary

2026-11-02. The backend pipeline proved in Gates 1–1E is now the thing `/ehr` draws, the
curated benchmark table it replaced is gone, and the trace records the surface a reader was
shown rather than only the tools the turn called.

Status: **complete, green, and verified live on six admissions.**

- `pytest tests -m "not live"` — **1684 passed, 1 skipped, 44 deselected**
- `python -m evaluation.platform.dry_run` — **STEP 7 PASSED, 100%**
- Live `/ehr` across 6 admissions — 6/6 turns composed, 6/8 verified rows, 3 honest empties

---

## 1. What was deleted

`services/agent/presentation/domain.py` was a curated corpus in a presentation module: nine
`Benchmark` records, a phenotype map keyed by `hadm_id`, an `UNRESOLVED` set and a
`NON_ADMISSION` set. Every one of those is a clinical or corpus fact compiled into the layer
that draws a table, and each was a second source of truth for something the engine now
settles. The module is now 92 lines: display precision, the shared label vocabulary, and two
predicates that read an argument rather than a table.

- `_literature_view` is unchanged; `_usable_articles` moved here from the corpus.
- `is_matrix_eligible(*, is_outpatient=None) -> bool` — pure. It used to take a `hadm_id`
  and consult `NON_ADMISSION`; the encounter arrives as a boolean from the orchestrator,
  which is where the fact lives (`00d7` Row 11).
- `admission_eligible(tool_calls, *, is_outpatient=None)` in `a2ui` now delegates to it.
  The old version scanned `tool_calls` for a `hadm_id` and compared it against the set.
- `RISK_DECIMALS`, `display_probability`, `FEATURE_LABELS`, `humanize`, `label_for`,
  `_LITERATURE_FIELDS` survive. `BENCHMARKS`, `benchmarks_for_phenotype`,
  `ADMISSION_PHENOTYPE`, `UNRESOLVED`, `NON_ADMISSION`, `GENERAL_INPATIENT` do not.

`canvas.compose_risk_canvas` lost `phenotype` and `query`; `_comparative_matrix` lost its
phenotype fallback. Rows come from `comparators` — what `deterministic.extract_comparators`
verified on the turn — and the canvas draws them in the order it received them. It does not
select, rank or add.

## 2. What was built

**`turn.py`** — `compose_presentation(..., comparators=state.get("comparators") or [])`, and
the trace writes moved *behind* composition. `set_trace_io` now receives
`presentation["answer"]`, so a trace cannot disagree with the surface it explains.

**`telemetry/observability.py`** — `record_presentation(presentation)`, writing on the root
span's metadata: the component names the canvas composed; the `ComparativeRiskMatrix` exactly
as the renderer received it (`baseline`, `rows`, `baseline_note`); the numbered channel by
number, kind and label; the drawer rows by number and PMID. No note text, no abstracts. Best
effort, never raises (rule 1).

**`postprocess/citations.py`** — `_searched_for_evidence`. `search_literature` now has two
callers: the model, asking what the literature says, and the readmission retrieval, asking
for the candidate pool an extraction turn will verify rates out of. Both arrive in
`tool_calls` under one name — one tool, one trace contract — so the call's own declared
`scope` separates them. Without the filter the `/ehr` turn drew itself as a literature turn:
risk widget suppressed, a caption about PubMed records, twenty-five candidates published as
though the answer had cited them.

## 3. The seam, restated

`run_command(..., extraction_llm=None)`, defaulting to
`deterministic.extraction_llm_factory`. `tests/agent/conftest.py` installs an autouse fixture
that makes the factory return `None`, which is a state the wiring already understands
(`source: "unavailable"`, no table rows). Production code inspects no model-name string:
`"test-model"` appears nowhere in `services/`.

## 4. Two defects found while verifying

**`NON_ADMISSION` ReferenceError.** `admission_eligible` still read the deleted set, and 79
tests failed on the `NameError` the moment the corpus went. Fixed by making the rule pure and
passing the encounter in. This is the shape the demolition was for: the fact was reaching the
presentation layer from a table instead of from the orchestrator.

**`mcp_client.REPO_ROOT` was the `services` directory.** `parents[2]` of
`services/agent/engine/mcp_client.py`; the constant was written when the module sat at
`services/agent/`. The stdio child is started with `cwd=REPO_ROOT` and
`-m services.mcp.server`, so the transport could not start at all — first "No module named
'services'", then, with `PYTHONPATH` masking that, `services/mcp/server.py` shadowing the
installed `mcp` SDK and importing itself. Nothing in production starts a stdio child (the
demo runs over `http`), and `tests/agent/test_mcp_stdio.py` builds its own
`StdioServerParameters` with `cwd` hard-coded to the repo root, so nothing caught it. It
failed the first time a local harness tried. `parents[3]`, plus a test asserting the spawner's
cwd is the root and naming the two off-by-one shapes.

## 5. Tests: what changed and why

Read-only test files were unprotecting → edited → re-protected. No assertion was weakened; the
ones that changed assert less about a deleted table and more about a supplied comparator.

| File | Change |
| --- | --- |
| `tests/agent/test_phenotype_coverage.py` | **deleted.** It tested the classifier that selected benchmark rows by `hadm_id`. The classifier is gone; the verification it stood in for is `test_live_literature_extraction.py`. |
| `tests/evaluation/test_tier1_cohort_sweep.py` | `UNSERVED` is now declared in the test (a corpus fact belongs to the corpus's test); `is_matrix_eligible` called with the pure signature. |
| `tests/agent/test_benchmark_sources.py` | **rewritten** around a comparator fixture. Same invariant — a table cohort is evidence, so it gets a source, a drawer row and a label — without a corpus. Added the case the corpus made impossible: a turn that verified nothing owes no source. |
| `tests/agent/test_a2ui_literature.py` | matrix section rewritten from phenotype selection to comparator supply: order preserved from the engine, no re-sort, honest empty, encounter gate, `/pubmed` draws no table. |
| `tests/agent/test_a2ui.py` | the comparative-turn test now asserts the table is **absent** on a `/pubmed` turn, which is the narrowing this gate made. |
| `tests/agent/test_reference_invariant.py` | the literature row is published when a comparator is supplied, and not invented when none is. |
| `tests/agent/test_ehr_section_citation_order.py` | the note channel is numbered 1..4 with no cohort appended; the old assertion required a benchmark at 5. |
| `tests/agent/test_literature_multi_search.py` | new class: the comparator retrieval is not the turn's literature. |
| `tests/agent/test_trace_presentation.py` | **new**, 5 tests: the matrix reaches the trace with its rows and its note, the empty table is recorded as empty, the numbered channel travels with the surface, no span is a no-op. |
| `tests/agent/test_mcp_stdio.py` | new test pinning the production spawner's cwd. |

## 6. Live verification

`scripts/agent/verify_gate2_live.py` drives the production path — real MCP toolbox, real
Vertex client, real PubMed — for one `/ehr` turn per admission, then composes the
presentation from the returned state. The table below is the composed surface, not a
reconstruction.

| hadm | derived target (procedure) | traced calls | baseline | rows | table |
| --- | --- | --- | --- | --- | --- |
| 90000090 | Femoral neck fracture (Hemiarthroplasty) | sections, predict, 2 searches | 0.0948 | 2 | drawn |
| 90000069 | Cervical spondylosis (cervical decompression) | sections, predict, 2 searches | 0.0751 | 2 | drawn |
| 90000081 | Pleural effusion (Thoracentesis) | sections, predict, 2 searches | 0.0947 | 0 | drawn, honest empty |
| 90000001 | Tibial plateau fracture (ORIF) | sections, predict, 2 searches | 0.1674 | 0 | drawn, honest empty |
| 90000019 | Total hip arthroplasty | sections, predict, 1 search | 0.1357 | 2 | drawn |
| 90000037 | Coronary artery disease (cardiac catheterization) | sections, predict, 2 searches | 0.2188 | 0 | drawn, honest empty |

Every turn composed `Column, CitedNarrative, RiskInfluencers, ComparativeRiskMatrix, Text`,
and every source array numbered continuously: note passages first, then the cohort rows.

**Rows.**

| hadm | PMID | journal · year | rate | tier |
| --- | --- | --- | --- | --- |
| 90000090 | 41004383 | Anesthesia and analgesia · 2026 | 4.60% vs. 4.97% (comparison) | propensity-matched retrospective cohort |
| 90000090 | 25550095 | Arch Orthop Trauma Surg · 2015 | 11.9% (point) | registry review |
| 90000069 | 28142261 | Neurosurgical focus · 2017 | 2.6% vs. 0.4% (comparison) | database review |
| 90000069 | 26872307 | Spine · 2018 | 6.20% (point) | retrospective cohort |
| 90000019 | 33422393 | J Arthroplasty · 2021 | 3.3% (point) | retrospective review |
| 90000019 | 32742739 | Geriatr Orthop Surg Rehabil · 2020 | 4.9% (point) | database review |

Two of the six rows are two-arm results and both are published as two figures (`A% vs. B%`,
`rate_kind: comparison`), which is the F4 rule. One took a single arm and said so: 33422393
states `7.3% vs 3.3%` for conversion-vs-primary THA, the cohort summary names the primary
group, and the published figure is the primary arm's.

**The three empties.** Pleural effusion, tibial plateau fracture and coronary artery disease
each retrieved 10–18 clinically aligned candidates and verified none that stated a 30-day
rate for the admission's own condition. Each drew the baseline with the sentence *"No
published 30-day readmission rate for this admission's primary condition was stated in the
abstracts retrieved for it… No cohort has been substituted."* 90000081 verified one row in
Gate 1E and none here on identical code: PubMed's relevance order and retrieved set move
between calls, recorded as a residual at Gate 1D.

**Trace boundary.** `set_trace_io`, `record_turn_tools` and `record_presentation` all run
after composition, inside the turn span. Langfuse is **not configured in this environment**
(`configured() is False`), so the metadata write is verified by unit test against a stub span,
not against the live project; the ordering property — the write lands while a span is
current — is the one `test_trace_lifecycle.py` pins.

## 7. Residuals

1. **Retrieval is not reproducible across calls.** Same code, same temperature, different
   retrieved set. Per-admission row counts are not a trend and are not asserted anywhere.
2. **Live trace metadata is unverified against Langfuse**, for the configuration reason above.
3. **Two of six admissions still verify nothing.** The pipeline is not silent about it, but
   thirty-day rates for a pleural effusion or a tibial plateau fracture are thin in abstracts;
   whether to widen the outcome clause further is a decision for the next gate, not a defect.
