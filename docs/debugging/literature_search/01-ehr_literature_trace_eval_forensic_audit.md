# Stage 1 Forensic Audit — `/ehr` Construction, Literature Matrix, Trace Boundary & Eval Blind Spot

**Date:** 2026-11-02
**Mode:** read-only. No files modified by this audit.
**Scope:** `services/agent/`, `services/mcp/`, `shared/`, `evaluation/platform/`

---

# 1. `/ehr` (`summarize`) execution flow

## Entry point — `services/agent/engine/graph.py:953-987`

```python
        with observability.observation(
            command_name,
            trace_metadata=None if turn_root is not None else command_meta,
        ) as root_span:
            command_state = await run_command(
                toolbox,
                routing.command_for(question_kind),
                hadm_id,
                topic=topic,
                question=question,
                model=model,
                on_event=on_event,
                callbacks=[tracing] if tracing is not None else None,
                prose_stream=prose_stream,
                trace_metadata=command_meta,
            )
            command_state["wants_literature"] = turn_is_about_literature(
                question, question_kind)
            _settle(command_state, prose=final_text(command_state))
```

## Command table — `services/agent/routing/routing.py:237-255`

```python
    Command(
        wire="summarize",
        slash="/ehr",
        tool="rag_search_sections",
        args={"hadm_id": "{hadm_id}"},
        supplementary=(ToolSpec("predict_readmission", {"hadm_id": "{hadm_id}"}),),
        forbidden=frozenset({"rag_search", "discharge_medications",
                             "search_literature", "get_lab_trends"}) | PREFLIGHT_TOOLS,
        sections=SUMMARY_SECTIONS,
        streams_prose=True,
        prose_is_model_output=True,
    ),
```

`Command.tools` (`routing.py:182-185`) is `(self.tool, *(spec.tool for spec in self.supplementary))`.

## Executor — `services/agent/engine/deterministic.py:387`, `async def run_command(...)`

```python
    calls = [{"name": command.tool, "args": arguments, "id": f"{command.wire}-1"}]
    supplementary = supplementary_arguments(command, hadm_id, topic)
    calls += [
        {"name": name, "args": extra, "id": f"{command.wire}-{index}"}
        for index, (name, extra) in enumerate(supplementary, start=2)
    ]
```

(`deterministic.py:417-424`; dispatched through `_execute_tool_calls` at `:445-456`,
`services/agent/engine/mcp_client.py:711`)

## Conditional third call — `deterministic.py:468-485`

```python
    if command.wire == "summarize" and hadm_id is not None and not _passage_count(recorded):
        fallback = [{
            "name": "rag_search",
            "args": {"hadm_id": hadm_id, "query": EHR_RECORD_QUERY},
            "id": f"{command.wire}-record",
        }]
```

## Tool calls executed on `/ehr`

| order | tool | trigger |
|---|---|---|
| 1 | `rag_search_sections` | always (`routing.py:240`) |
| 2 | `predict_readmission` | always (`routing.py:244`) |
| 3 | `rag_search` (record-wide) | only when call 1 returned zero passages (`deterministic.py:468`) |

## Is `search_literature` ever invoked on `/ehr`?

**No.** It is a member of the command's `forbidden` set (`routing.py:249`) and `/ehr`
declares no `ToolSpec` for it. The only literature call in the system belongs to the
`/pubmed` command (`routing.py:256-283`, `tool="search_literature"`).

---

# 2. Literature matrix & phenotype construction

## `turn_phenotype` — `services/agent/presentation/a2ui.py:291-334`

```python
def turn_phenotype(tool_calls: list[dict],
                   evidence: list[dict] | None = None,
                   *, is_outpatient: bool | None = None) -> str:
    calls = [*(evidence or []), *(tool_calls or [])]
    hadm_id = next(
        (int((call.get("args") or {})["hadm_id"]) for call in calls
         if isinstance(call, dict)
         and isinstance((call.get("args") or {}).get("hadm_id"), int)
         and not isinstance((call.get("args") or {}).get("hadm_id"), bool)),
        None,
    )
    if hadm_id is not None and not is_matrix_eligible(hadm_id, is_outpatient=is_outpatient):
        return ""
    return resolve_phenotype(
        _note_sections(calls),
        fallback=GENERAL_INPATIENT if hadm_id is not None else "",
    )
```

Called from `compose_presentation` (`a2ui.py:591`):

```python
        phenotype=turn_phenotype(tool_calls, evidence_calls, is_outpatient=is_outpatient),
```

## Helpers it depends on — `services/agent/presentation/domain.py`

`is_matrix_eligible` (`:284-302`) — the whole body:

```python
    return is_outpatient is not True
```

`hadm_id` is retained in the signature only; eligibility is now purely "is this record
not an outpatient visit".

`resolve_phenotype` (`:449-473`):

```python
    mapping = {str(key): str(value or "") for key, value in (sections or {}).items()}
    for section in SIGNAL_SECTIONS:
        key = phenotype_of_text(mapping.get(section, ""))
        if key:
            return key
    rest = " ".join(
        text for name, text in sorted(mapping.items()) if name not in SIGNAL_SECTIONS
    )
    return phenotype_of_text(rest) or fallback
```

Supporting vocabulary in the same module:

| symbol | line | role |
|---|---|---|
| `BUCKETS` | `:134-203` | `Bucket` records keyed by phenotype; each carries `query_terms` |
| `GENERAL_INPATIENT` | `:203` | `"general_inpatient"` — the all-cause fallback |
| `ADMISSION_PHENOTYPE` | `:227-250` | the reviewed map, keyed by `hadm_id` |
| `NON_ADMISSION` | `:258-261` | corpus records that are visits, not stays |
| `UNRESOLVED` | `:265` | records with neither note nor feature row |
| `bucket_for` | `:268-282` | map lookup, used by pre-flight, not by the surface |
| `phenotype_of_query` | `:480-...` | condition-lock for `/pubmed`; longest term wins |
| `SIGNAL_SECTIONS` | — | order `resolve_phenotype` tries sections in |

## `_draws_matrix` — `services/agent/presentation/canvas.py:114-152`

```python
    wire = str(command or "").strip().lower()
    if wire == "summarize":
        return True
    if wire == "literature":
        return bool(has_literature) and bool(has_estimate) \
            and phenotype_of_query(query) == phenotype
    return False
```

## `_comparative_matrix` — `canvas.py:155-229`

```python
    if not phenotype:
        return None
    benchmarks = benchmarks_for_phenotype(phenotype)
    if not benchmarks and phenotype != GENERAL_INPATIENT:
        benchmarks = benchmarks_for_phenotype(GENERAL_INPATIENT)
    if not benchmarks:
        return None
    rows = [
        {key: value for key, value in row.items() if key != "matched_features"}
        for row in benchmarks
    ]

    component: dict = {
        "id": "comparison",
        "component": "ComparativeRiskMatrix",
        "rows": rows,
        "baseline_note": "",
    }
    if estimate is None:
        component["baseline_note"] = (
            "No readmission estimate was requested for this turn, so the table shows "
            "the published cohorts alone."
        )
        return component

    component["baseline"] = {
        "source": _BASELINE_SOURCE,
        "probability": float(estimate["probability"]),
        "threshold": float(estimate["threshold"]),
        "decision": int(estimate.get("decision") or 0),
        "cohort": "Index admission, this patient",
        "provenance": f"Model {estimate.get('model_version') or 'unknown'}",
    }
    return component
```

Row selection: `benchmarks_for_phenotype` (`domain.py:951-980`), which filters
`BENCHMARKS` by `phenotype` and by `horizon_days == MODEL_HORIZON_DAYS` (30), ordering by
evidence tier then PMID.

## The static benchmark table — `services/agent/presentation/domain.py:620-925`

Schema — `CohortBenchmark`, frozen dataclass at `:580-618`:

```python
@dataclass(frozen=True)
class CohortBenchmark:
    pmid: str
    phenotype: str
    reported_rate: str
    cohort_summary: str
    evidence_tier: str
    horizon_days: int
    rate_sentence: str
    journal: str = ""
    year: str = ""
    title: str = ""
    authors: str = ""
    matched_features: tuple[str, ...] = ()
```

`MODEL_HORIZON_DAYS = 30` (`:573`). Eight rows, keyed by PMID:

| line | PMID | phenotype | reported rate | journal · year |
|---|---|---|---|---|
| 620 | 37495280 | `heart_failure_cardiorenal` | 13.2% | Journal of the American College of Cardiology · 2023 |
| 649 | — | `copd_respiratory` | 11% | Respiratory Medicine · 2023 |
| 685 | — | `copd_respiratory` | ≈20% | Am J Respiratory and Critical Care Medicine |
| 713 | — | `spine_neurosurgery` | 3.8% | North American Spine Society Journal |
| 746 | — | `orthopedic_arthroplasty` | 3.4% | The Journal of Arthroplasty · 2022 |
| 782 | — | `spine_neurosurgery` | 5.5% | Spine · 2020 |
| 818 | — | `gi_hepatobiliary_surgery` | 17.4% | Journal of Gastroenterology and Hepatology |
| 851 | — | `sepsis_infectious` | 18.4% | JAMA Network Open |
| 886 | 19339721 | `general_inpatient` | 19.6% | The New England Journal of Medicine · 2009 |

Full verbatim row — the shape every row carries (`:886-920`):

```python
    "19339721": CohortBenchmark(
        pmid="19339721",
        phenotype=GENERAL_INPATIENT,
        reported_rate="19.6%",
        journal="The New England Journal of Medicine",
        year="2009",
        title=(
            "Rehospitalizations among patients in the Medicare fee-for-service "
            "program."
        ),
        authors="Jencks SF, Williams MV, Coleman EA.",
        rate_sentence=(
            "Almost one fifth (19.6%) of the 11,855,702 Medicare beneficiaries who had "
            "been discharged from a hospital were rehospitalized within 30 days, and "
            "34.0% were rehospitalized within 90 days."
        ),
        cohort_summary=(
            "Hospital-wide, all conditions — Medicare fee-for-service claims for "
            "11,855,702 beneficiaries discharged from a hospital, 2003-2004, national "
            "analysis"
        ),
        evidence_tier="Observational Cohort",
        horizon_days=30,
        matched_features=(),
    ),
```

## How matrix citations reach `sources` / `references`

Definition: `a2ui.py:807-916` (`register_benchmark_sources`). Wiring: `a2ui.py:822-833`.

```python
    benchmark_sources, benchmark_references = register_benchmark_sources(
        canvas, start=len(sources) + 1,
        existing_pmids=[entry.get("pmid") for entry in sources],
    )
    published = sources + benchmark_sources
    references = reference_rows(published)
```

Per `ComparativeRiskMatrix` row (`a2ui.py:840-916`):

```python
            citation = str(row.get("source") or "").strip() or f"PMID {pmid}"
            label = f"{index} · {citation}"
            ...
            row["source_index"] = index
            row["source_label"] = label
            sources.append({
                "cite": index,
                "kind": "Literature",
                "badge": _LITERATURE_BADGE,
                "title": title,
                "authors": authors,
                "label": label,
                "subtitle": subtitle,
                "journal_citation": journal_citation,
                "section": citation,
                "text": passage,
                "tier": str(row.get("evidence_tier") or "").strip(),
                "pmid": pmid,
                "url": url,
            })
```

So `3 · Spine`, `4 · The Journal of Arthroplasty · 2022` are composed here from
`row["source"]`, which `benchmarks_for_phenotype` builds from the hard-coded `journal` and
`year` (`domain.py:975-980`). Numbering starts after every note passage and every recorded
literature PMID already present in `sources`.

---

# 3. `search_literature` and the MCP backend

`services/mcp/tools/literature.py` — `_fetch` at `:330`, `search_literature` at `:391`.

```python
            found = await _get(
                client,
                f"{NCBI_EUTILS_BASE}/esearch.fcgi",
                {
                    **common,
                    "term": term,
                    "retmode": "json",
                    "retmax": str(limit),
                    "sort": SORT_ORDER,
                    "datetype": "pdat",
                    "reldate": str(RECENCY_DAYS),
                },
            )
            found.raise_for_status()
            identifiers = _pmids(found.json())
            ...
            records = await _get(
                client,
                f"{NCBI_EUTILS_BASE}/efetch.fcgi",
                {**common, "id": ",".join(identifiers), "retmode": "xml"},
            )
```

Constants (`:78-108`): `SORT_ORDER = "relevance"`, `RECENCY_YEARS = 5`,
`RECENCY_DAYS = RECENCY_YEARS * 366`, `DEFAULT_MAX_RESULTS = 3`,
`PUBLICATION_TYPE_FILTER = None`.

Record shape returned by `_articles` (`:352-381`): `pmid`, `title`, `journal`,
`pub_date`, `authors`, `abstract`, `url`. `_abstract` (`:222-238`) concatenates labelled
`<AbstractText>` fragments into one string.

**No rate extraction exists in the path.** There is no parser, no numeric rate field and no
`readmission_rate` key anywhere between the E-utilities response and the matrix. The
30-day figures in the table are hand-curated constants in `domain.BENCHMARKS`, each with
the abstract sentence it was read from reproduced as `rate_sentence`, and the module
docstring at `domain.py:507-568` states the curation rules (horizon match, qualifier
preserved, one row per record). `scripts/agent/find_readmission_benchmarks.py:13` states
that the helper does not write the benchmark file.

---

# 4. Langfuse trace boundary

## Write site — `services/agent/turn.py:255-270`

```python
    observability.set_trace_io(
        question,
        (guarded["answer"] or "").strip() or command_trace_output(state),
    )
    observability.record_turn_tools(
        state.get("tool_calls") or [],
        replayed=len(state.get("replayed_tool_calls") or []),
    )
```

## Composition site — `services/agent/turn.py:297`

```python
    presentation = compose_presentation(
```

`compose_presentation` is called **after** both trace writes. Nothing it returns is
written back to the trace.

## What `set_trace_io` writes — `services/agent/telemetry/observability.py:268-310`

```python
        langfuse.get_client().set_current_trace_io(
            input=(user_input or None), output=(output or None),
        )
    ...
    span.update(input=(user_input or None), output=(output or None))
```

## Fields currently on the trace

| record | fields |
|---|---|
| trace row | `input` = question, `output` = served prose (or `command_trace_output(state)` when the turn has no prose) |
| root span | `input`, `output` (the same two), `metadata.tool_calls[]` = `{name, args, error\|returned}`, `metadata.replayed_calls` |
| tags | `tools:<n>`, intent/outcome tags |

`canvas`, `phenotype`, `sources`, `references` and `ComparativeRiskMatrix` are not
written. The exclusion is mechanical rather than decided: `set_trace_io` takes two strings
and is called before the composed envelope exists. The only canvas-shaped artefact that can
reach the trace row is `command_trace_output(state)` on a prose-less turn, which serialises
the payload as text rather than the composed presentation contract.

---

# 5. Evaluation harness blind spot

## What is captured — `evaluation/platform/task.py:336-362`

```python
        output = TaskOutput(
            turn_1_response=answers[0],
            turn_responses=answers,
            turn_prompts=prompts,
            turn_follow_ups=follow_ups,
            turn_components=components,
            turn_verified=verified,
            turn_sources=sources,
            turn_tool_calls=tool_calls_per_turn,
            turn_1_verified=verified[0],
            tool_evidence=tool_evidence,
            session_id=session_id,
            langfuse_trace_id=turn_traces[0],
            turn_trace_ids=list(turn_traces),
            turn1_trace=turn_traces[0],
            turn_1_status=first.get("status"),
            turn_1_command=first.get("command") or first.get("command_executed"),
            turn_1_components=components[0],
            turn_1_card=first.get("a2ui") if isinstance(first.get("a2ui"), dict) else None,
            turn_1_error=first.get("error"),
        )
```

`components` is **names only** — `task.py:144-147`, via `shared/a2ui.py:18-32`:

```python
def _components(response: dict) -> list[str]:
    """The component names a turn rendered, in surface order."""
    return (a2ui.components(response.get("a2ui"))
            or list(response.get("a2ui_components") or []))
```

```python
def components(node: Any, out: set[str] | None = None) -> list[str]:
    """Every component name in one A2UI payload node, sorted and deduplicated."""
```

## Deterministic checks — `evaluation/platform/evaluators.py:88-137`

```python
    if not str(output.get("turn_1_response") or "").strip():
        failures.append("turn_1_response is blank")

    answers = output.get("turn_responses")
    if isinstance(answers, list):
        for index, answer in enumerate(answers[1:], start=2):
            if not str(answer or "").strip():
                failures.append(f"turn_{index}_response is blank")
    ...
    want_component = expectations.get("component")
    if want_component and want_component not in (output.get("turn_1_components") or []):
        failures.append(
            f"component: {want_component} missing from "
            f"{output.get('turn_1_components') or []}"
        )

    failures.extend(_dev_suite_failures(expectations, output))
    failures.extend(_artefact_failures(output))
    return (not failures), failures
```

Supporting checks: `_dev_suite_failures` (`:160-198`), `_artefact_failures` (`:81-85`),
`_citation_marker_failures` (`:215-265`, resolved against `turn_sources`).

## Judge prompt — `evaluation/platform/judge_call.py:47-105`

```python
    manifest = json.dumps(input_doc.get("patient_manifest") or {}, indent=2,
                          sort_keys=True)
    evidence = json.dumps(output.get("tool_evidence") or [], indent=2)[:120000]
    turns = input_doc.get("turns") or []
    answers = _turn_answers(output)

    parts = [MANIFEST_RULES + "\n" + manifest + "\n"]
    if not turns:
        return "\n".join(parts + ["NO QUESTION RECORDED\n", "EVIDENCE:\n" + evidence])
    for index, question in enumerate(turns):
        answer = answers[index] if index < len(answers) else ""
        parts.append(f"TURN {index + 1} — QUESTION:\n{question}\n\nANSWER:\n{answer}\n")
        cited = _cited_sources(output, index)
        if cited:
            parts.append(
                f"TURN {index + 1} — THE NUMBERING THIS ANSWER USES: every `^[k]` in the "
                "answer above resolves to the entry labelled [k] below, and to nothing "
                "else. A number with no entry here addresses no unit this turn published.\n"
                + cited + "\n")
    parts.append(
        "RAW TOOL EVIDENCE (what the tools returned, in the order they returned it).\n"
        "Each passage and each analyte carries the marker the answer must write for it:\n"
        "`\"cite\": \"^[10]\"` is the unit `^[10]` addresses. A number can be checked\n"
        "against this block directly, and a unit with no `cite` (an estimate, a medication\n"
        "list, a search result set) consumes none.\n" + evidence)
    return "\n".join(parts)
```

## Confirmed

- `canvas` is passed to **no** deterministic check and **no** judge prompt. The only
  canvas-aware assertion in the harness is component-**name** presence.
- `ComparativeRiskMatrix` and `LabTrendCard` are graded on nothing: not their rows, not
  the selected cohort, not the rate cells, not `baseline`, not `baseline_note`, not the
  phenotype that selected the rows.
- The matrix's **citations** do reach the judge, but only as generic numbered source lines
  through `turn_sources` → `_cited_sources` (`judge_call.py:96-120`). That makes the
  benchmark paper checkable as a source; it does not make the comparison checkable.
- `turn_1_card` holds the full turn-1 `a2ui` dict and is read by nothing in
  `evaluators.py` or `judge_call.py` — the cheapest existing hook for closing the gap.

---

# Structural notes for the next stage

1. `is_matrix_eligible` (`domain.py:302`) no longer inspects `hadm_id`: eligibility is
   exactly `is_outpatient is not True`. Every inpatient, including an unclassified one,
   draws the table.
2. The phenotype is resolved from note text by keyword match with `GENERAL_INPATIENT` as
   the fallback, so a wrong-condition table renders identically to a correct one and
   neither the trace nor the harness distinguishes them.
3. The matrix's figures are curated constants, not extracted values. Any claim that a
   live PubMed search produced them would be false; `search_literature` returns abstracts
   and no rate field.
