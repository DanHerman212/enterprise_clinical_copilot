# Turn-3 Remediation — Lab Scoping, Display Compaction, Trace, Feedback, Evidence Kind

Thread: patient 90000023, `agent-00225-q67`, 2026-10-31 15:27Z — `/ehr`, then two free-text
follow-ups. Five defects, five fixes, each in the layer the charters assign the decision to.
No new owner, no second vocabulary, no heuristic in a layer that does not own one.

## Gates

| Command | Result |
|---|---|
| `pytest test_layer_separation.py test_architecture_remediation_contracts.py test_freetext_boundary.py` | **76 passed** (27 + 11 + 38) |
| `pytest test_cite_offset.py test_freetext_replayed_citations.py` | **8 passed** |
| `pytest tests -m "not live"` | **1642 passed**, 1 skipped, 43 deselected |
| `pytest tests/evaluation` | **197 passed** |
| `python -m evaluation.platform.dry_run` | **STEP 7 PASSED — 100%** |

## 1. Display compaction — `postprocess/citations.py`, applied in `postprocess/answer.py`

`citations.compact_display_numbers` renumbers an answer's markers to the order it cites and
returns `{display_number: array_position}`. `answer.sanitize_answer_markers` applies it at
settlement — after enforcement, before the trace is written — and records the map on the
state; `turn.py` hands it to `compose_presentation`, which resolves `sources` through it.

Measured on the thread's shape: `^[14], ^[15]` in, `^[1], ^[2]` served, map
`{"1": 14, "2": 15}`, and the two sources still resolve to the analyte and the passage they
always did.

The array keeps its cumulative numbering, because that is what makes a follow-up citable at
all (`test_cite_offset.py` locks it, and still passes untouched). Law 1 holds: the rewrite
is the citation owner's, in `postprocess`; presentation only resolves through the map it is
given.

## 2. Trace completeness — `telemetry/observability.py`

`record_turn_tools` writes this turn's calls to the root span's metadata — name, arguments,
returned count or error, never the payload — called from `turn.py` beside `set_trace_io`.
Free-text turns open no tool spans by design (a span opened there becomes its own root), so
until now the trace row for turn 3 was prose citing `^[14]` with nothing to say a laboratory
panel had been read first. The trace's `output` is still the served prose, unchanged (Law 5);
langfuse 4.14.4 has no trace-level metadata writer, which is why the root span carries it.

## 3. Card scoping — `presentation/a2ui.py`, vocabulary in `shared/labels.py`

`labs` is built from this turn's calls **and only when the turn asked for it**: `command ==
"labs"`, or a question that names an analyte. The tool stays bound — `prompts.py:232-239`
prefers a structured series for every quantitative value, so withholding it would trade a
surface defect for a worse answer.

The analyte vocabulary moved from `routing/` to `shared/labels.py`, because presentation may
not import routing (Charter 13 §5) and a second copy is the duplication Law 4 forbids.
`routing.asks_about_a_lab_analyte` now asks the shared map, so the routing decision and the
surface cannot disagree about whether a question named a measurement.

## 4. Feedback for a withdrawn tool — `engine/prompts.py` + `engine/graph.py`

`prompts.TRANSCRIPT_NOTICE` says once, in the first model round, what the transcript already
holds and that those tools will not run again; `graph.agent_node` builds it from
`allowed - _still_bound(allowed, state)`, naming units with `routing.RESULT_NOUNS`. The
wording lives in `prompts.py` (Law 3); the injection is the engine's own orchestration, the
same mechanism `demand_retrieval_node` already uses. Turn 2 of the thread spent four model
rounds on one executed call, three of them asking for tools the caps had already withdrawn.

## 5. Wrong-kind evidence — `postprocess/citations.py`, raised in `guardrail.py`

`citations.evidence_kind_mismatches` flags a sentence whose markers all resolve to laboratory
series while the sentence claims to be reporting the note. Raised on the production sentence
verbatim: `citation_kind_mismatch:^1`, `citation_kind_mismatch:^2`, prose untouched.

Reported, not rewritten. `enforce_citations` can check that a marker resolves; it cannot
check that it resolves to the right kind, because a note passage and an analyte are both in
range. Replacing the sentence needs a claim classifier this system does not have, and the
flag is what makes the defect visible to the reader and countable by the harness.

## Contracts added to `test_architecture_remediation_contracts.py`

- `test_settlement_compacts_the_markers_while_the_array_keeps_its_positions` — both halves:
  the served text and the map. Compaction without the map would point every pill at the
  wrong passage.
- `test_a_free_text_turn_that_read_the_panel_does_not_draw_it` — the absence, plus the
  analyte-question baseline that must survive it.

## Residual trade-off, stated

Compaction means the stored turn's markers are display numbers while the replayed tool
results keep array numbers. A follow-up that copies a marker out of the previous answer
rather than out of the tool result would now read the wrong unit. The prompt tells the model
to copy from the tool result contract (`prompts.py:414`), which is why this is acceptable —
but it is the cost of Law 5, and it is the reason deduplication was not attempted: two
numbering systems already exist by design, and a third would not help.

**Not done, deliberately:** ledger deduplication (no stable unit identity — `(section, text)`
shifts with chunk boundaries) and unbinding `get_lab_trends` on free-text turns (contradicts
the prompt's structured-data rule). Both were the hacks.

## Files

`services/agent/postprocess/citations.py`, `postprocess/answer.py`, `postprocess/guardrail.py`,
`presentation/a2ui.py`, `engine/graph.py`, `engine/prompts.py`, `telemetry/observability.py`,
`turn.py`, `routing/routing.py`, `shared/labels.py`,
`tests/agent/test_architecture_remediation_contracts.py`.
