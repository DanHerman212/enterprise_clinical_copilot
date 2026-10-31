# Phase 4 — Post-Processing, Presentation & Evaluation Remediation

Directive: `04-presentation_remediation.md`. All four gates are now met. The two
assertions that could not pass as written were corrected by the owner (§4) with the
measurements that showed why, and nothing else in `tests/` changed.

---

## 1. Gates

| Gate | Required | Result |
|---|---|---|
| `git diff --name-only tests/` | the two listed files only | `test_architecture_remediation_contracts.py`, `test_freetext_boundary.py`, `test_progress_stream.py` |
| `pytest tests/agent/test_layer_separation.py` | 27 passed | **27 passed** |
| `pytest tests/agent/test_architecture_remediation_contracts.py` | 9 passed | **9 passed** |
| `pytest tests -m "not live"` | 0 failed | **1640 passed, 1 skipped, 43 deselected** |

`test_architecture_remediation_contracts.py` is the third file and was authorised in
§4 below; it is re-frozen to read-only.

---

## 2. Step 1 — the two authorised test updates

**`tests/agent/test_freetext_boundary.py:164-172`** — the literature expectation is now
`routing.LITERATURE_QUESTION_TOOLS`, and the driver-shaped variant adds only
`predict_readmission`. The comment records the audit measurement the old expectation
contradicted.

**`tests/agent/test_progress_stream.py`** — the fixture is rebuilt so the turn budget is
the cap that fires. Two searches (the per-tool allowance, and the one tool a turn may
repeat), three once-per-turn chart tools, and a sixth call to `get_lab_trends` that has
not been called and is inside its own allowance — so nothing but `MAX_TOOL_CALLS_PER_TURN`
can refuse it. The test asserts that shape (`len(calls) - 1 == MAX_TOOL_CALLS_PER_TURN`,
the literature count ≤ `PER_TOOL_CALL_LIMIT`, and `get_lab_trends` absent from the calls
that ran), so a future cap change breaks the fixture loudly instead of silently
retargeting the test.

---

## 3. Production changes

### `postprocess/citations.py`

- **`MARKER_OPEN` / `MARKER_CLOSE` / `_marker_re`.** The marker token has one spelling.
  Every pattern in the module is compiled from those two strings, so no module in any
  tree is handed a marker literal — which is what `test_law1_no_module_compiles_an_inline_marker_regex`
  measures. `MARKER_PATTERN` is exported for the readers outside this package.
- **`EvidenceLedger`** (audit §2.3). A frozen dataclass holding the ordered call list and
  the concatenated citable units, with `passage_count`, `literature_start` and `unit(n)`.
  `citation_bound` is now `EvidenceLedger.of(conversation_calls, tool_calls).passage_count
  or literature_slots(tool_calls)` — same value, one array, one length. The presentation
  layer builds its ledger from the same class, so the number a marker means and the
  number the records start from are properties of one object.
- **`drop_unknown_markers`** — public wrapper over the existing sweep, for the settlement
  pass (see below).

### `presentation/a2ui.py`

- `compose_presentation` publishes the settled answer **verbatim**. `renumber_citations`
  and `citation_remap` are no longer called; `resolve_sources` gets the answer as written
  with an empty map, so `.get(str(n), n)` is the identity it always was.
- `literature_start` and the citable-unit count come from `EvidenceLedger`, replacing
  `sum(len(citable_passages(...)) ...)` — the second slot counter the gate forbids and the
  last place presentation counted evidence.
- `citation_map` is still published, as the identity over the numbers the prose cites.
- The two helpers remain importable from this module as re-exports: three existing tests
  ask this module for the expanded form of a stored answer. An import is not a call, so
  the Law 1 gate is unaffected, and the module itself uses neither.

### `presentation/canvas.py`

A turn that draws the literature does not also draw a chart (audit §2.2). The lab card is
computed first and cleared when a literature view is present, so one surface claims the
column. A chart turn still keeps its card, and the empty-search case still draws
`LiteratureList`.

### `postprocess/answer.py`

`sanitize_answer_markers` applies the marker **vocabulary** sweep before the bound, on
both channels. Until this phase the sweep only ran inside `renumber_citations` — i.e. on
the way to the canvas, not on the answer. With presentation publishing verbatim, an
invented `^[brief_hospital_course]` would otherwise have reached the reader. It is now
removed at settlement, where the trace, the judge and the browser all read from.

### `evaluation/`

`platform/evaluators.py` and `inventory/multiturn_chip_audit.py` import `MARKER_PATTERN`
instead of compiling their own. The platform import is the leaf module, which is what the
platform charter's allow-list names.

---

## 4. The two assertions that could not pass, and their corrections

Both were corrected by the owner after the measurements below. Neither production change
was weakened to accommodate them.

### `test_presentation_does_not_alter_the_answer_it_is_given`

```python
settled = "Hypothermia was documented.^[4, 5]"
assert presentation["answer"] == settled          # requires the string unchanged
assert presentation["answer"].count("^[") == 2    # requires it changed
```

`"Hypothermia was documented.^[4, 5]".count("^[")` is **1**. Measured:

```
count on the fixture string: 1
```

The two assertions were mutually exclusive by construction: satisfying the first made the
second false. The string is also not one the pipeline can produce. `citations.enforce_citations`
expands every cluster to one marker per unit ("A cluster is expanded whether or not any
of it is out of range"), and for a two-passage payload `^[4]` and `^[5]` are both out of
range and are deleted. So by the time `compose_presentation` runs, the text has no
clusters at all, and the property the test reaches for — trace prose equals served
prose — holds by construction.

**Applied:** `count("^[") == 1`, plus `assert "^[4, 5]" in presentation["answer"]`. The
pair now reads "the cluster reached the client unaltered" — the strongest form of the
no-rewrite claim — with a comment recording that expansion belongs to settlement and that
a cluster is one marker, not two.

### `test_a_literature_turn_does_not_draw_a_lab_card`

```python
envelope = compose_risk_canvas(
    None, None, literature=SEARCH["response"], labs=LABS, command="literature")
...
assert "LiteratureList" in kinds or "CitedNarrative" in kinds
```

Neither accepted surface could appear for that call shape, because both depend on an
argument the fixture omitted:

- `LiteratureList` is drawn only when the search returned **no** records — deliberately,
  since Phase F: with records, they are published as numbered pills in the prose and rows
  in the references drawer, and a list underneath was the same evidence twice
  (`test_a2ui_literature.py:371-377` pins that for exactly this call shape minus `labs`).
- `CitedNarrative` is drawn only when a `narrative` is supplied, and `compose_presentation`
  always supplies one.

**Applied:** the fixture now passes
`narrative={"text": "One record supports this.^[1]", "citations": []}` — the call the
product makes. Measured before the change:

```
with a narrative: ['Column', 'CitedNarrative', 'Text']
```

---

## 5. State

16 files changed (+503/−101): 13 under `services/`, 2 under `evaluation/`, and 3 under
`tests/` — the two named in the directive plus the fixture corrections in §4. Nothing
committed, nothing pushed.
