# Free-Text RAG & Vector Chunking Pipeline — Read-Only Audit

**Date:** 2026-11-03
**Scope:** `/ehr` has been repaired end to end; the free-text retrieval path has not been
audited since the doc-00 section rewrite. This is the read-only trace of what the vector
index can and cannot return, and of how the free-text prompt and post-processing compare
with the deterministic commands.
**Rule:** no files were modified. All findings are from reading source and from one
read-only measurement run against the deployed cohort.

---

## Summary

Three defects, in the order they cost recall:

1. **The index cannot represent a headingless note.** 3 of 87 admissions yield zero
   chunks and are unreachable by `rag_search` under any query.
2. **The whitelist omits narrative sections that carry the operation, the plan and the
   follow-up** — roughly 15k characters beyond the deliberately excluded lab noise.
3. **`SYSTEM_PROMPT` still describes the pre-doc-00 retrieval tool and the
   pre-`EHR_FORMATTING` labels**, so a free-text summary and `/ehr` disagree about the
   document they write and about its section headings.

---

## 1. Chunking and the index build

### Two files, two roles

`services/mcp/retrieval/chunking.py` is the chunker: `chunk_note` (:99),
`_chunk_section` (:123), `INDEX_SECTIONS` (:45).

`services/mcp/pipelines/components/chunk_notes.py` is the cloud build component, and it
is where the allowlist is applied — :90:

```python
for chunk in chunk_note(note, max_chars=DEFAULT_MAX_CHARS, pack_to=pack_to):
    if chunk.section not in whitelist:
        continue
```

`scripts/agent/build_chunks.py` is the local twin of the same step. The chunker itself
does **no** filtering; the whitelist is a build-time decision, and `INDEX_SECTIONS`
(13 names) is the single tuple both the build side and the serving-side datapoint-id
parser read.

### Preamble (unheaded text): never chunked

`chunk_note:115` iterates `parse_note(note["text"]).sections`, and a `Section` begins at
its heading match (`services/mcp/retrieval/sections.py:598`, `start=match.start()`). Text
before the first recognised heading belongs to no section, so it produces no chunk.

Measured over the deployed cohort: **0 of 87 notes** have such a preamble today, so this
is a latent path rather than a live leak — but `/ehr` does index it, as
`history_of_present_illness` (`services/mcp/tools/retrieval.py:650-654`).

### Headingless notes: invisible to the index

`parse_note` returns `sections=()` for a note with no recognised heading, so `chunk_note`
returns `[]`. Measured:

| admission | note chars | chars in the index |
|---|---|---|
| 90000026 | 1,153 | **0** |
| 90000051 | 1,447 | **0** |
| 90000073 | 1,758 | **0** |

3 of 87 notes — 4,358 characters of clinical narrative — cannot be returned by
`rag_search` under any query.

`_search_sections` handles exactly this case by emitting the whole note as one
`discharge_summary` passage (`retrieval.py:668-673`; the same rule in
`_ordered_sections:750-753`). The chunker has no equivalent, and `discharge_summary` is in
`INDEX_SECTIONS` even though nothing ever generates a section with that name.

### Duplicate section bodies

Ordinals are per section name (`chunk_note:113-118`), so repeated headings do not collide.
But `admission_diagnosis` is **not** in `INDEX_SECTIONS`, so only the discharge copy is
indexed — the index never sees the duplicate that `_search_sections` has to dedupe
(`seen_bodies`, `retrieval.py:619-631`; `_keep_final_diagnosis:545`). Not a correctness
problem in the index; it is a silent omission.

### Everything outside the whitelist

30,055 characters across 18 section names:

| section | chars | notes |
|---|---|---|
| pertinent_results | 7,094 | 24 |
| assessment_and_plan | 5,809 | 10 |
| major_procedure | 4,826 | 39 |
| followup_instructions | 3,742 | 20 |
| admission_diagnosis | 3,644 | 37 |
| review_of_systems | 1,824 | 4 |
| chief_complaint | 1,188 | 11 |
| consults | 553 | 9 |
| allergies | 505 | 11 |
| indications | 254 | 1 |
| past_surgical_history | 251 | 5 |
| complications | 105 | 7 |
| medications_on_transfer | 69 | 1 |
| pathology | 69 | 2 |
| service | 60 | 4 |
| transfusions | 41 | 2 |
| attending | 11 | 1 |
| infections | 10 | 2 |

`pertinent_results` is excluded on purpose (lab-line noise). The others are narrative the
free-text path can never retrieve: **39 notes carry a `major_procedure` section the index
does not hold**, and 10 carry an `assessment_and_plan`.

### Precomputed, then re-derived

The index is built offline (`rag_ingest_pipeline`: chunk → embed → `build_index`). At
serving, `_search` re-runs the chunker over the note fetched from BigQuery to recover each
chunk's exact text — `_chunk_texts_for:263`, using the same `DEFAULT_MAX_CHARS` /
`DEFAULT_PACK_TO` and keying on `chunk_id.replace(":", "_")`. No chunk-text table is read.

If the two ever disagree the passage degrades to whole-note text and is tagged
`granularity="note"` with a warning (`retrieval.py:410-425`).

---

## 2. Free-text retrieval: what filters a chunk

`rag_search(hadm_id, query, top_k)` (`retrieval.py:445`) takes **no section or intent
argument**. Nothing filters chunks by section at query time: the index ranks top-k by
distance under the `hadm_id` restrict (`_search:305-320`), and every hit is returned.

Section name enters the free-text path in exactly three places, none of which drops a hit:

1. **Id parsing.** `_KNOWN_SECTIONS = INDEX_SECTIONS` (:93), `_SECTION_RE` (:94),
   `_parse_datapoint_id` (:198). An id whose section token is not in the tuple is a hard
   `unparsed_datapoint` error (:396-408) rather than a dropped passage — so build and
   serving must move together, which is why the tuple lives in one place.
2. **The anchored retry.** `_section_for_query` (:515-533) maps six keyword groups
   (`medication`, `discharge diagnosis`, `instruction`, `hospital course`…) to a section,
   and `_search` retries **once** with that section's body as the query when the intended
   section did not rank first (:343-352) or when nothing came back (:354-368). It can add
   recall; it cannot remove a passage.
3. **Presentation only.** `intent_sections` never touches retrieval.
   `a2ui.resolve_sources:985` applies it solely to a single-citation answer —
   `intent = sections if len(cites) == 1 else ()` — and `resolve_source` treats it as a
   preference, never a veto.

So a `brief_hospital_course` chunk is reachable, and `brief_hospital_course` is in the
whitelist. The losses are upstream of retrieval, in the build: the sections in the table
above, and the three headingless notes.

`_freetext_allowed` (`services/agent/engine/graph.py:357`) does not touch the retrieval
tools — `rag_search` and `rag_search_sections` are unconditional members of
`FREETEXT_TOOLS` (`services/agent/routing/routing.py:312-333`).

---

## 3. `SYSTEM_PROMPT` against the current tool and the current label set

Three stale passages, all in the SUMMARIZATION block.

### Lines 339-346 — the four-section contract

> it returns one passage per summary section (history of present illness, hospital course,
> discharge diagnosis, discharge instructions) merged in that order. It does NOT return
> the discharge medications section

Both halves are false now. `_search_sections` returns every section whole in document
order — 9 on 90000015, including `discharge_medications` — and its docstring
(`retrieval.py:571-595`) records that the four-section version was the defect.

### Lines 347-353 — the same claim, restated

> ANY question that clearly targets one of the fixed discharge-note sections … should call
> `rag_search_sections` ONCE … The discharge medications section is the one exception:
> `rag_search_sections` does not return it

This sends a free-text medication question away from a tool that now answers it.

### Lines 363-377 — the legacy labels

```
  **Clinical course.** The patient was seen …

  **Discharge diagnosis.** The patient was diagnosed with …
```

and the rule that `**Hospital course**` is the inpatient label, `**Clinical course**` the
outpatient one.

`EHR_FORMATTING` (`prompts.py:756+`) now specifies four different labels —
`**Presentation & diagnosis.**`, `**Hospital course & procedures.**`,
`**Discharge plan & medications.**`, `**Readmission risk & trajectory.**` — and none of
them appears in `SYSTEM_PROMPT`. A free-text "summarize this admission" and `/ehr`
therefore produce different section headings for the same question:
`**Discharge diagnosis.**` versus `**Presentation & diagnosis.**`, with no
`**Readmission risk & trajectory.**` paragraph at all on the free-text route.

### What is not in conflict

Citation grammar is consistent between the two prompts. `SYSTEM_PROMPT:454-456` teaches
"copy the `cite_header`, write nothing around it", and `citation_header`
(`services/agent/postprocess/citations.py:1340+`) prints exactly `^[6]`. Lines 388-390's
"cite each passage in the order it appears in the returned list" still holds, because that
order is now the document's.

What `SYSTEM_PROMPT` lacks is any instruction for the `^[PMID: …]` badge — it teaches only
`^[n]`. That is harmless while a free-text literature turn's records get numbered
positions, but it means the badge form is a `/ehr`-only contract.

---

## 4. Free-text post-processing

**Same pipeline, and no divergence in the citation rules.** Both routes settle and then
guard:

- command: `graph.ask` → `_settle(command_state, prose=final_text(...))` at
  `graph.py:981`
- free text: `graph.ask` → `_settle(state, prose=final_text(state))` at `graph.py:1185`

Both then return to `turn.compose_success`, which runs `guard_answer` (settlement's
`enforce_citations` again, with the same allowance, at
`services/agent/postprocess/guardrail.py:1030`) and `compose_presentation`.

One implementation of `strip_printed_constructs` → `drop_unknown_markers` →
`enforce_citations` → `collapse_repeated_citations` → `compact_display_numbers`, reached
identically.

Five differences, none of them a citation rule:

1. The free-text branch runs `read_evidence_reply` and `no_evidence_answer` first
   (`graph.py:1151-1174`). A declined turn sets `deterministic_answer` to the fixed
   abstention sentence, and settlement then settles *that* string instead of the model's
   prose. Commands skip the envelope entirely.
2. `comparators` is empty on free text, so no comparator allowance — correct by
   construction, since no table is drawn.
3. `literature_replay = bool(wants_literature)` can be True on a free-text follow-up,
   widening the PMID allowance from the replay. Available on both routes; normally False
   for commands.
4. `command=None` at `compose_presentation`, so the lab rule does not empty
   `shown_sources` and no matrix is composed.
5. `deterministic_answer` versus the model message only decides *which string* is settled.

---

## Measurement method

`parse_note` and `chunk_note` were run over all 87 rows of
`readmission.hybrid_notes` with the production parameters
(`DEFAULT_MAX_CHARS`, `DEFAULT_PACK_TO`), counting section names against
`INDEX_SECTIONS`. The query was a `SELECT`; nothing was written. The script is
throwaway and lives outside the repository.
