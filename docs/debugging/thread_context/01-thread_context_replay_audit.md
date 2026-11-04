# Multi-Turn Conversation Replay & Thread Context on Free-Text Turns

**Date:** 2026-11-04
**Scope:** a free-text follow-up to an `/ehr` turn cannot see or cite the thread's
readmission comparators. This is the read-only trace of what the replay carries, how the
follow-up is classified, and what it is allowed to cite.
**Rule:** no files modified. The measurements below come from a probe that replays the wire
exactly as the BFF sends it.

---

## 1. What the BFF sends, and what `ingress/` accepts

**The payload, per turn** — `demo/conversations.py:339` `replay_turns`:

```python
turns.append({
    'question': asked,
    'answer': _clamp(row.answer),
    'tool_calls': _replay_calls(row.tool_calls),
})
```

Three fields, and the contract refuses anything else —
`services/agent/ingress/contracts.py:180`:

```python
REPLAYED_TURN_FIELDS = frozenset({"question", "answer", "tool_calls"})
REPLAYED_TOOL_CALL_FIELDS = frozenset({"name", "args", "payload"})
```

`_parse_turns` (:256) raises `unsupported_field` for a fourth key, so **there is no channel
for `comparators`, `sources` or `references`.** Bounds: 6 turns (the window is the anchor
turn plus the last five, `conversations.py:375`), 5 tool calls per turn, answers clamped at
8,000 chars with a visible marker.

**A tool call's payload is usually absent.** `_stored_calls` keeps `payload` only when the
agent marks the call non-derivable, and
`RE_DERIVABLE_TOOLS = routing.PASSAGE_TOOLS | routing.LITERATURE_TOOLS`
(`contracts.py:93`). So `rag_search`, `rag_search_sections` **and `search_literature`** are
stored as `{name, args, payload: None}`. Only
`RESULT_KEPT_TOOLS = {"predict_readmission", "get_lab_trends"}` (`routing.py:892`) carry a
payload.

**Parsing into state** — `services/agent/engine/graph.py`:

- `ask()` computes `wants_literature = turn_is_about_literature(question, question_kind)`
  (:1015), then `replayed, replayed_calls = await replay_messages(toolbox, turns,
  keep_literature=wants_literature)` (:1027).
- State is seeded at :1105 with `messages = [SystemMessage, *replayed, HumanMessage]`,
  `replayed_tool_calls = replayed_calls`, `wants_literature`, `retrieval_demanded`.
- `replay_messages` (:1253) emits, per turn, `HumanMessage(question)` → one tool-result
  message per call → `AIMessage(answer)`, and returns `calls` in the live
  `{name, args, response}` shape. A call whose payload is absent is **re-executed**
  (`await toolbox.call(...)`), which is why passage text never reaches the store.
- `state["comparators"]` is never seeded by `ask()`. It is set only by the command path,
  `deterministic.run_command` (`deterministic.py:526`, `:630`).

---

## 2. Why the follow-up loses `/ehr`'s comparators

Probe output, for `"how does the peer reviewed medical research for readmission risk
compare to the risk model?"` against a replayed `/ehr` turn:

### a) The prose arrives — yes

`replay_messages:1327` appends `AIMessage(content=turn["answer"])`. Measured:
`assistant prose carries the badge: True`. The model reads its own
`**Readmission risk & trajectory.**` paragraph with `^[PMID: 24108298]` in it.

### b) The comparators do not arrive — no

```
keep_literature=False  replayed_calls=['rag_search_sections', 'predict_readmission']
keep_literature=True   replayed_calls=['rag_search_sections', 'predict_readmission',
                                       'search_literature']
```

Two independent filters:

- `replay_messages:1290` — `if not keep_literature and name in routing.LITERATURE_TOOLS:
  continue`. On a turn not classified as literature, the `/ehr` comparator search is
  **deleted from the transcript**, messages and evidence both. The comment records why: a
  stale `search_literature` result once bound `^[1]` to a World Neurosurgery article on a
  `/meds` follow-up (90000058).
- Even when kept, it is not citable evidence. `_searched_for_evidence`
  (`citations.py:1130`) requires `args.scope == "general"`, and the comparator retrieval
  declares `scope="readmission"`:

  ```
  literature_pmids(replayed, replayed)  : []
  ```

- The verified rows themselves — what `comparator_pmids` exists to allow — are
  `state["comparators"]`, populated only by the `/ehr` command run. Nothing carries them
  across turns.

### c) The classification, and the forced note search

`asks_for_literature` (`routing.py:355`) matches only this cue set (`:344`): `literature,
evidence, study, studies, trial, trials, meta analysis, metaanalysis, systematic review,
pubmed, guideline, guidelines, efficacy, effectiveness, benchmark, benchmarks`.

Measured on the reported question: **False** — "peer reviewed research" and "compare … to
the risk model" hit no cue.

Three consequences, in order:

1. `_freetext_allowed` (`graph.py:409`) returns `FREETEXT_TOOLS` instead of
   `LITERATURE_QUESTION_TOOLS`, so `search_literature` is **not bound**. The model cannot
   consult PubMed even if it wanted to.
2. `keep_literature=False` → the `/ehr` search is dropped from the replay (b above).
3. `route()` falls through to `demand_retrieval` (`graph.py:~745`) because the answer calls
   no tool, no retrieval was demanded, the reply is not a command redirect, there is no
   manifest, and `risk_driver_declared` / `follow_up_declared` are False. That is the
   forced `rag_search` observed live, and "the discharge notes do not document peer-reviewed
   research" is the model correctly reporting on the only evidence it was left with.

**Why `follow_up_declared` was False:** only a clicked suggestion chip sends it —
`static/js/copilot_turn_facts.js:231` is the only `intent: 'follow_up'` in the client. A
typed question in an existing thread declares no intent, so the agent treats it as a first
turn about the chart.

One nuance: `TRANSCRIPT_NOTICE` (`prompts.py:1047`) is emitted only for tools that were in
`allowed` and were then withdrawn. `search_literature` was never in `allowed`, so the model
was never told that published records existed in the thread — and on this path they did not.

---

## 3. Citation and grounding allowance on a replayed turn

The allowance is two channels, both from `settle_answer` (`answer.py:343-363`):

```python
bound = citations.citation_bound(tool_calls, replayed_tool_calls)
pmids = citations.literature_pmids(tool_calls,
            replayed_tool_calls if wants_literature else None) \
        | citations.comparator_pmids(comparators)
```

Measured, with Turn 1's replay in hand and the marker copied from Turn 1's prose:

```
wants_literature=False -> 'The published benchmark ... reports 2.5%.'
  flags: ('citation_out_of_range:^24108298',)
wants_literature=True  -> 'The published benchmark ... reports 2.5%.'
  flags: ('citation_out_of_range:^24108298',)

citation_bound([], replayed)         : 1     (note units carry over correctly)
comparator_pmids([])                 : []    (a free-text turn holds none)
```

So **the badge dies either way**, and `guard_answer` (`guardrail.py:1030`) cannot revive it:
it runs on text settlement has already cleaned, and it builds the same two channels.

Note passages are fine. `compose_presentation` resolves against
`evidence=[*replayed_tool_calls, *tool_calls]` (`turn.py:301`), so a follow-up that cites a
replayed section gets a real pill and card, and the numbering continues across turns. The
asymmetry is that a *replayed literature* record has no equivalent path:
`literature_payload(tool_calls, evidence if literature_replay else None)` yields articles
only when `literature_replay` is true, which is the same `wants_literature` flag.

---

## 4. The gaps, and the minimal path

Four gaps, each independently sufficient to produce the observed behaviour:

| # | Gap | Where |
|---|---|---|
| 1 | No transport channel for the verified cohort rows; the replay contract is closed at three fields | `contracts.py:180`, `conversations.py:339` |
| 2 | A comparator search is dropped from the replay on a non-literature turn, and is never citable evidence even when kept | `graph.py:1290`, `citations.py:1130` |
| 3 | The literature cue set misses ordinary phrasings ("peer reviewed", "research", "published") | `routing.py:344` |
| 4 | A typed follow-up declares no intent, so the router forces a note retrieval | `graph.py:~745` vs `copilot_turn_facts.js:231` |

**Minimal change, in the order that buys the most for the least:**

**(a) One line on the cue set** — `("peer", "reviewed")`, `("research",)`, `("published",)`.
This reclassifies the reported question as a literature turn, which binds
`search_literature` and keeps the `/ehr` search in the replay. It fixes the answer *and* the
wrong tool. It does not by itself let the model cite Turn 1's PMIDs.

**(b) Let the console declare provenance** — send `intent: "follow_up"` for any typed
message in an existing thread, not only for a clicked chip. Then `route()` returns `END`
instead of forcing a note search, and the model answers from a transcript that already
contains the rates. One line in the client, no agent change. This is the cheapest fix for
the symptom as experienced.

**(c) Carry the comparators** — the only change that makes Turn 2 able to cite Turn 1's
cohorts. The data is already stored client-side: the a2ui envelope's
`ComparativeRiskMatrix` rows are in `Turn.presentation`. Add `"comparators"` to
`REPLAYED_TURN_FIELDS`, store the row subset (`pmid`, `reported_rate`, `journal`, `year`,
`title`, `excerpt`) on the `Turn`, and in `graph.ask` union the replayed rows into
`state["comparators"]`.

Everything downstream already takes comparators as plain data and needs no change:
`settle_answer`, `guard_answer` and `compose_presentation` all accept them, and the
comparison table will **not** be redrawn on the follow-up because `_draws_matrix`
(`canvas.py:168`) requires `command == "summarize"` while a free-text turn passes
`command=None`. So the rows widen the citation allowance without adding a second table to
the column — which is the property required.

(c) subsumes the `_searched_for_evidence` question: the allowance should be the *verified*
rows, never the 25-candidate pool, which is the rule `comparator_pmids`' own docstring sets.
