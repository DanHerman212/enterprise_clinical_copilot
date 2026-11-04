# Engineering Directive: Full Thread Evidence Continuity & Grounding Across Multi-Turn Conversations

## 0. Governance & Strict Boundaries
1. **Strict File Allowlist:** You may ONLY modify the following production files (plus unit test files under `tests/`):
   - `services/agent/engine/graph.py`
   - `services/agent/postprocess/citations.py`
   - `services/agent/postprocess/answer.py`
   - `services/agent/postprocess/guardrail.py`
   - `services/agent/presentation/a2ui.py`
   - `services/agent/turn.py`
   Run `git status` and `git diff --name-only` before finishing to verify no other production files were touched.
2. **Architectural Invariants:**
   - Do NOT modify `ingress/contracts.py` or the Django BFF repo (`danielmherman`). The wire contract (`REPLAYED_TURN_FIELDS = {"question", "answer", "tool_calls"}`) must remain untouched.
   - Do NOT add keyword hacks to `routing.py` (`asks_for_literature`). Thread continuity must work globally regardless of how the user phrases a follow-up question.
   - Ensure `tests/agent/test_layer_separation.py` passes unmodified.

---

## Step 1: Unconditionally Replay All Prior Tool Calls in the Thread (`services/agent/engine/graph.py`)
When a conversation is underway, the agent must have visibility into every tool result that was executed earlier in the thread.
1. In `replay_messages` (`services/agent/engine/graph.py`), remove the `if not keep_literature and name in routing.LITERATURE_TOOLS: continue` filter so that all prior tool calls (`rag_search`, `rag_search_sections`, `predict_readmission`, `get_lab_trends`, `search_literature`) are replayed into the transcript and returned in `replayed_tool_calls`.
2. Ensure that replaying `search_literature` does not collide with `^[1..N]` clinical note passage numbering (`citation_bound` only counts passage tools; literature articles in `/ehr` and follow-up turns are addressed by `^[PMID: <pmid>]` or resolved via `literature_payload`).

---

## Step 2: Keep Thread-Verified Literature & Comparators Citable Across Turns (`citations.py`, `answer.py`, `guardrail.py`, `a2ui.py`, `turn.py`)
Any PubMed article that was retrieved and cited in a prior turn (or retrieved in a replayed `search_literature` call) must remain citable and resolvable on follow-up turns without requiring `wants_literature=True` or `scope="general"`:
1. **Extract Thread-Verified Comparator / Literature Records:**
   - When replaying turns in `graph.py` (or resolving replayed evidence in `citations.py` / `a2ui.py`), any article from a replayed `search_literature` call (including `scope="readmission"`) whose PMID was cited (`^[PMID: <pmid>]`) in a prior turn's assistant `answer` (or returned in a replayed `scope="general"` literature search) is verified thread evidence.
   - Populate/reconstruct `comparators` (or the allowed replayed PMIDs and literature source records) from the replayed `turns` (`turn["answer"]` badges matched against replayed `search_literature` articles) so that:
     a) `settle_answer` (`answer.py`) and `guard_answer` (`guardrail.py`) permit `^[PMID: <pmid>]` on follow-up turns without stripping it as `citation_out_of_range`.
     b) `compose_presentation` (`a2ui.py` / `turn.py`) resolves the cited `^[PMID: <pmid>]` marker into a valid inline literature pill and `REFERENCES` drawer entry on the follow-up turn (while keeping `_draws_matrix` scoped strictly to `/ehr` so the `ComparativeRiskMatrix` table is not redrawn on free-text follow-ups).

---

## Step 3: Do Not Force a Redundant Chart Search When Thread Evidence Already Exists (`services/agent/engine/graph.py`)
In `route()` (`services/agent/engine/graph.py:~745`), a typed follow-up question in an existing thread currently falls through to `demand_retrieval` (forcing a `rag_search` against the discharge note) whenever the model answers directly from the replayed conversation history without calling a new tool.
1. Update `route()` so that when prior thread evidence already exists (`bool(state.get("replayed_tool_calls"))` is True) and the model produces a direct grounded answer from the replayed thread transcript without calling a new tool, `route()` does NOT force `demand_retrieval` (matching the behavior when `follow_up_declared` is True).
2. The model remains completely free to call `rag_search`, `rag_search_sections`, `get_lab_trends`, `predict_readmission`, or `search_literature` whenever the follow-up question requires new evidence not yet in the thread.

---

## Step 4: Verification (`tests/`)
1. Add unit and integration tests in `tests/` verifying that:
   - After an `/ehr` turn (with `rag_search_sections`, `predict_readmission`, and `search_literature(scope="readmission")` in `turns`), a typed free-text follow-up asking `"how does the peer reviewed medical research for readmission risk compare to the risk model?"` (with `intent=None`):
     a) Sees both the replayed `predict_readmission` payload and the replayed `search_literature` comparator articles in the transcript.
     b) Is not forced into `demand_retrieval` if it answers from the replayed thread evidence.
     c) Preserves `^[PMID: ...]` in `settle_answer` and `guard_answer` with zero `citation_out_of_range` flags.
     d) Resolves the `^[PMID: ...]` marker in `compose_presentation` into a valid literature pill and reference card without redrawing the `ComparativeRiskMatrix` table.
   - Replaying a prior turn with `search_literature` does not shift or corrupt `^[1..N]` note passage citations on a subsequent `/meds` or chart follow-up turn.
2. Run `python -m pytest tests -m "not live" -q` and ensure 100% pass.