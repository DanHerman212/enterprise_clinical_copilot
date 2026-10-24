---

## Section 5: End-to-End Multi-Turn Follow-Up Audit & Answerability Remediation

### 5.1 Problem Statement
Manual multi-turn UI testing across patients exposed two remaining gaps:
1. **Inconsistent Multi-Turn De-Duplication in the UI:** For at least one patient flow in the browser UI, clicking a follow-up chip did not remove or advance the follow-up suggestions as expected.
2. **Unhelpful Turn-2 Outputs on Clicked Chips:** Clicking certain generated follow-up chips on other patients produced unhelpful or incomplete Turn-2 responses (e.g., when a chip asks about precautions or follow-up appointments after `/meds`, or when the Turn-2 ReAct loop fails to retrieve or synthesize the passage containing the answer).

### 5.2 Step 1: Audit Both Failure Modes Systematically (Before Further UI Testing)

1. **Audit the Browser-to-Backend Wire Contract for Chip Clicks (Zero LLM Calls):**
   - Inspect the frontend code (`services/web/` JS/template where follow-up chips are clicked and sent to `/ask` or `/ask/stream`) and `services/agent/http.py`.
   - Verify what payload the browser sends when a follow-up chip is clicked (specifically how `turns` / conversation history and `follow_ups: true` are serialized) and verify whether `http.py` passes the complete prior user questions into `followups.suggest()`.
   - Simulate a 3-turn progression (`Turn 1: /ehr` $\rightarrow$ `Turn 2: click follow_ups[0]` $\rightarrow$ `Turn 3: click follow_ups[0]`) and (`Turn 1: /meds` $\rightarrow$ `Turn 2: click follow_ups[0]` $\rightarrow$ `Turn 3: click follow_ups[0]`) across all 5 inpatient Dev archetypes (`90000015`, `90000108`, `90000106`, `90000105`, `90000023`).

2. **Audit Turn-2 Answer Quality for Every Generated Chip across the Dev Cohort:**
   - For every unique `(hadm_id, turn_1_command, chip)` tuple generated in the Dev Cohort Matrix (`90000015`, `90000108`, `90000106`, `90000105`, `90000023`), execute the Turn-2 call and produce an **End-to-End Chip Answerability Table**:
     - Columns: `Record` | `Turn 1 Cmd` | `Chip Question` | `Turn 2 Tools Used (Replayed + Fresh)` | `Turn 2 Answer Excerpt` | `Citations Verified?` | `Quality Verdict (GOOD / BAD + Reason)`
   - Flag every case where:
     - The Turn-2 answer says the information is not documented / unavailable,
     - The Turn-2 answer fails to answer the specific question asked by the chip,
     - `verified` is `False` or citations (`^[n]`) are missing, or
     - Turn 2's returned `follow_ups` repeats the clicked chip or a prior turn's question.

### 5.3 Step 2: Remediate All Flagged Trigger & Answerability Defects
For every `BAD` row identified in Step 1:
1. **False-Positive Chip Trigger in `followups.py`:** If a chip fired because a weak regex matched a word (e.g., `"Dr."`, `"follow"`, `"return"`, `"diet"`, `"activity"`) when the chart does not actually contain a substantive answer to that question, tighten the extractor in `followups.py` so the template only fires when a verifiable clinical instruction/entity is present.
2. **Turn-2 Context / Tool Gap (`http.py` / `graph.py` / `routing.py`):** If the answer *does* exist in the patient's chart (or in Turn 1's replayed payload) but Turn 2 failed to retrieve or synthesize it properly, fix the Turn-2 routing/prompt/replay handling so the ReAct agent answers the chip directly with verified `^[n]` citations.
3. **Frontend / Wire History Gap:** If the browser UI or `http.py` was dropping prior turn questions or `replayed_tool_calls` across chip clicks, fix the wire handling and pin it with a test.

### 5.4 Deliverable & Stop Gate
1. Present the **End-to-End Chip Answerability Table** (before and after fixes) covering all generated chips across the Dev cohort, plus the 3-turn progression trace proving zero repetition.
2. Deploy the verified fix to Cloud Run and **STOP** for a quick Architect UI spot-check before running Gate 1 Canary (Run 5) and Gate 2.