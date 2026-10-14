# Phase 3: Session Continuity & Layer 8 Memory Policy Revision

## Executive Summary
This phase aligns the Layer 8 Memory & Session State architecture with the multi-patient A2UI interface. The objectives are to verify and enforce vertical canvas appending, enable full visual state persistence so users can seamlessly switch between parallel conversations and pick up where they left off, update the retention and turn limits to match clinical shift workflows, and wire the sidebar navigation controls.

## 1. Live Verification: Canvas Appending & Structural Replay
**Objective:** Establish ground truth on the live endpoints and ensure multi-turn continuity works end-to-end.
* **Two-Turn Live Probe:** Execute a single page load with two consecutive questions against an active `hadm_id` to verify whether the A2UI canvas appends turn blocks vertically or overwrites the active frame.
* **Vertical Feed Enforcement:** Ensure every turn (user prompt + A2UI surface: CitedNarrative, MedsGrid, LiteratureList, Risk) appends chronologically into a scrolling vertical feed for the active conversation.
* **Tool-Call Structural Replay:** Verify that the Django site transmits earlier turns—including tool calls, arguments, stored non-re-derivable payloads (prediction scores), and re-derivability flags—and that the agent reconstructs `ToolMessage` frames inside `<tool_result>` delimiter guards while re-deriving retrieved passages from the corpus.

## 2. Layer 8 Policy Revision (Updating D1–D7 for the A2UI Console)
**Objective:** Revise `layer-08-memory-session-state.md` and the site settings to support a modern, multi-thread clinical workspace while preserving agent statelessness (A3) and patient isolation (D4).
* **Retiring "Session = Page Load" (Amending D5 & G1):** A conversation's lifetime is governed by its retention window in the database, not by browser reloads. Reloading the page or selecting an active thread from the sidebar must restore the conversation and allow the user to continue asking follow-up questions without burning a new conversation credit.
* **Full Visual Recall / Composed Artifact Persistence (Closing G1):** Extend the stored turn shape in the site's PostgreSQL database to persist the rendered A2UI presentation payload alongside the question/answer pair. When a user switches back to a parallel conversation or recalls a thread from **History** or **Search**, the UI must render the exact visual state (Meds Grids, Risk Cards, Literature Cards, and Cited Narratives) identically to when they left it.
* **Anchor + Sliding Replay Window & Turn Ceiling (Amending D6 & G2):**
  * **Conversation Ceiling:** Raise the hard per-conversation turn ceiling from **6 turns to 20 turns**, and increase the daily account allowance from **10 to 25 questions/day** (keeping the 3/day timeout refund cap).
  * **6-Turn Replay Window (Anchor + Last 5):** Keep the agent's context window bounded at **6 replayed turns** (max 5 tool calls per turn, 8,000-character answer truncation). When a thread exceeds 6 turns, the site must send **Turn 1 (the clinical anchor) + the 5 most recent turns** so the agent retains initial baseline context while staying strictly within token bounds.
* **7-Day Retention Window (Amending D3):**
  * **Conversations:** Update the conversation retention window from **24 hours to 7 days (168 hours)** to cover a full inpatient admission cycle and multi-day parallel workflows. Enforced by the daily 03:00 America/New_York sweep job (`purge_expired_conversations`) deleting in batches of 500 with cascade deletion for turns and stored A2UI payloads.
  * **Note Cache:** Remains ephemeral and regenerable (`~/.cache/enterprise_clinical_copilot`), purged whole and re-fetched as needed.
  * **Evaluation Trace Archives:** Remains on a fixed 90-day audit window across the `traces*.jsonl` set in `evaluation/agent/results/`.

## 3. Strict Patient Isolation (`hadm_id` Pinning)
**Objective:** Maintain D4's clinical safety guarantee that a conversation can never cross admissions.
* **UI-Governed Context Switching:** Pin `hadm_id` at conversation creation and re-assert it on every turn. Switching patients is strictly an explicit UI action via the `@patient` selector or the **New Conversation** button—never inferred from free text.
* **Parallel Thread Support:** Selecting a different `@patient` or clicking a past thread in the sidebar cleanly switches the active workspace to that admission's thread without contaminating state.

## 4. Sidebar UI Activation & Orchestrator Follow-Ups
**Objective:** Make the stubbed sidebar navigation and follow-up affordances functional.
* **"New Conversation" Button:** Clears the active canvas, drops the current `hadm_id` pin (clearing the Patient Pill), and initializes a fresh, unbound conversation state.
* **"History" Drawer:** Fetches and lists active conversations for the account within the 7-day retention window (displaying Patient Name/`hadm_id`, timestamp, and title/snippet). Clicking a thread restores the full visual A2UI canvas, sets the active `@patient` pill to the pinned `hadm_id`, and allows seamless continuation.
* **Magnifying Glass (Search):** Filters active 7-day conversations by patient name, `hadm_id`, or query/answer text to quickly recall parallel threads.
* **Follow-Up Suggestion Chips:** Wire the orchestrator's generated follow-up suggestions into clickable chips rendered at the bottom of the latest turn block.

## 5. Script & Documentation Maintenance
**Objective:** Close the audit defects in the retention script and Layer 8 documentation.
* **Consolidated Layer 8 Doc:** Update `layer-08-memory-session-state.md` so it serves as the single canonical statement of the three holdings, their lifetimes (7-day conversations, ephemeral note cache, 90-day evaluation traces), the stored A2UI turn shape, and the Anchor + 5 replay rule.
* **Fix `retention.py` CLI Defect:** Add `--note-cache-dir` to `argparse` in `scripts/agent/retention.py` so lines 236–245 do not raise an `AttributeError` when running the default dry run or note-cache purge.