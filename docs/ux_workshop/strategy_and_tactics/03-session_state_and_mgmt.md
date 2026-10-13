# Phase 3: Session & State Management (The "Continuity")

## Executive Summary
This phase bridges the stateless AI agent with the stateful Django backend to establish session continuity. The objective is to enforce the 24-hour fixed retention policy, wire the structural replay mechanism for follow-up questions, and ensure the frontend strictly maintains the visual and data state of the active patient without dropping previous turns.

## 1. The Store & Patient Isolation
**Objective:** Enforce the rule that a conversation is an immutable pairing of one account and one patient.
* **Initialization & Pinning:** When a session opens, the backend must pin the conversation to the active `hadm_id`. A reload of the page terminates the session and requires a new credit.
* **Context Switching:** If the orchestrator detects an intent to query a different patient, the system must definitively close the current thread and initialize a new conversation rather than appending it.

## 2. Frontend Continuity & Canvas Appending
**Objective:** Preserve the visual history of the conversation during an active session so clinicians do not lose context when asking follow-up questions.
* **Scrolling Vertical Feed:** The frontend A2UI canvas must stop overwriting its contents on submit. It must be configured as a scrolling vertical feed.
* **Turn Appending:** New tool executions, Meds Grids, Cited Narratives, and Literature Cards must **append** below previous turns. The full visual thread of the rendered page load must remain visible and accessible until the session is explicitly cleared or the page is reloaded.

## 3. Structural Replay (Orchestrator Integration)
**Objective:** Enable follow-up questions without violating the agent's stateless constraint.
* **Payload Construction:** The Django site must construct and send a structural replay of previous turns alongside the new question.
* **Context Window Bounds:** The payload must strictly enforce the backend limits: at most 6 earlier turns, at most 5 tool calls per replayed turn, and automatic truncation (with visible markers) for any stored answer exceeding 8,000 characters.
* **Delimiter Guarding:** Replayed clinical data must be strictly framed within instructional delimiters to prevent prompt injection channels from re-opening.

## 4. Economics, Retention & Sweeping
**Objective:** Wire the session lifecycle to the 24-hour sweep and daily allowances.
* **Fixed 24-Hour TTL:** Enforce the rule that a conversation lives for a fixed 24 hours from the moment it opens (not a sliding window). 
* **The 03:00 Sweep:** Ensure the `demo_conversation` data model is fully compatible with the existing 03:00 America/New_York scheduled deletion job. Turns must cascade delete with the conversation.
* **Credit Tracking:** Wire the UI to respect the account's 10-question daily allowance, claiming a credit only when a conversation successfully opens.

## 5. Sidebar UI Activation
**Objective:** Wire the stubbed sidebar elements while respecting backend storage constraints.
* **"New Conversation":** Wire this button to cleanly drop the active `hadm_id` context, clear the canvas, and initialize a new backend session.
* **History Recall (Constraint Acknowledged):** Due to the deferred artifact re-rendering policy, the History sidebar will either be suppressed for this phase or wired to display read-only, text-based transcripts without interactive A2UI components.