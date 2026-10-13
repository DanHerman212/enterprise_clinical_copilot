# Enterprise Clinical Copilot: UX Workshop Agenda

**Objective:** Transition the copilot from a probabilistic prototype to a deterministic, composable clinical application. We will work deliberately, step-by-step, to establish a flawless golden path that delights users, followed by proactive features that excite evaluators.

## Phase 1: Nailing the Baseline Intent (The "Delight")
* **Deconstruct the Turns:** Analyze specific survey turns to define exactly how the orchestrator should route intent.
* **Locking the Control Flow:** Establish the strict boundary between deterministic slash commands and probabilistic natural language.

## Phase 2: Visuals & Streaming Performance (The "Elegance")
* **A2UI Component Refactor:** Design clean, structured payload contracts to replace the legacy text formatting, ensuring the visual widgets seamlessly fit the modern UI shell.
* **Parallel Execution & TTFT:** Architect the backend to stream narrative tokens immediately (Time to First Token) while deterministic tools and guardrails execute asynchronously to eliminate the visual loading lag.

## Phase 3: Session & State Management (The "Continuity")
* **Thread Binding:** Map the Django `demo_conversation` state precisely to the active `hadm_id`.
* **Sidebar Navigation:** Design the data retrieval bridge to dynamically populate the left-hand "History" menu, allowing seamless context switching between different patient threads without losing state.

## Phase 4: Designing the Proactive UX (The "Excite")
* **Anticipating Next Steps:** Design context-aware follow-up suggestions (e.g., transitioning from a `/risk` score directly to a targeted `/pubmed` query).
* **Advanced Tooling Integration:** Weave in low-overhead, high-impact capabilities to make the interface feel intelligent and anticipatory.

## Phase 5: The Golden Path Blueprints
* **Patient-by-Patient Choreography:** Script the ideal, step-by-step narrative for our anchor patients to perfectly demonstrate the architecture to evaluators.

## Phase 6: Eval Rubric & Handoff
* **Defining Success:** Formalize a strict evaluation rubric for the code editor to meet before we shift into production hardening.