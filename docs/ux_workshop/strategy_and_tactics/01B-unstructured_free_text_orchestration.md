# 01B - Unstructured Free Text Orchestration

**Objective:** Define the strict orchestration boundary for unstructured natural language queries (e.g., "Was there any concern about a blood clot in her leg?"). When a query enters the probabilistic ReAct loop, the orchestrator must dynamically constrain the agent's tool access, enforce strict data-grounding rules, and return a structured JSON payload for clean A2UI rendering.

### 1. Tool Pruning (The Constrained Sandbox)
To prevent the ReAct loop from hallucinating risk scores or retrieving overly broad medication data, the agent's toolkit must be dynamically pruned for free-text queries.

*   **Forbidden Tools:** `predict_readmission` and the new 100%-recall `discharge_medications` tool must be completely hidden from the LLM during free-text reasoning. If a user requires a risk score or a comprehensive medication list, they must explicitly use the designated `/slash` commands.
*   **Allowed Tools:** The LLM may only access standard clinical semantic search (`rag_search_sections`) and, conditionally, literature search (`search_literature`).

### 2. The Semantic Search Guardrail
When the LLM utilizes semantic search to answer a clinical question, it must be strictly anchored to the returned evidence.

*   **Execution Rule:** The semantic search tool must force the LLM to synthesize its answer based *exclusively* on the retrieved chunks. General medical knowledge must not be used to fill in factual gaps about the patient's record.
*   **Fallback Protocol:** If the search returns no relevant chunks for the specific question, the LLM is strictly forbidden from guessing or hallucinating an answer. It must reply with a standard fallback: "There is no information in the patient's chart regarding [Topic]."

### 3. The Free-Text Payload Contract
Even though the user input is unstructured, the backend output must remain strictly structured to prevent raw markdown dumps on the canvas.

*   **Target A2UI Component:** **Conversational Chat Bubble + Source Cards (New Clean-Sheet Component - Visuals to be designed in Phase 2)**. The code editor will discard legacy text rendering and prepare a fresh chat component.
*   **Payload Contract:** The agent must return the exact same strict JSON structure designed for the `/ehr` command. This includes a `narrative_text` field for the clinical answer and a strict `citations` array. The LLM must use explicit `^[1]` inline markers in the narrative that map perfectly to the citation array, allowing the frontend to render clickable citation popovers.