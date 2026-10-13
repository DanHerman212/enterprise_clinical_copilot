# 01 - Deterministic Tool Mapping and Routing

**Objective:** Slash commands (`/risk`, `/meds`, `/ehr`, `/pubmed`) must bypass the LLM reasoning (ReAct) loop entirely. The orchestrator must execute the specific tool directly and force the agent to return a strict JSON payload that maps perfectly to specific A2UI visual components, eliminating raw text dumps and hallucinations.

### 1. The /risk Command Architecture
The ReAct loop previously hallucinated risk scores or ran them unprompted when users asked unrelated clinical questions. This command must be strictly deterministic.

*   **Intent Mapping:** `predict_readmission`
*   **Execution Rule:** Bypasses all text retrieval. Queries the `hybrid_features` table via the Vertex endpoint.
*   **Target A2UI Component:** **Risk Visualizer (New Clean-Sheet Component - Visuals to be designed in Phase 2)**. The code editor will discard the legacy UI code and create a new, purpose-built conversational widget.
*   **Payload Contract:** The agent must output a structured JSON payload containing the probability score and feature contributions to render the widget, accompanied by a strict 2-sentence clinical narrative summarizing the top driving factors. It is strictly forbidden from running secondary searches.

### 2. The /meds Command Architecture
The agent previously truncated lists or contradicted the UI cards, claiming no medications were available while the UI displayed them.

*   **Intent Mapping:** `discharge_medications`
*   **Execution Rule:** Bypasses semantic search. Retrieves all text chunks exclusively from the discharge medications section to guarantee 100% recall and prevent silent truncation.
*   **Target A2UI Component:** **Medication Table (New Clean-Sheet Component - Visuals to be designed in Phase 2)**. The code editor will discard the legacy UI code and create a new, purpose-built conversational widget.
*   **Payload Contract:** The agent must output a JSON array extracting `{"medication", "dosage", "frequency"}`. The A2UI component will render this as a clean, scannable clinical grid. The fallback disclaimer ("No specific discharge medication information is available") may only be triggered if this dedicated tool returns a genuinely empty array.

### 3. The /ehr Command Architecture
Summaries were previously presented as monolithic blocks of text with hallucinated citations and misplaced disclaimers.

*   **Intent Mapping:** `rag_search_sections`
*   **Execution Rule:** Retrieves the primary summary sections (`history_of_present_illness`, `hospital_course`, `discharge_diagnoses`, `discharge_instructions`).
*   **Target A2UI Component:** **Markdown Narrative + Source Cards (New Clean-Sheet Component - Visuals to be designed in Phase 2)**. The code editor will discard the legacy UI code and create a new, purpose-built conversational widget.
*   **Payload Contract:** The agent formats the narrative response using OpenEvidence-style bolded domain headers. It must provide a strict `citations` array. The A2UI canvas will map this array to render the clickable `[1]` popovers cleanly. It is strictly forbidden from appending unrelated disclaimers (e.g., missing medications) to this summary.

### 4. The /pubmed Command Architecture
Literature queries were previously triggering clinical warehouse searches and Vertex AI models unnecessarily.

*   **Intent Mapping:** `search_literature`
*   **Execution Rule:** Bypasses the clinical data warehouse completely and queries PubMed via NCBI E-utilities.
*   **Target A2UI Component:** **Literature Cards (New Clean-Sheet Component - Visuals to be designed in Phase 2)**. The code editor will discard the legacy UI code and create a new, purpose-built conversational widget.
*   **Payload Contract:** The agent must pass the `search_literature` results strictly into the literature card JSON payload (extracting titles, PMIDs, and abstracts). It must not dump PMIDs or raw abstracts into the text narrative, and it must never trigger a Vertex AI risk assessment when this command is fired.