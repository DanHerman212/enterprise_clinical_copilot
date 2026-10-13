# Clinical Copilot: UI/UX Redesign & Architectural Strategy

## 1. The Challenge: Escaping the "Legacy Trap"
The initial UI design relied on a traditional CRUD (Create, Read, Update, Delete) layout, featuring a large left-hand patient database sidebar and rigid, static buttons for AI actions. 

**Issues Identified:**
*   **Wasted Real Estate:** The persistent database list consumed valuable screen space, leaving the actual workspace feeling empty once a patient was selected.
*   **Constrained Interaction:** Static buttons for features (Risk, EHR, Meds, PubMed) limited the physician's ability to ask organic, open-ended questions.
*   **Misalignment with Clinical Workflows:** Modern clinical applications (like OpenEvidence) use immersive, conversation-driven interfaces embedded directly into patient charts, rather than separate dashboards.

## 2. The Target Experience: Modern, Conversational, Context-Aware
The goal is to replicate the highly adopted, dense, and engaging chat interfaces seen in leading AI tools, tailoring them for medical workflows. 

**Core UX Principles:**
*   **Implicit Patient Context:** Simulate a SMART on FHIR integration where the app operates *within* a patient's context, rather than forcing the doctor to search for them globally.
*   **Unified Chat Canvas:** The text input bar becomes the primary engine for all interactions.
*   **Traceability:** Every AI claim must include clickable inline citations linking back to deterministic EHR chunks or primary medical literature.

## 3. Key Architectural Shifts

### A. The `@` Command Entry Point & Database Split
Instead of a permanent left-hand sidebar, patient selection is moved into the chat bar using an `@` command (e.g., `@Andre Ashford`). 
*   **Local Application DB (Django/SQL):** Stores a narrow table of patient metadata (`patient_id`, `name`, `dob`) to power sub-millisecond, real-time UI autocomplete keystrokes.
*   **Data Warehouse (BigQuery):** Stores the heavy structured synthetic EHR data.
*   **The Handoff:** Once a patient is selected via the `@` command, the frontend passes the hidden `patient_id` to the Model Context Protocol (MCP), where the SQL Agent injects it into BigQuery queries.

### B. Slash Commands (`/`) for Core Features
The four static UI buttons are converted into floating slash commands within the chat input, training the user on system capabilities while allowing open text for complex queries:
1.  `/risk`: Runs the 30-day readmission model (surfaces SHAP explanations).
2.  `/ehr`: Deterministically searches and summarizes the clinical timeline.
3.  `/meds`: Reconciles and organizes discharge medications.
4.  `/pubmed`: Triggers RAG search for clinical literature with citations.

### C. Dynamic Component Rendering (A2UI)
To prevent overwhelming "walls of text," the A2UI renderer will intercept specific data structures from the agent's stream. When the agent outputs complex data (like a readmission probability or medication table), A2UI will render it as a rich, visual UI component embedded directly in the chat feed or the right-hand Context Canvas.

## 4. Immediate Next Steps / Action Plan
1.  **Tear Down:** Remove the persistent left-hand patient list to reclaim screen width.
2.  **Implement Entry Point:** Work with the coding agent to build the `@patient_name` auto-complete functionality using the lightweight local SQL database.
3.  **Wire First Feature:** Integrate the first slash command (e.g., `/risk` or `/ehr`) to test the handoff between the UI context, the MCP tools, and BigQuery. 
4.  **Componentize Outputs:** Begin mapping agent data outputs to A2UI visual cards.