# 02 - Visuals & Streaming Performance

**Objective:** Replace static, legacy dashboard widgets with clean-sheet, conversational Lit/A2UI components designed for a flowing chat interface. Implement text streaming to optimize Time-To-First-Token (TTFT) and enforce strict clinical UX patterns.

### 1. The Streaming Engine
To reduce perceived latency, the orchestrator and A2UI frontend must support streaming for the `narrative_text` payload.
*   **Execution:** The narrative prose (whether from a slash command or free text) must stream to the canvas token-by-token. 
*   **Component Rendering:** Structured visual components (Risk Visualizer, Medication Grid, Literature Cards) should render immediately once their respective JSON payloads are complete, seamlessly integrating below or alongside the streaming text.

### 2. Clinical Citations ("Glance-then-Verify" Pattern)
Citations must provide instant verification without forcing context-switching or aggressive screen-takeovers.
*   **Interaction:** Inline citation markers (e.g., `^[1]`) must be clickable/hoverable elements.
*   **Visual Component:** Triggering the marker opens a lightweight **Rich Popover**. This tooltip must display the exact source text chunk and the section origin (e.g., "History of Present Illness") mapped from the `citations` JSON array. 

### 3. The Risk Visualizer (`/risk`)
A purpose-built Lit/A2UI conversational widget replacing the legacy dashboard view, designed to group positive and negative influencers visually so clinicians do not have to read dense paragraphs.
*   **Narrative Constraint:** The text narrative must *only* state the score and threshold relation (e.g., "The readmission risk score is 0.14, which is above the operating threshold of 0.11."). The LLM must **not** list the driving factors in the prose.
*   **Data Binding:** Consumes the exact probability score (rounded to 2 decimal places for display) and the translated feature contributions from the JSON payload.
*   **Visual Layout (Influencer Grouping):** A clean, two-column (or clearly divided) A2UI card grouping the features into **"Increasing Risk"** (positive influencers) and **"Decreasing Risk"** (negative influencers) using color-coded icons and typography.
*   **Color Banding:** Implement conditional styling mapped to the clinical threshold (e.g., > 0.11 renders the overall header with a "High Risk" warning color, < 0.11 renders with a "Low Risk" safe color).

### 4. The Clinical Data Grid (`/meds`)
*   **Visual Component:** A lightweight, scannable A2UI data table.
*   **Data Binding:** Maps the `{medication, dosage, frequency}` JSON payload into a strict grid, replacing hallucinatory bulleted lists with a structured clinical layout.

### 5. Literature Cards (`/pubmed`)
*   **Visual Component:** Clean, stacked A2UI cards.
*   **Data Binding:** Extracts the title, PMID (hyperlinked to PubMed), and abstract from the JSON payload. 

### 6. Composer UX Cleanup
*   **Execution:** Completely remove the legacy static "chip" UI affordances from the user input container. Rely entirely on the existing `/` slash-command pop-up menu for feature discovery and command execution.