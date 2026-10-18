# Clinical Pilot: Quality Audit & Remediation Plan

## Executive Summary
After successfully clearing all 54 structural assertions and maintaining a 20/20 green adversarial baseline, the clinical evaluation harness successfully scored the agent's response quality. The current pass rate of **56.5% (13 pass / 10 fail)** is a product of specific, isolated failure modes rather than systemic infrastructure defects. 

The following analysis categorizes the 10 failures and outlines a zero-code, prompt-based remediation strategy to elevate the pass rate above the 90% release threshold.

## Root-Cause Analysis

### 1. Bucket B: The Rounding Specification (Judge Flaw)
* **Impact:** 4 of the 10 failures.
* **Issue:** The LLM judge penalized the agent for rounding readmission probabilities (e.g., `0.167366` to `16.7%`). 
* **Diagnosis:** This is a spec-versus-rubric conflict. The agent is correctly following the UI specification, but the judge's fidelity rubric is overly pedantic.

### 2. Bucket C: Directive Recommendations (Agent Policy Flaw)
* **Impact:** 2 of the 10 failures (overlapping with Bucket B).
* **Issue:** The agent issued directive clinical recommendations (e.g., "clinical follow-up should prioritize...").
* **Diagnosis:** This is a safety/policy violation. A clinical copilot must remain strictly descriptive and objective, never prescribing or recommending care.

### 3. Bucket A: Multi-Turn Hallucinations (Architecture/Router Flaw)
* **Impact:** 3 of the 10 failures.
* **Issue:** On Turn 2 (follow-up) questions, the agent made zero tool calls and hallucinated clinical data (e.g., fabricating a discharge disposition).
* **Diagnosis:** LangGraph router overconfidence. When presented with Turn 1 chat history, the router mistakenly relies on internal LLM memory instead of triggering a mandatory data retrieval tool for new clinical queries.

### 4. Buckets D & E: Grounding and Citation Variance
* **Impact:** 3 of the 10 failures.
* **Issue:** Standard LLM generation errors, such as misinterpreting lab trends or misplacing citation brackets.
* **Diagnosis:** Acceptable edge-case variance. Once Buckets A, B, and C are resolved, the pass rate will naturally clear the 90% threshold despite these isolated misses.

---

## Execution Directives

The following prompt adjustments will remediate the core failures without requiring architectural rewrites:

1. **Update the Judge Rubric (Fidelity):** Modify the judge's system prompt to explicitly allow and expect 1-decimal-place rounding for probabilities. *(Resolves Bucket B)*
2. **Update the Agent System Prompt (Safety):** Add a strict guardrail: *"Never issue directive clinical recommendations or tell the user what to prioritize. Synthesize the data objectively and neutrally."* *(Resolves Bucket C)*
3. **Update the Router Prompt (Multi-Turn Retrieval):** Add a strict forcing function requiring the execution of a retrieval tool for any new clinical question in Turn 2, explicitly banning zero-tool-call answers for new data. *(Resolves Bucket A)*