# Architectural Specification & Execution Directive: `/ehr` Live PubMed Retrieval, Structured Comparator Extraction & Stage-Gated Verification

**Date:** 2026-11-02  
**Status:** Authoritative Implementation Directive (Stage-Gated)  
**Target Scope:** `services/mcp/`, `services/agent/`, `evaluation/platform/`, `tests/`

---

## 1. Executive Objective & Two-Stage Execution Gate

The Stage 1 Forensic Audit (`01-ehr_literature_trace_eval_forensic_audit.md`) revealed that `/ehr` (`command.summarize`) explicitly forbids `search_literature` (`routing.py:249`) and instead fabricates the `ComparativeRiskMatrix` inside `presentation/` (`a2ui.py`, `canvas.py`, `domain.py`) using naive keyword matching against a 9-row static dictionary (`BENCHMARKS`).

We are replacing that static lookup completely with **100% live PubMed E-utilities retrieval and LLM/structured comparator extraction**—with **zero hardcoded study tables, zero static phenotype buckets, and zero `hadm_id` maps**.

### Mandatory Two-Stage Gate (STOP AFTER STAGE GATE 1)
To prove that live PubMed retrieval and structured 30-day readmission extraction work reliably on real patient charts before wiring into `presentation/` or deploying:
* **Stage Gate 1 (Execute Now):** Build and test ONLY the backend pipeline—(1) clinical query derivation from EHR note sections, (2) live NCBI PubMed E-utilities retrieval tailored for 30-day readmission studies, and (3) structured comparator extraction & clinical alignment verification. Run this backend pipeline against **8 real inpatient admissions** (the 3 diagnostic admissions `90000090`, `90000069`, `90000081` + **5 unseen held-out `hadm_id` admissions**) and print the complete diagnostic report. **DO NOT wire into `presentation/` or delete legacy UI wiring until Stage Gate 1 is reviewed and approved.**
* **Stage Gate 2 (Execute Only After Approval of Stage Gate 1):** Wire the verified backend output into `/ehr` (`deterministic.py`), dismantle the static `BENCHMARKS` / `register_benchmark_sources` hack in `presentation/`, move the Langfuse telemetry serialization boundary after presentation composition in `turn.py`, and upgrade `evaluation/platform/` (`evaluators.py` and `judge_call.py`).

---

## 2. Non-Negotiable Governance & Anti-Reward-Hacking Laws

1. **Law 1 — Zero Static Study Stores or Phenotype Buckets Anywhere:**
   * Do NOT create any hardcoded dictionary of PubMed studies, PMIDs, or phenotype buckets in `mcp/`, `engine/`, or `presentation/`.
   * Every candidate study MUST be fetched dynamically from live NCBI E-utilities (`esearch.fcgi` + `efetch.fcgi`) based on the patient's actual clinical record.
2. **Law 2 — Zero Patient-ID (`hadm_id`) Hardcoding or Demo Special-Casing:**
   * Never branch on `hadm_id` or write bespoke rules targeting `90000090`, `90000069`, or `90000081`.
3. **Law 3 — Strict Six-Subpackage Layer DAG (`test_layer_separation.py`):**
   * Unidirectional import DAG: `ingress` $\rightarrow$ `routing` $\rightarrow$ `engine` $\rightarrow$ `postprocess` $\rightarrow$ `presentation` $\leftarrow$ `telemetry`.
4. **Law 4 — Strict Verbatim Grounding & Clinical Specificity:**
   * Every extracted comparator row (`reported_rate`, `cohort_summary`, `evidence_tier`, `rate_sentence`, `pmid`) MUST come verbatim from a live retrieved PubMed abstract that (a) explicitly reports a 30-day readmission rate (or numeric readmission percentage) in its abstract text, and (b) matches the patient's primary admission condition/procedure in `Discharge diagnosis` / `Brief hospital course`.
   * Never substitute an anatomically or clinically mismatched cohort (e.g., never match a knee arthroplasty study to a hip fracture patient; never match a lumbar spine study to a cervical spine patient; never match a COPD study to a patient without COPD).
   * If live PubMed returns zero abstracts that both match the patient's clinical condition and contain an explicit readmission rate, return an empty comparator list `[]` (which the UI will render honestly via `baseline_note` / index admission baseline rather than fabricating or forcing an unrelated disease study).

---

## 3. STAGE GATE 1: Backend Live PubMed Retrieval & Structured Extraction Pipeline

### 3.1 Step 1 — Upgrade Live PubMed Retrieval for Readmission Benchmarks (`services/mcp/tools/literature.py`)

Currently, `search_literature` (`services/mcp/tools/literature.py:330-391`) queries NCBI E-utilities with `DEFAULT_MAX_RESULTS = 3` and `RECENCY_YEARS = 5`, returning raw abstracts that frequently lack numeric readmission percentages.

Enhance `services/mcp/tools/literature.py` (while keeping existing `search_literature` behavior backward-compatible for general `/pubmed` queries):
1. **Targeted Readmission Query Construction & E-Utilities Fetch:**
   * Support a wider retrieval pool when searching for readmission benchmarks (or add an optional parameter / query builder in `literature.py`):
     * Construct an NCBI E-utilities search query combining the patient's primary clinical condition/procedure with explicit readmission outcome terms:
       `(<clinical condition/procedure terms>) AND ("30-day readmission"[Title/Abstract] OR "readmission rate"[Title/Abstract] OR "unplanned readmission"[Title/Abstract] OR "hospital readmission"[Title/Abstract])`
     * Use a broader publication window (e.g., 10–15 years or unrestricted `datetype` for landmark readmission cohorts) and fetch up to `retmax=12` candidate records via `esearch.fcgi` + `efetch.fcgi`.
2. **Candidate Pre-Filtering on Abstract Text:**
   * Among the live PubMed records returned by `efetch.fcgi`, prioritize/filter to articles whose `<AbstractText>` contains both a readmission term (`readmission` / `readmitted` / `rehospitalization`) and an explicit percentage (`\d+(?:\.\d+)?\s*%`), so the top `max_results` (e.g., 5–6) abstracts returned to the agent are rich in quantitative readmission data.
   * If the primary specific query returns fewer than 2 quantitative abstracts, allow a secondary live E-utilities query broadening the procedure/diagnosis terms to the core primary condition (still strictly derived from the patient's `Discharge diagnosis`—never unrelated diseases).

### 3.2 Step 2 — Clinical Query Derivation & Structured Comparator Extraction (`services/agent/engine/`)

Create a dedicated, testable backend module or functions in `services/agent/engine/` (e.g., `services/agent/engine/literature_benchmarks.py` or within `deterministic.py`):

1. **Primary Condition & Procedure Extraction from EHR Note Sections:**
   * Input: The note sections returned by `rag_search_sections` (`Discharge diagnosis`, `Brief hospital course`).
   * Extract the **primary admission diagnosis and index surgical/medical procedure** (prioritizing `Discharge diagnosis` first, and the opening admission/procedure sentences of `Brief hospital course` second).
   * Explicitly ignore incidental past medical history, chronic stable bystanders, or negative statements.
   * Formulate the live PubMed search query string targeting that primary admission condition/procedure.

2. **LLM / Structured Comparator Extraction & Clinical Alignment Gate:**
   * Input:
     * Patient's `Discharge diagnosis` and `Brief hospital course` summary.
     * The live PubMed articles (`pmid`, `title`, `journal`, `pub_date`, `authors`, `abstract`, `url`) returned by `search_literature`.
   * For each returned PubMed abstract, evaluate and extract structured comparator rows (`0` to `2` rows maximum):
     * **`clinical_match` (boolean + rationale):** Does the study cohort's primary pathology and anatomical site/procedure directly match this patient's index admission?
       * *Strict rejection rules:* Reject if the study is for a different anatomical joint/level (e.g., TKA/knee for a hip fracture; lumbar for cervical spine) or a condition absent from `Discharge diagnosis` (e.g., COPD when patient has pleural effusion/AFib without COPD).
     * **`horizon_days` (int):** Must report a 30-day readmission rate (`horizon_days == 30`).
     * **`reported_rate` (str):** The exact 30-day readmission percentage stated in the abstract (e.g., `"8.4%"`).
     * **`rate_sentence` (str):** The **exact verbatim sentence** from the PubMed abstract containing `reported_rate` (verified via substring/normalized check against `article["abstract"]` so the model can never hallucinate a rate or sentence).
     * **`cohort_summary` (str):** Concise description of the study population, sample size (`n=...`), and setting stated in the abstract.
     * **`evidence_tier` (str):** Study design stated in the abstract/title (e.g., `"Retrospective Cohort"`, `"Meta-Analysis"`, `"Observational Cohort"`, `"Registry Study"`).

3. **Deterministic Grounding Verification on Extracted Rows:**
   * After structured extraction, enforce a deterministic post-check on every extracted row:
     * `row.pmid` must exist in the live `search_literature` response.
     * `row.reported_rate` (the numeric figure) must literally appear in `article["abstract"]`.
     * `row.rate_sentence` must be a verbatim sentence from `article["abstract"]`.
     * Any row failing deterministic verification is discarded.

---

## 4. STAGE GATE 1 VERIFICATION PROTOCOL (STOP HERE & REPORT)

Before wiring anything into `presentation/` (`a2ui.py`, `canvas.py`, `domain.py`), `turn.py`, or `evaluation/platform/`:

1. **Write Backend Unit & Integration Tests (`tests/agent/test_live_literature_extraction.py`):**
   * Test deterministic verification (rejecting any non-verbatim `rate_sentence`, non-matching `pmid`, or non-30-day rate).
   * Test clinical alignment filtering (verifying that a hip fracture chart rejects a TKA abstract, a cervical spine chart rejects a lumbar durotomy abstract, and a pleural effusion/AFib chart without COPD rejects a COPD abstract).

2. **Run Live End-to-End Backend Extraction on 8 Real Patients & Print the Full Diagnostic Report:**
   * Execute the new backend pipeline (`rag_search_sections` $\rightarrow$ query derivation $\rightarrow$ live `search_literature` against NCBI PubMed $\rightarrow$ structured comparator extraction & deterministic verification) on:
     * **The 3 Diagnostic Admissions:** `90000090`, `90000069`, `90000081`
     * **5 Unseen Held-Out Inpatient Admissions (`hadm_id`s)** chosen from the corpus across diverse clinical conditions.
   * For each of the 8 admissions, print:
     1. `hadm_id`
     2. Patient's actual `Discharge diagnosis` and primary procedure from `Brief hospital course`
     3. The exact PubMed query sent to `search_literature`
     4. Number of live PubMed abstracts returned and their PMIDs + titles
     5. Which PMIDs were **accepted** vs. **rejected** (with the rejection reason for rejected abstracts)
     6. The final extracted comparator row(s): `pmid`, `journal`, `pub_date`, `cohort_summary`, `evidence_tier`, `reported_rate`, and the verbatim `rate_sentence` from the abstract.

**STOP AFTER PRINTING THIS 8-PATIENT BACKEND REPORT.** Wait for user review and approval of the live backend extraction quality before proceeding to Stage Gate 2.

---

## 5. STAGE GATE 2 (Preview — Execute Only After Stage Gate 1 Approval)

Once Stage Gate 1 is approved:
1. **Wire into `/ehr` (`routing.py` & `deterministic.py`):** Remove `"search_literature"` from `/ehr`'s `forbidden` set, execute the live `search_literature` + comparator extraction step inside `run_command` for `summarize`, and store the verified comparator rows in state.
2. **Retire the Static Hack in `presentation/` (`domain.py`, `canvas.py`, `a2ui.py`):** Remove static `BENCHMARKS` lookup and `register_benchmark_sources` fabrication; populate `ComparativeRiskMatrix` and its `sources` entries strictly from the turn's settled `search_literature` comparators.
3. **Fix Langfuse Telemetry (`turn.py` & `observability.py`):** Move trace output/metadata recording after `compose_presentation` so `tool.search_literature` and the rendered `ComparativeRiskMatrix` rows appear in Langfuse.
4. **Upgrade Evaluation Harness (`task.py`, `evaluators.py`, `judge_call.py`):** Pass rendered `canvas` / `ComparativeRiskMatrix` rows to both deterministic evaluators and the LLM Judge prompt to enforce clinical cohort alignment across the evaluation suite.