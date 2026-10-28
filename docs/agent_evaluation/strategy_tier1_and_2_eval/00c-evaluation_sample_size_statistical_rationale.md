# Evaluation Sample Size & Statistical Rationale: Why Tier 1 ($N=87$) + Tier 2 ($N=41$) Establishes Production Trust

## Executive Summary
A common question in applied AI engineering is whether a **26-scenario Dev Suite** and a **15-scenario Blind Holdout Suite** ($N = 41$ multi-turn scenarios total) are sufficient to establish technical trust.

**Yes—because the evaluation architecture is two-tiered and stratified across orthogonal failure modes, rather than a standalone 41-question random sample.**

If an open-domain consumer chatbot were evaluated with only 41 single-turn questions and zero deterministic underlying checks, that sample size would be insufficient. In domain-specific clinical RAG systems (following industry patterns from Anthropic, Hamel Husain, and Eugene Yan), sample size requirements are governed by **population tiering, 2D stratification, and variance control**.

---

## 1. Tier 1 Covers 100% of the Data Population ($N = 87 / 87$)
In typical enterprise RAG pipelines, engineering teams rely on hundreds of LLM-as-a-Judge examples because they inadvertently use the LLM judge to catch data-parsing bugs, schema hydration failures, and retrieval crashes across an unvalidated corpus.

* **Zero Sampling Error on Data Integrity:** Our **Tier 1 Deterministic Cohort Sweep** tests **100% of the served cohort** in Python—all 82 inpatient records, 5 outpatient records, and 2 unserved boundary records.
* **Complete Population Verification:** Every patient manifest hydration, underlying tool schema output, and outpatient/unserved boundary guardrail is verified deterministically across the entire population before Tier 2 begins.

---

## 2. Tier 2 Is Stratified Across Orthogonal Failure Modes (Not Random Sampling)
Randomly sampling 40 prompts from user logs produces weak coverage because ~80% of unstratified queries test the exact same "happy path." Instead, Tier 2 uses a **2D Stratified Taxonomy (10 Patient Phenotypes $\times$ 7 Interaction & Adversarial Trajectories)**:

* **41 Multi-Turn Scenarios $\approx$ 90–100 Individual Agent Turns:** Across 26 Dev scenarios and 15 Holdout scenarios, the harness executes nearly 100 distinct conversational turns, ~180 tool invocations, and **246+ individual rubric dimension grades** (`fidelity`, `grounding`, `citation`, `responsiveness`, `safety`, `context`).
* **Proven Defect-Discovery Yield:** The empirical test of whether a Dev set is appropriately sized is its defect-discovery rate. The 26-scenario Dev matrix surfaced **10 distinct architectural and clinical defects** across iterations `00c` through `00c6` (including outpatient encounter routing, `/pubmed` citation channels, score deflection, excursion-based lab ranking, structured-vs-unstructured lab precedence, and cross-modality citation bleed). A dense, orthogonal matrix of 26 scenarios that exposes 10 real architectural defects provides far higher signal than 500 synthetic single-turn questions that trivially pass at 99%.

---

## 3. Statistical Boundaries of the 26 Dev / 15 Holdout Split ($N = 41$)

Understanding both the statistical power and the mathematical boundaries of the 26 Dev / 15 Holdout split ensures rigorous technical defense during architectural reviews:

| Metric | Dev Suite ($N = 26$) | Blind Holdout ($N = 15$) | Combined Tier 2 ($N = 41$) |
| :--- | :--- | :--- | :--- |
| **Target Pass Rate ($\ge 85\%$)** | $24/26 = 92.3\%$ | $\ge 13/15 = 86.7\%$ | $\ge 37/41 = 90.2\%$ |
| **Safety / Structural Tolerance ($0$ failures)** | $0 / 26$ | $0 / 15$ | $0 / 41$ (~$92.7\%$ lower-bound confidence via Rule of Three: $1 - 3/N$) |
| **Binomial 95% Confidence Interval (at $\sim 92\%$ pass)** | $\pm 10.2\%$ | $\pm 14.0\%$ | **$\pm 8.3\%$** |
| **Judge Run Cost / Wall-Clock Time (Concurrency 1)** | ~12–18 mins | ~8–10 mins | ~25 mins total |

### What $N = 15$ Holdout Proves Reliably
* **Detects Systemic Overfitting / Reward Hacking (Goodhart's Law):** If the Dev Suite only passed because of brittle keyword interceptors, regex rules, or prompt phrases tuned to the 6 Dev patients, the Holdout pass rate will collapse (e.g., dropping to $50\%\text{–}65\%$) when confronted with 4 unseen patient records and paraphrased clinical shorthand.

### What $N = 15$ Holdout Cannot Detect
* **Fine-Grained Percentage Differentiation:** Because a single scenario in a 15-item suite carries a weight of $\frac{1}{15} = 6.67\%$, a 15-item Holdout cannot statistically distinguish between an $88\%$ underlying capability and a $94\%$ capability (which represents a single scenario flipping).

---

## 4. Architectural Defense Summary
When explaining the evaluation sample size to stakeholders or technical evaluators:

> *"We separated deterministic population testing (**Tier 1**, covering 100% of the 87 patient records in Python with zero sampling error) from stratified multi-turn behavioral evaluation (**Tier 2**, covering 41 multi-turn scenarios across 10 patient archetypes and 7 interaction/adversarial trajectories). Using a strict 65/35 Dev-Holdout split ($26$ Dev / $15$ Blind Holdout) across ~100 conversational turns prevents Goodhart's Law and proves generalization to unseen patient records while maintaining zero-tolerance gating on clinical safety and structural integrity."*