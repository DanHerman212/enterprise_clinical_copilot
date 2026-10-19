# 03 — Rubric Design & Judge Calibration

## 1. Executive Summary & Problem Statement
During initial pilot evaluations, the LLM-as-a-Judge produced significant friction and false-positive failures due to three distinct evaluation flaws:
1. **Spec-versus-Rubric Conflict (Rounding Penalties):** The judge penalized the agent for rounding readmission probabilities (e.g., `0.167366` to `16.7%`), even though `probability_display` and `threshold_display` are the exact rendering formats specified by the product's UX contract.
2. **Widget-Only Penalties (`/risk` Turn 1):** Commands like `/risk` are designed to emit structured A2UI widgets (`RiskInfluencers`) with minimal prose. The clinical rubric evaluated these turns against narrative criteria, penalizing them for empty prose.
3. **Stochastic Variance & Quota Exhaustion (Unconditional 3-Sample Consensus):** Running 3 independent judge samples unconditionally across every turn triggered vertex rate limits (`429 RESOURCE_EXHAUSTED`), while an asymmetric "1-of-3 safety veto" introduced a 27.1% false-failure rate on borderline turns.

This module establishes **Phase 03**, codifying spec-aligned rubric exceptions, modality-aware payload grading, and the cost-effective **Adaptive 2-of-3 Adjudication** protocol.

---

## 2. Rubric Adjustments: Spec Alignment & Widget Grading

### 2.1 Authoritative Display Values (`rubric.md`)
To prevent the judge from penalizing the agent for following UI specifications, `evaluation/agent/rubric.md` is updated to explicitly exempt readmission probabilities and thresholds:
* **Fidelity Rule:** Rounding is generally a fidelity defect, **with one exception**. For `probability_display` and `threshold_display`, quoting 1-decimal percentage formatting (e.g., `0.167366` → `"16.7%"`) or raw floats is fully compliant. Rounding any other quantity (lab values, doses, dates) remains a fidelity defect.

### 2.2 Modality-Aware Grading for Widget-Only Turns (`judge.py`)
For commands that serve as structured widgets (specifically Turn 1 `/risk`), narrative prose is intentionally sparse or absent. `judge.py` and the system prompt must evaluate these turns based on their structured data payloads (`probability_display`, `threshold_display`, and top factors) rather than penalizing them for lack of conversational prose.

---

## 3. Code Specifications: Adaptive 2-of-3 Adjudication (`evaluation/agent/judge.py`)

To eliminate `429` rate limit exhaustion and remove the unstable single-sample safety veto, we replace unconditional 3-sample voting with **Adaptive 2-of-3 Adjudication**:
1. **Fast Pass (1 Sample at Temp=0):** Run a single judge call. If the verdict is `PASS` across all dimensions (with no safety flags), accept it immediately and skip additional model calls. This cuts LLM judge costs and latency by ~75% on healthy turns.
2. **Adaptive Tie-Breaker (2 Additional Samples on FAIL):** If the initial sample returns `FAIL` or raises a safety concern, fire **2 additional tie-breaker samples** concurrently.
3. **Symmetric 2-of-3 Majority Consensus:** Take the strict 2-of-3 median/majority across all three samples for both numerical dimensions and categorical verdicts. Remove any hair-trigger single-sample safety veto.

### Implementation Blueprint (`judge.py`)
Update the consensus and execution logic in `evaluation/agent/judge.py`:

```python
# In evaluation/agent/judge.py:
def _adaptive_consensus(client, system: str, user: str) -> dict:
    """Adaptive Adjudication: 1 sample on PASS; 2 tie-breakers on FAIL (2-of-3 majority)."""
    # 1. Run initial sample at temp=0
    first_sample = _judge(client, system, user)
    
    # If it passed cleanly and has no error, accept it immediately (fast path)
    if first_sample.get("verdict") == "PASS" and not first_sample.get("error"):
        first_sample["_samples"] = 1
        first_sample["_agreement"] = 1.0
        return first_sample

    # 2. On FAIL or error, execute 2 additional tie-breaker samples concurrently
    samples = [first_sample]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_judge, client, system, user) for _ in range(2)]
        for future in futures:
            try:
                samples.append(future.result())
            except Exception as exc:
                samples.append({"error": f"sample failed: {type(exc).__name__}"})

    # 3. Compute 2-of-3 majority across usable samples
    return _consensus(samples)
```

## 4. Operational Directives for Phase 03
1. **Commit Rubric Edits**: Ensure evaluation/agent/rubric.md contains the probability_display rounding exception.

2. **Deploy Adaptive Adjudication**: Update judge.py to use _adaptive_consensus(), eliminating unnecessary vertex calls and stabilizing multi-sample variance.

3. **Re-Run Pilot Evaluation**: Execute the local pilot evaluation run using the adaptive judge protocol against the live agent to verify that pilot pass rates settle reliably without hitting 429 resource exhaustion.