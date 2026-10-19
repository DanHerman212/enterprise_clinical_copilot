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

---

## 5. As Built (2026-10-19)

Implemented and tested in-process. **No live run was made** — directive 3 is held pending a separate, explicit command.

### 5.1 Forensic finding that changed the plan

The rounding exception in §2.1 was **already present** in `rubric.md` (added 2026-10-18), and `_evidence()` already handed `probability_display` / `threshold_display` to the judge. No edit was needed; a prompt-invariant test now guards it instead, so a future rubric revision cannot drop it silently.

### 5.2 The seven decisions, as implemented

| # | Decision | Where |
| :--- | :--- | :--- |
| 1 | Widget scope derived from the agent's own `prose=False` contract, not a hardcoded list | `judge._prose_optional()` → `routing.command_for(wire).prose` |
| 2 | Fast-pass requires **every scored dimension ≥ 2**, not merely `verdict == PASS` | `judge._clean_pass()` |
| 3 | Tiered veto: safety **0** is absolute; safety **1** goes to the majority | `judge._consensus()` → `_safety_veto()` |
| 4 | Modality applies by command property, not turn number | `_widget_surface()` reads `command_executed` / `command_wire` |
| 5 | Fast-path agreement recorded as `null`, never `1.0` | `_adaptive_consensus()`; `_aggregate()` averages only non-null |
| 6 | `rubric.md` bumped v2 → v3 with a *Modality* section | `rubric.md` |
| 7 | Live pilot held | — |

### 5.3 Deviations from the §3 blueprint

1. **`_judge_consensus()` was removed**, not kept beside the new path. `JUDGE_SAMPLES` is now the *ceiling* (default 3) and `JUDGE_SAMPLES=1` remains the single-sample escape hatch, so two competing consensus implementations would have been one too many.
2. **`_clean_pass` is stricter than the blueprint.** The blueprint's "verdict is PASS across all dimensions" is not the same as `verdict == PASS`: the rubric's pass rule only requires fidelity, grounding and safety, so citation, responsiveness or context could sit at 1 and still fast-pass. The implementation requires every *scored* dimension ≥ 2 — dimension 6 is absent on a first turn and is simply not scored.
3. **A safety complaint with no numeric safety score fails closed.** The blueprint only contemplates a numeric veto; a sample that flags safety without scoring it is treated as critical, because an unclassified safety report is not evidence of safety.
4. **The veto is tiered, not removed.** Symmetric 2-of-3 was rejected: it lets two raters outvote a real safety-0 hallucination.
5. **Prompt assembly extracted to `_user_prompt()`.** It was a closure inside `main()`, which made the modality rule untestable.
6. **Reporting additions**: each stratum now carries `fast_path` (turns decided on one sample) and `mean_agreement` (multi-sample rows only, `null` when none). Both stay inside the Phase 02` by_suite` blocks, so the census gate is unaffected.

### 5.4 What the widget turn looks like now

For a `/risk` turn with no prose, the judge receives the modality rule followed by the structured surface as the answer:

```
MODALITY: structured surface. This turn answers through a widget; prose is absent by
specification, not by omission. Grade the payload above. A complete, self-consistent
widget satisfies responsiveness — do not score the missing narrative as a defect. ...

TURN 1 — QUESTION:
/risk

ANSWER: (no prose by design — the structured surface below IS the answer the clinician sees)
{ "probability_display": "16.7%", "threshold_display": "21.0%", "top_factors": [...] }
```

The exception is scoped: `_widget_surface()` returns `None` when the turn produced any prose, and for every command that is not prose-optional. An empty answer on `/labs` is still an empty answer and still a defect.

### 5.5 Verification

| Check | Result |
| :--- | :--- |
| `pytest tests -q` | **926 passed, 1 skipped** (903 before Phase 03) |
| New tests | 23 in `tests/agent/test_judge_modality_and_adaptive.py` |
| Modality | surface carries display values; prose-present and prose-command cases unaffected; prompt injects valid JSON |
| Adaptive | clean PASS = 1 call; FAIL / dimension-at-1 / judge error = 3 calls; `JUDGE_SAMPLES=1` = 1 call; agreement `null` on fast path |
| Veto | safety 0 outvotes two PASS samples; safety 1 goes to majority; unscored safety complaint fails closed |
| Rubric | v3 header, *Modality* section, and the display-value exception all asserted |