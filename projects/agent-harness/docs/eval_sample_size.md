# Eval Sample-Size Grounding — Readmission Agent (2026-08-23)

_Reference for how we sized the hybrid agent evaluation after the ground-up
LangChain rewrite (Sprint C). Written to be citable in eval docs / results._

## The question

Validating the rewrite requires evaluating the live agent. How large must the
eval be to produce a **statistically defensible** result for a clinical/technical
audience — not just "we ran it and it looked fine"?

Options considered:

| Option | Questions | Corpus |
|---|---|---|
| 24-patient eval | 72 (24 × 3 prompt types) | live hybrid demo cohort |
| 108-note eval | 324 (108 × 3 prompt types) | full curated MTSamples corpus |

## Framing: the claim determines the required sample size

Each agent answer is judged pass/fail, so the eval is a **binomial process**
(observed pass rate = estimate of an unknown true rate). The right `n` depends
entirely on **which claim** we need to make:

1. **"The rewrite didn't break the architecture"** — catch systemic failures
   (broken tool loop, no traces, guardrail off). Systemic failures fail at
   near-100%, so even `n=24` catches them. Small `n` is adequate here.
2. **"The system is ≥ X% accurate"** — absolute threshold claim. Requires the
   **Wilson lower bound** ≥ X.
3. **"The system is safe (0 safety failures)"** — requires the **rule of three**
   upper bound on the hidden failure rate.
4. **"The new system is equivalent to the old (no regression)"** — requires
   **two-proportion power**; much larger `n`.

## The machinery

### 1. Wilson score interval — precision of an observed pass rate

For observed $\hat{p} = k/n$ at confidence $z = 1.96$:

$$
\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

Preferred over the normal approximation near 0/1 (where a 95% pass rate lives).
**The lower bound is the number you can actually claim.**

| Sample | Questions | Observed | 95% CI | Lower bound |
|---|---|---|---|---|
| 24 patients × 3 | 72 | 94.4% | (86.6%, 97.8%) | **87%** |
| 108 notes × 3 | 324 | 95.1% | (92.1%, 96.9%) | **92%** |
| prior golden eval | 300 | 95.0% | (91.9%, 96.9%) | **92%** |

- `n=72` cannot support a "≥90% accurate" claim (lower bound 87%).
- `n=324` supports it (lower bound 92%).

Per-task (each prompt type is its own stratum):
- `n=24`/task → CI (79.8%, 99.3%), lower bound **80%**
- `n=108`/task → CI (89.6%, 98.0%), lower bound **90%**

### 2. Rule of three — bound on a hidden rate when 0 failures observed

If 0 failures in `n` trials, the 95% **upper bound** on the true failure rate is
found from $(1-p)^n = 0.05$, which for small $p$ gives $p \le 3/n$.

| Questions | 0-failure upper bound |
|---|---|
| 72 | **4.2%** |
| 324 | **0.9%** |

"Zero safety failures" at `n=72` only rules out rates above ~4% — not a
defensible safety bound for a clinical tool. At `n=324` it rules out above 0.9%.

### 3. Two-proportion power — can we detect a regression?

Per-group sample size to detect a drop $p_1 \to p_2$ at 80% power, α=0.05:

$$
n = \frac{\left(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_1-p_2)^2}
$$

| Detect | Per group needed |
|---|---|
| 95% → 90% (5-pt regression) | **435** |
| 95% → 85% | 141 |
| 95% → 80% | 76 |

Smaller effects need quadratically more data. Neither 72 nor 324 can honestly
claim "equivalent to the old system within ±5 points" — but both can catch a
catastrophic (15+ pt) regression, which is the architecture claim.

## The decision

**Chosen: 108 notes → 324 questions.** It supports:
- "≥90% accurate" (Wilson lower bound 92%)
- "≤0.9% safety-failure rate" (rule of three)
- Direct scale-comparability with the prior published 300-question eval (95%)

`n=72` was rejected because it only supports the narrow "architecture is sound"
claim — it undersells the accuracy and safety claims a clinical audience will
scrutinize.

## Caveats

- Neither 72 nor 324 supports an equivalence claim within ±5 points (~435/group).
- The judge is an **LLM-as-judge** (rubric-scored), which adds measurement
  variance on top of the binomial. True required `n` is therefore **at least** as
  large as the binomial math suggests — never smaller.
- Report **confidence intervals**, not point estimates.
- **Reproducibility**: run the eval twice and show stability (prior golden eval
  reproduced 285/300 across two runs).

## References

- Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and
  Statistical Inference.* JASA 22(158), 209–212.
- Hanley, J. A. & Lippman-Hand, A. (1983). *If Nothing Goes Wrong, Is Everything
  All Right? Interpreting Zero Numerators.* JAMA 249(13), 1743–1745.
- Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates
  and Proportions* (3rd ed.). Wiley.
- Karpinska, N., Akef, M., & Iyyer, M. (2021). *The Perils of Using Mechanical
  Turk to Evaluate Open-Ended Text Generation.* EMNLP 2021.
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot
  Arena.* NeurIPS 2023.

## Decision log

- 2026-08-23 — Chose the 108-note (324-question) eval over 24 (72) after the
  statistical grounding above. Owner: Dan Herman. Driver: trust with a
  clinical/technical audience; "no shortcuts."
