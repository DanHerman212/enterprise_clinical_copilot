# MIMIC-IV DUA — Compliance Note & Checklist

_Created 2026-08-20. Owner: Dan._

> **Disclaimer:** This is a factual risk assessment of our data handling against the
> publicly-documented MIMIC-IV Data Use Agreement (PhysioNet). It is **not legal
> advice**. For definitive questions about a specific provision, consult the DUA text
> you signed and, if needed, an attorney.

## 0. Locked decisions (2026-08-20)

- **Deploy the REAL model (`model.bst`) in production.** This is a prior, locked
  decision (recorded in serving-architecture-decisions / go-live plan). The model
  is a trained *function*; it is deployed to serve **synthetic** patient features
  only. No MIMIC patient content reaches the public surface. A trained model
  consuming synthetic inputs is **not** a redistribution or derivative of MIMIC-IV
  data under the DUA. The earlier "model.bst = exposure" finding is superseded.
- **Public demo = synthetic data, real system.** The DATA (notes, features, cohort)
  is synthetic; the SYSTEM (agent pipeline, RAG index, predict path) is live.

## 1. DUA obligations (summary)

The MIMIC-IV DUA is a signed agreement with the data steward (PhysioNet / BIDMC). Its core
prohibitions/obligations:

- **No re-identification** — do not attempt to identify individual patients.
- **No redistribution** — do not share the raw data or derivatives.
- **No sharing of credentialed access** — the data is tied to the signatory's account
  (CITI training required).
- **Research use** — use for research; commercial use generally requires permission.
- **Acknowledgment** — acknowledge PhysioNet/MIMIC in any publications.

## 2. Our use — what is compliant

| Activity | Status |
|---|---|
| Training / feature engineering / eval on MIMIC-IV in a dev environment | **Compliant** — standard permitted research use (assuming a valid signed DUA on a credentialed account). |
| Discussing methodology (training on MIMIC, model design) in publications | **Compliant and expected** — the DUA requires acknowledgment. |
| Shipping the app with **synthetic** data only, no real MIMIC content | **Compliant** — the correct public posture (Phase 2 gate). |
| Real MIMIC note passages (traces/judged JSONL) committed to a public repo | **Avoided** — these are gitignored. |

## 3. Verified exposure findings (2026-08-20)

Both GitHub repos are **public**. `enterprise_clinical_copilot` commits MIMIC-derived
artifacts that are the exposure points to address:

| Artifact | Content | Concern |
|---|---|---|
| `model.bst` | Trained XGBoost weights | **Superseded (see §0)** — locked decision to deploy the real model serving synthetic features only. Not a derivative/redistribution of MIMIC data. |
| `eval/results/golden_report*.json`, `golden_sample.json` | `hadm_id`s, probabilities, small clinical fragments quoted in judge reasons | Derivative identifiers + tiny real clinical content. |
| `docs/probes/*.json`, `spikes/*.json`, `manifest.json` | Aggregate stats, render specs, feature schema | Benign — no patient content. |

Raw note passages (`eval/results/*.jsonl`) are **not committed** (gitignored). ✅

## 4. Remediation checklist

- [x] **Resolve `model.bst` posture** — locked decision to deploy the real model on
      synthetic inputs (see §0). Not an exposure; no removal required.
- [ ] **Gate or scrub the eval report JSONs** containing `hadm_id`s + clinical fragments
      (gitignore them or move to private storage).
- [ ] Confirm no raw note text is ever committed (extend the `.gitignore` patterns if needed).
- [ ] Keep the **synthetic-cohort gate** hard: nothing public ships real MIMIC data.
- [ ] **Acknowledge PhysioNet / MIMIC-IV** in any published write-up, and describe the
      synthetic-only public surface.
- [ ] Confirm the signed DUA + CITI training are under the user's own credentialed account.

## 5. AI tooling & data handling

- Content that enters an AI-assistant conversation is **sent to the model provider** for
  processing. Raw MIMIC content has entered dev chats (via file reads/searches).
- Provider-side retention is governed by the provider's terms + account data settings —
  **verify with the provider's documentation**, do not assume.
- Practical rule: **keep raw MIMIC content out of chat.** Prefer aggregates/summaries;
  never paste notes or raw passages; rely on the synthetic cohort for public-facing work.

## 6. References

- MIMIC-IV DUA / PhysioNet credentialing: <https://physionet.org/content/mimiciv/>
- `docs/go_live_plan.md` — Phase 2 (synthetic swap) + Phase 7 (publish) compliance notes.
