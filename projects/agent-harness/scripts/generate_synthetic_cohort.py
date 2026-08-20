"""Synthetic demo cohort — feature-row generator (Task 1-2).

Builds 49-feature rows (per the served manifest.json) for the signed-off
archetypes, scores them with the served model.bst, and targets the
low/borderline/high band spread (8/8/8 at the 0.12 threshold).

Usage (repo root):
    .venv/bin/python projects/agent-harness/scripts/generate_synthetic_cohort.py
"""

import json
import random
import sys
from pathlib import Path

import xgboost as xgb

REPO = Path(__file__).resolve().parents[3]  # enterprise_clinical_copilot
MANIFEST = json.loads((REPO / "manifest.json").read_text())
FEATURES: list[str] = MANIFEST["feature_order"]
THRESHOLD = 0.12  # from threshold.json


def _jitter(rng: random.Random, lo: float, hi: float, n: int = 2) -> float:
    """Deterministic rounded jitter around (lo+hi)/2, min..max inclusive."""
    return round(rng.uniform(lo, hi), n)


# --- Archetype templates: 3 per band, each yielding 49-feature rows ----------
# 'pressure' scales the risk-driving features. one-hot groups pick exactly one.

ARCHETYPES = [
    # ------------------------------------------------------------- HIGH (8)
    {
        "band": "high", "name": "elderly_chf",
        "base": {
            "age": (75, 85), "prior_admission_count": (2, 3),
            "prior_inpatient_days": (25, 55), "recent_ed_visits": (2, 5),
            "index_los_days": (8, 14), "procedure_count": (2, 4),
            "medication_count": (14, 22), "medication_order_count": (20, 38),
            "rbc_last": (3.2, 4.0), "rbc_min": (2.8, 3.5),
            "rdw_max": (15.5, 19.5), "monocytes_min": (0.3, 0.9),
            "hemoglobin_min": (8.5, 11.5), "sodium_last": (132, 140),
            "sodium_max": (140, 148), "sodium_min": (128, 136),
            "has_procedure": 1, "oncology_flag": 0,
            "discharge": "discharge_location_snf",
        },
    },
    {
        "band": "high", "name": "oncology_infection",
        "base": {
            "age": (55, 75), "prior_admission_count": (6, 9),
            "prior_inpatient_days": (40, 70), "recent_ed_visits": (3, 6),
            "index_los_days": (10, 21), "procedure_count": (4, 8),
            "medication_count": (16, 25), "medication_order_count": (25, 45),
            "rbc_last": (2.5, 3.4), "rbc_min": (2.2, 3.0),
            "rdw_max": (16.5, 21.0), "monocytes_min": (0.0, 0.4),
            "hemoglobin_min": (6.5, 9.5), "sodium_last": (128, 136),
            "sodium_max": (138, 146), "sodium_min": (124, 132),
            "has_procedure": 1, "oncology_flag": 1,
            "discharge": "discharge_location_hospice",
        },
    },
    {
        "band": "high", "name": "copd_readmission",
        "base": {
            "age": (68, 80), "prior_admission_count": (2, 4),
            "prior_inpatient_days": (20, 45), "recent_ed_visits": (3, 6),
            "index_los_days": (7, 12), "procedure_count": (1, 3),
            "medication_count": (12, 20), "medication_order_count": (18, 34),
            "rbc_last": (3.8, 4.8), "rbc_min": (3.4, 4.4),
            "rdw_max": (14.5, 18.0), "monocytes_min": (0.4, 1.1),
            "hemoglobin_min": (9.5, 12.5), "sodium_last": (133, 141),
            "sodium_max": (141, 149), "sodium_min": (129, 137),
            "has_procedure": 1, "oncology_flag": 0,
            "discharge": "discharge_location_rehab",
        },
    },
    # ------------------------------------------------------- BORDERLINE (8)
    {
        "band": "borderline", "name": "postop_infection",
        "base": {
            "age": (52, 65), "prior_admission_count": (1, 2),
            "prior_inpatient_days": (8, 20), "recent_ed_visits": (1, 3),
            "index_los_days": (5, 8), "procedure_count": (1, 2),
            "medication_count": (8, 14), "medication_order_count": (12, 24),
            "rbc_last": (3.6, 4.6), "rbc_min": (3.2, 4.2),
            "rdw_max": (13.5, 16.5), "monocytes_min": (0.4, 1.0),
            "hemoglobin_min": (9.0, 12.0), "sodium_last": (134, 141),
            "sodium_max": (141, 148), "sodium_min": (130, 137),
            "has_procedure": 1, "oncology_flag": 0,
            "discharge": "discharge_location_home_health",
        },
    },
    {
        "band": "borderline", "name": "ckd_pneumonia",
        "base": {
            "age": (58, 72), "prior_admission_count": (1, 2),
            "prior_inpatient_days": (6, 16), "recent_ed_visits": (1, 2),
            "index_los_days": (4, 7), "procedure_count": (0, 1),
            "medication_count": (8, 14), "medication_order_count": (12, 22),
            "rbc_last": (3.4, 4.4), "rbc_min": (3.0, 4.0),
            "rdw_max": (14.0, 17.0), "monocytes_min": (0.5, 1.2),
            "hemoglobin_min": (9.0, 11.5), "sodium_last": (133, 140),
            "sodium_max": (140, 147), "sodium_min": (129, 136),
            "has_procedure": 0, "oncology_flag": 0,
            "discharge": "discharge_location_home_health",
        },
    },
    {
        "band": "borderline", "name": "diabetic_foot",
        "base": {
            "age": (48, 65), "prior_admission_count": (1, 2),
            "prior_inpatient_days": (5, 14), "recent_ed_visits": (1, 2),
            "index_los_days": (4, 7), "procedure_count": (1, 2),
            "medication_count": (7, 12), "medication_order_count": (10, 20),
            "rbc_last": (3.8, 4.8), "rbc_min": (3.4, 4.4),
            "rdw_max": (13.0, 15.5), "monocytes_min": (0.5, 1.2),
            "hemoglobin_min": (9.5, 12.5), "sodium_last": (135, 142),
            "sodium_max": (142, 149), "sodium_min": (131, 138),
            "has_procedure": 1, "oncology_flag": 0,
            "discharge": "discharge_location_home",
        },
    },
    # --------------------------------------------------------------- LOW (8)
    {
        "band": "low", "name": "routine_short",
        "base": {
            "age": (24, 40), "prior_admission_count": (0, 1),
            "prior_inpatient_days": (0, 4), "recent_ed_visits": (0, 1),
            "index_los_days": (1, 2), "procedure_count": (0, 1),
            "medication_count": (1, 4), "medication_order_count": (1, 6),
            "rbc_last": (4.2, 5.2), "rbc_min": (3.9, 4.9),
            "rdw_max": (11.5, 13.5), "monocytes_min": (0.5, 1.1),
            "hemoglobin_min": (11.0, 14.5), "sodium_last": (137, 143),
            "sodium_max": (142, 147), "sodium_min": (135, 140),
            "has_procedure": 0, "oncology_flag": 0,
            "discharge": "discharge_location_home",
        },
    },
    {
        "band": "low", "name": "minor_elective",
        "base": {
            "age": (30, 50), "prior_admission_count": (0, 1),
            "prior_inpatient_days": (0, 3), "recent_ed_visits": (0, 1),
            "index_los_days": (1, 3), "procedure_count": (1, 1),
            "medication_count": (1, 5), "medication_order_count": (2, 8),
            "rbc_last": (4.0, 5.0), "rbc_min": (3.8, 4.8),
            "rdw_max": (12.0, 14.0), "monocytes_min": (0.5, 1.1),
            "hemoglobin_min": (11.0, 14.0), "sodium_last": (136, 143),
            "sodium_max": (142, 147), "sodium_min": (134, 140),
            "has_procedure": 1, "oncology_flag": 0,
            "discharge": "discharge_location_home",
        },
    },
    {
        "band": "low", "name": "observation",
        "base": {
            "age": (20, 35), "prior_admission_count": (0, 0),
            "prior_inpatient_days": (0, 1), "recent_ed_visits": (0, 1),
            "index_los_days": (1, 1), "procedure_count": (0, 0),
            "medication_count": (0, 3), "medication_order_count": (0, 4),
            "rbc_last": (4.3, 5.3), "rbc_min": (4.0, 5.0),
            "rdw_max": (11.5, 13.0), "monocytes_min": (0.5, 1.1),
            "hemoglobin_min": (12.0, 15.0), "sodium_last": (137, 143),
            "sodium_max": (142, 146), "sodium_min": (135, 140),
            "has_procedure": 0, "oncology_flag": 0,
            "discharge": "discharge_location_home",
        },
    },
]

# One-hot groups and their options (first is default; pick one per row).
RACE = ["race_white", "race_black", "race_hispanic", "race_asian", "race_amind", "race_nhpi", "race_unknown"]
ADMIT = ["admission_type_ew_emer", "admission_type_eu_obs", "admission_type_obs_admit",
         "admission_type_urgent", "admission_type_direct_emer", "admission_type_ambulatory_obs",
         "admission_type_direct_obs", "admission_type_unknown"]
DISCHARGE = ["discharge_location_home", "discharge_location_home_health", "discharge_location_snf",
             "discharge_location_rehab", "discharge_location_ltac", "discharge_location_hospice",
             "discharge_location_ama", "discharge_location_psych", "discharge_location_assisted_living",
             "discharge_location_unknown"]
INSURANCE = ["insurance_medicare", "insurance_medicaid", "insurance_private", "insurance_other", "insurance_unknown"]


def make_row(spec: dict, rng: random.Random, sex: str = "M") -> dict:
    b = spec["base"]
    row = {}
    for f in FEATURES:
        row[f] = 0.0
    # continuous + flags
    for f, v in b.items():
        if isinstance(v, tuple):
            row[f] = _jitter(rng, v[0], v[1])
        elif f in ("has_procedure", "oncology_flag"):
            row[f] = float(v)
    # categorical one-hots
    row["gender"] = 1.0 if sex == "F" else 0.0
    row[rng.choice(RACE)] = 1.0
    row[rng.choice(ADMIT)] = 1.0
    row[b["discharge"]] = 1.0
    row[rng.choice(INSURANCE)] = 1.0
    return row


def main() -> int:
    rng = random.Random(42)
    booster = xgb.Booster()
    booster.load_model(str(REPO / "model.bst"))

    candidates = []
    for spec in ARCHETYPES:
        for sex in ("M", "F"):
            for _ in range(3):  # 6 variants per archetype -> oversample for selection
                row = make_row(spec, rng, sex)
                row["_archetype"] = spec["name"]
                row["_band_target"] = spec["band"]
                candidates.append(row)

    scored = []
    for row in candidates:
        vec = [[row[f] for f in FEATURES]]
        dm = xgb.DMatrix(vec, feature_names=FEATURES)
        prob = float(booster.predict(dm)[0])
        scored.append((prob, row))

    scored.sort(key=lambda x: x[0])

    def band_of(p: float) -> str:
        if p < THRESHOLD:
            return "low"
        if p < THRESHOLD + 0.08:
            return "borderline"
        return "high"

    # report all candidate bands vs targets
    from collections import Counter
    tgt = Counter(r["_band_target"] for _, r in scored)
    actual = Counter(band_of(p) for p, _ in scored)
    print("Candidates:", len(scored))
    print("  target bands  :", dict(tgt))
    print("  actual bands  :", dict(actual))

    # selection: 8 per actual band, closest to band target midpoint
    mid = {"low": THRESHOLD / 2, "borderline": THRESHOLD + 0.04, "high": THRESHOLD + 0.18}
    chosen = {}
    for band in ("low", "borderline", "high"):
        pool = [(abs(p - mid[band]), p, row) for p, row in scored if band_of(p) == band]
        pool.sort(key=lambda x: x[0])
        chosen[band] = pool[:8]

    print("\nSelected cohort (8/8/8 target):")
    for band in ("low", "borderline", "high"):
        print(f"  {band}:")
        for _, p, row in chosen[band]:
            print(f"    p={p:.4f}  {row['_archetype']:<18} age={row['age']:.0f} "
                  f"prior={row['prior_admission_count']:.0f} los={row['index_los_days']:.0f}")

    counts = Counter(band_of(p) for _, p, _ in (y for band in chosen.values() for y in band))
    print("\nSelected band counts:", dict(counts))

    # Emit the final 24-patient cohort with synthetic hadm_ids (90000001+).
    out = []
    idx = 1
    for band in ("low", "borderline", "high"):
        for _, p, row in chosen[band]:
            out.append({
                "hadm_id": 90_000_000 + idx,
                "archetype": row["_archetype"],
                "band": band,
                "probability": round(float(p), 6),
                "threshold": THRESHOLD,
                "features": {f: row[f] for f in FEATURES},
            })
            idx += 1
    out_path = REPO / "projects/agent-harness/eval/results/synthetic_cohort.json"
    out_path.write_text(json.dumps({"seed": 42, "n": len(out), "patients": out}, indent=2))
    print(f"\nWrote {out_path} ({len(out)} patients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
