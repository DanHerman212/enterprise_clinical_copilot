"""fill_features.py — Step 2.7: story-anchored feature fill + coherence gate.

Band fit (2.6) showed the naive healthy-baseline fill is a *story-blind lower
bound*: it zeros prior admissions/LOS, so clinically complex notes (e.g. 1195,
89yo, 18 meds, SNF) score artificially low. This script replaces that with a
STORY-ANCHORED fill: every feature family gets a rule that parses it from the
text when present, and otherwise derives a plausible value from signals in the
note (prior-admission mentions, ED mentions, length-of-stay phrases, chronic
disease count, age-based insurance, etc.).

Scoring mirrors the deployed ReadmissionPredictor exactly (local model.bst +
manifest.json + threshold.json): calibrated probability + TreeSHAP attributions
aggregated to parent groups + top_factors. The output includes, per note, the
provenance of every feature (parsed vs filled + basis) so the coherence rule
(note story <-> feature row <-> risk score) can be verified note by note.

Usage (from projects/agent-harness):
  ../../.venv/bin/python scripts/fill_features.py
Output: data/mtsamples/fill.json  { sample_id: {...} }
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[3]  # enterprise_clinical_copilot
HARNESS = REPO / "projects" / "agent-harness"
DATA_DIR = HARNESS / "data" / "mtsamples"
MANIFEST = json.loads((REPO / "manifest.json").read_text())
FEATURES: list[str] = MANIFEST["feature_order"]
GROUPS: dict[str, list[str]] = MANIFEST.get("groups", {})
THRESHOLD = float(json.loads((REPO / "threshold.json").read_text())["threshold"])

RACE_KEYS = ["race_white", "race_black", "race_hispanic", "race_asian",
             "race_amind", "race_nhpi", "race_unknown"]
ADMIT_KEYS = ["admission_type_ew_emer", "admission_type_eu_obs",
              "admission_type_obs_admit", "admission_type_urgent",
              "admission_type_direct_emer", "admission_type_ambulatory_obs",
              "admission_type_direct_obs", "admission_type_unknown"]
DISCHARGE_KEYS = ["discharge_location_home", "discharge_location_home_health",
                  "discharge_location_snf", "discharge_location_rehab",
                  "discharge_location_ltac", "discharge_location_hospice",
                  "discharge_location_ama", "discharge_location_psych",
                  "discharge_location_assisted_living",
                  "discharge_location_unknown"]
INSURANCE_KEYS = ["insurance_medicare", "insurance_medicaid", "insurance_private",
                  "insurance_other", "insurance_unknown"]

TOP_K = 10


# --- parsing primitives ------------------------------------------------------

def _parse_age(text: str) -> int | None:
    m = re.search(r"\b(\d{2,3})\s*(?:-year-old|-yr-old|-year|-yr)\b", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\bage(?: of)?\s+(\d{1,3})\b", text, re.I)
    return int(m.group(1)) if m else None


def _parse_gender(text: str) -> str | None:
    # female check first so "male ... female" phrasing stays coherent.
    if re.search(r"\b(?:female|woman|she|her)\b", text, re.I):
        return "F"
    if re.search(r"\b(?:male|man|he|him)\b", text, re.I):
        return "M"
    return None


def _parse_race(text: str) -> str | None:
    for key, pat in [
        ("race_white", r"\b(caucasian|white)\b"),
        ("race_black", r"\b(african[- ]american|black)\b"),
        ("race_hispanic", r"\b(hispanic|latino?)\b"),
        ("race_asian", r"\b(asian|vietnamese|filipino|korean|chinese|indian)\b"),
        ("race_amind", r"\b(american indian|native american)\b"),
        ("race_nhpi", r"\b(pacific islander|hawaiian)\b"),
    ]:
        if re.search(pat, text, re.I):
            return key
    return None


def _parse_meds(text: str) -> int:
    """Count distinct medication names, tolerant of MTSamples dose formats.

    Handles "ganciclovir 275 mg" (lowercase drug), "Percocet 5/500" (fraction
    dose), and "metoprolol tartrate 25 mg" (multi-word drug) — the naive
    uppercase+mg counter misses all of these and would fill 0 meds for notes
    that clearly list meds, producing incoherent cards.
    """
    drugs = set()
    # multi-word drug + dose: "metoprolol tartrate 25 mg", "ganciclovir 275 mg IV"
    for m in re.finditer(
        r"\b([A-Za-z][A-Za-z\-]+(?: [A-Za-z\-]+){0,2})\s+"
        r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?)\b", text):
        drugs.add(m.group(1).lower())
    # fraction-dose style: "Percocet 5/500", "Vicodin 5/500"
    for m in re.finditer(
        r"\b([A-Za-z][A-Za-z\-]+(?: [A-Za-z\-]+){0,1})\s+\d+/\d+\b", text):
        drugs.add(m.group(1).lower())
    # ignore words that are not drug names
    stop = {"the", "with", "was", "and", "for", "plus", "patient", "were",
            "from", "this", "that", "have", "has", "will", "her", "his",
            "she", "he", "status"}
    return len(drugs - stop)


def _parse_discharge(text: str) -> str | None:
    for key, pat in [
        ("discharge_location_hospice", r"\bhospice\b"),
        ("discharge_location_snf", r"\b(skilled nursing|snf|nursing home)\b"),
        ("discharge_location_rehab", r"\brehab(?:ilitation)?\b"),
        ("discharge_location_ltac", r"\blong[- ]term (?:care|acute)\b"),
        ("discharge_location_home_health", r"\bhome health|home care\b"),
        ("discharge_location_assisted_living", r"\bassisted living\b"),
        ("discharge_location_psych", r"\bpsych(?:iatric)? (?:hospital|unit|facility)\b"),
        ("discharge_location_ama", r"\bagainst medical advice|\bAMA\b"),
    ]:
        if re.search(pat, text, re.I):
            return key
    return None


def _count_patterns(text: str, patterns: list[str]) -> int:
    n = 0
    for pat in patterns:
        n += len(re.findall(pat, text, re.I))
    return n


def _parse_los_days(text: str) -> float | None:
    for pat in [
        r"\blength of stay\s+(?:was|is)?\s*(\d+)\s*days?",
        r"\b(?:hospitalized|admitted|hospital) for\s+(\d+)\s+days?",
        r"\b(\d+)[- ]day (?:hospital|admission|stay)",
        r"\bhospital day[s]?\s+(\d+)\b",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            return float(m.group(1))
    return None


def _parse_hgb(text: str) -> float | None:
    m = re.search(r"\b(?:hemoglobin|hgb)\s*(?:of|was|is)?\s*(\d{1,3}(?:\.\d+)?)\b", text, re.I)
    return float(m.group(1)) if m else None


def _parse_na(text: str) -> float | None:
    m = re.search(r"\bsodium\s*(?:of|was|is)?\s*(\d{1,3}(?:\.\d+)?)\b", text, re.I)
    return float(m.group(1)) if m else None


# --- story-anchored fill -----------------------------------------------------

PRIOR_ADM_PATTERNS = [
    r"\bprior (?:hospital|inpatient|admission|admissions|icu|stay|stays)\b",
    r"\bprevious (?:hospital|inpatient|admission|admissions|icu|stay|stays)\b",
    r"\bre[- ]?admission\b",
    r"\bwas (?:previously|prior) (?:admitted|hospitalized)\b",
    r"\bhad been admitted\b",
    r"\bhistory of multiple admissions\b",
]
# MTSamples records prior surgical history as "status post [procedure]". A
# distinct prior procedure IS prior inpatient contact, so it counts toward
# prior_admission_count (a big coherence miss if ignored — complex patients
# otherwise score artificially low).
PRIOR_PROC_PATTERNS = [
    r"\bstatus post\b\s+(?:an?\s+|the\s+|bilateral\s+)?[A-Za-z][^,.;]{3,60}",
]
ED_PATTERNS = [
    r"\b(?:emergency room|emergency department|emergency center)\b",
    r"\b(?:e\.?r\.?|ed)\s+(?:visit|evaluation|presentation)\b",
    r"\bpresented to the (?:emergency|er|ed)\b",
]
CHRONIC_PATTERNS = [
    r"\b(?:diabetes|hypertension|chf|congestive heart failure|copd|ckd|chronic kidney|esrd|asthma|cad|coronary artery disease|pvd|peripheral vascular)\b",
]


def _count_prior_procedures(text: str) -> int:
    """Distinct prior procedures ("status post X") minus known false anchors.

    "status post" can also precede a non-procedure noun (e.g. "status post
    bilateral carotid stenting" is fine; "status post admission" is not a
    procedure). We only count matches that do NOT mention admission-related
    words, and dedupe by the phrase so one sentence lists count once.
    """
    phrases = re.findall(PRIOR_PROC_PATTERNS[0], text, re.I)
    seen: set[str] = set()
    for p in phrases:
        low = p.lower()
        if any(w in low for w in ("admission", "admit", "hospital", "discharge", "clinic")):
            continue
        # collapse to first 4 words for dedupe
        key = " ".join(low.split()[:4])
        seen.add(key)
    return len(seen)


def _provisional_row(text: str) -> tuple[dict[str, float], dict]:
    """Story-anchored fill. Returns (row, provenance).

    provenance[sid][feature] = ("parsed", value) | ("filled", basis, value)
    """
    row: dict[str, float] = {f: 0.0 for f in FEATURES}
    prov: dict = {}
    age = _parse_age(text)
    gender = _parse_gender(text)
    race = _parse_race(text)
    discharge = _parse_discharge(text)
    meds = _parse_meds(text)
    hgb = _parse_hgb(text)
    na = _parse_na(text)
    los = _parse_los_days(text)
    prior = _count_patterns(text, PRIOR_ADM_PATTERNS)
    prior += _count_prior_procedures(text)
    ed = _count_patterns(text, ED_PATTERNS)
    chronic = _count_patterns(text, CHRONIC_PATTERNS)
    has_proc = 1.0 if re.search(r"\b(?:underwent|surgery|operation|procedure)\b", text, re.I) else 0.0
    onco = 1.0 if re.search(r"\b(?:oncology|chemotherapy|malignant|metastatic|cancer)\b", text, re.I) else 0.0

    # age
    if age is not None:
        row["age"] = float(age); prov["age"] = ("parsed", age)
    elif re.search(r"\b(?:elderly|geriatric)\b", text, re.I):
        row["age"] = 78.0; prov["age"] = ("filled", "elderly signal", 78.0)
    elif re.search(r"\b(?:young|infant|child)\b", text, re.I):
        row["age"] = 30.0; prov["age"] = ("filled", "young signal", 30.0)
    else:
        row["age"] = 60.0; prov["age"] = ("filled", "no signal default", 60.0)

    # gender — model encoding is 1 = male (mirrors mlops/src/encoding.py
    # `CAST(gender = 'M' AS INT64)`), so a male note fills 1.0, female 0.0.
    if gender is not None:
        row["gender"] = 1.0 if gender == "M" else 0.0; prov["gender"] = ("parsed", gender)
    else:
        # No gender signal in the note. Default 0.0 (=F, model encoding) so a
        # signal-free note reads as female; the display name follows the same
        # rule, keeping the two consistent.
        row["gender"] = 0.0; prov["gender"] = ("filled", "no signal default F", "F")

    # race
    race = race or "race_unknown"
    row[race] = 1.0
    prov["race"] = ("parsed" if race != "race_unknown" else "filled", race)

    # prior admissions / inpatient days / ED
    row["prior_admission_count"] = float(min(prior, 9))
    prov["prior_admission_count"] = ("parsed" if prior else "filled",
                                     f"{prior} mention(s)/prior procedure(s)"
                                     if prior else "no prior-admission signal")
    row["prior_inpatient_days"] = float(min(prior * 14.0, 70.0)) if prior else 0.0
    prov["prior_inpatient_days"] = ("filled", f"~14d x {prior} prior admission signal"
                                    if prior else "no prior signal -> 0")
    row["recent_ed_visits"] = float(min(ed, 5))
    prov["recent_ed_visits"] = ("parsed" if ed else "filled",
                                f"{ed} ED mentions" if ed else "no ED signal")

    # LOS
    if los is not None:
        row["index_los_days"] = los
        prov["index_los_days"] = ("parsed", los)
    else:
        # no explicit LOS -> infer from complexity signals (chronic burden,
        # procedures, age) with a modest default.
        inferred = 3.0 + 1.5 * chronic + (2.0 if has_proc else 0.0)
        if age and age >= 75:
            inferred += 1.5
        row["index_los_days"] = round(min(inferred, 14.0), 1)
        prov["index_los_days"] = ("filled", f"inferred {row['index_los_days']}d from chronicity/proc/age")

    # procedures
    row["has_procedure"] = has_proc
    row["procedure_count"] = 1.0 if has_proc else 0.0
    prov["has_procedure"] = ("parsed", has_proc)
    prov["procedure_count"] = ("filled", "1 if has_procedure else 0")

    # meds
    row["medication_count"] = float(meds)
    row["medication_order_count"] = round(float(meds) * 1.6, 1)
    prov["medication_count"] = ("parsed", meds)
    prov["medication_order_count"] = ("filled", "meds x 1.6")

    # oncology
    row["oncology_flag"] = onco
    prov["oncology_flag"] = ("parsed", onco)

    # labs (parse hemoglobin/sodium; derive rbc family from hgb)
    if hgb is not None:
        row["hemoglobin_min"] = hgb
        # rough rbc estimate: hgb ~= 3*rbc
        rbc = round(hgb / 3.0, 2)
        row["rbc_last"] = rbc
        row["rbc_min"] = round(rbc - 0.3, 2)
        prov["hemoglobin_min"] = ("parsed", hgb)
        prov["rbc_last"] = ("filled", f"hgb/3 = {rbc}")
        prov["rbc_min"] = ("filled", "rbc_last - 0.3")
    else:
        row["hemoglobin_min"] = 12.5; row["rbc_last"] = 4.1; row["rbc_min"] = 3.9
        prov["hemoglobin_min"] = ("filled", "no lab signal -> 12.5")
        prov["rbc_last"] = ("filled", "no lab signal -> 4.1")
        prov["rbc_min"] = ("filled", "no lab signal -> 3.9")
    row["rdw_max"] = 13.5
    prov["rdw_max"] = ("filled", "no lab signal -> 13.5")
    row["monocytes_min"] = 0.7
    prov["monocytes_min"] = ("filled", "no lab signal -> 0.7")

    if na is not None:
        row["sodium_last"] = na
        row["sodium_max"] = na + 2.0
        row["sodium_min"] = na - 2.0
        prov["sodium_last"] = ("parsed", na)
        prov["sodium_max"] = ("filled", "last + 2")
        prov["sodium_min"] = ("filled", "last - 2")
    else:
        row["sodium_last"] = 139.0; row["sodium_max"] = 141.0; row["sodium_min"] = 137.0
        prov["sodium_last"] = ("filled", "no lab signal -> 139")
        prov["sodium_max"] = ("filled", "no lab signal -> 141")
        prov["sodium_min"] = ("filled", "no lab signal -> 137")

    # admission type: ED mention -> emergency; else default urgent-ish
    if re.search(r"\b(?:emergency room|emergency department|er|ed)\b", text, re.I):
        row["admission_type_ew_emer"] = 1.0
        prov["admission_type"] = ("parsed", "ew_emer via ED mention")
    else:
        row["admission_type_unknown"] = 1.0
        prov["admission_type"] = ("filled", "no admit-type signal -> unknown")

    # discharge location
    discharge = discharge or "discharge_location_home"
    row[discharge] = 1.0
    prov["discharge_location"] = ("parsed" if discharge != "discharge_location_home"
                                  else "filled", discharge)

    # insurance: age-based (never in-text)
    if age is not None and age >= 65:
        row["insurance_medicare"] = 1.0
        prov["insurance"] = ("filled", "age>=65 -> medicare")
    elif re.search(r"\b(?:medicaid|uninsured|self[- ]pay)\b", text, re.I):
        row["insurance_medicaid"] = 1.0
        prov["insurance"] = ("parsed", "medicaid/uninsured mention")
    else:
        row["insurance_private"] = 1.0
        prov["insurance"] = ("filled", "age<65 -> private")

    return row, prov


# --- scoring (mirrors ReadmissionPredictor) ---------------------------------

def _score(row: dict[str, float]) -> dict:
    vec = np.asarray([[row[f] for f in FEATURES]], dtype=np.float32)
    dm = xgb.DMatrix(vec, feature_names=FEATURES)
    booster = _load_booster()
    probs = booster.predict(dm)
    contribs = booster.predict(dm, pred_contribs=True)
    base = float(contribs[0][-1])
    by_index = {name: float(contribs[0][j]) for j, name in enumerate(FEATURES)}
    attributions = {
        parent: sum(by_index.get(c, 0.0) for c in cols)
        for parent, cols in GROUPS.items()
    }
    top = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_K]
    prob = float(probs[0])
    return {
        "probability": prob,
        "prediction": int(prob >= THRESHOLD),
        "threshold": THRESHOLD,
        "base_value": base,
        "top_factors": [{"feature": name, "attribution": val} for name, val in top],
    }


_booster: xgb.Booster | None = None


def _load_booster() -> xgb.Booster:
    global _booster
    if _booster is None:
        _booster = xgb.Booster()
        _booster.load_model(str(REPO / "model.bst"))
    return _booster


def band_of(p: float) -> str:
    if p < THRESHOLD:
        return "low"
    if p < THRESHOLD + 0.08:
        return "borderline"
    return "high"


def main() -> int:
    results: dict[str, dict] = {}
    for path in sorted(DATA_DIR.glob("*.txt")):
        sid = path.stem
        text = path.read_text(encoding="utf-8")
        row, prov = _provisional_row(text)
        score = _score(row)
        results[sid] = {
            "probability": round(score["probability"], 4),
            "band": band_of(score["probability"]),
            "prediction": score["prediction"],
            "base_value": round(score["base_value"], 4),
            "top_factors": [
                {"feature": f["feature"], "attribution": round(f["attribution"], 4)}
                for f in score["top_factors"]
            ],
            "provenance": prov,
        }

    DATA_DIR.joinpath("fill.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )

    from collections import Counter
    bands = Counter(r["band"] for r in results.values())
    print("story-anchored band distribution (of %d):" % len(results))
    print(" ", dict(bands))
    print("\nsorted by probability:")
    for sid in sorted(results, key=lambda s: results[s]["probability"]):
        r = results[sid]
        top = ", ".join(f"{t['feature']}={t['attribution']:+.2f}"
                        for t in r["top_factors"][:3])
        print(f"  {sid:>5}  {r['probability']:.4f}  {r['band']:<10}  top: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
