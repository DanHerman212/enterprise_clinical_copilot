"""Select the demo cohort, assign synthetic names, and emit the seed artifact.

Supersedes `build_demo_cohort.py`, which selected on the *label*. That was a
defensible default but produced a poor demo: of its 41 patients, 31 scored
above the 0.12 threshold and the probabilities jumped from 0.0991 straight to
0.1314, so the cohort contained no patient just *below* the line. The borderline
cases are the whole reason a threshold and SHAP factors are worth showing, so
selection here is driven by predicted risk.

What it does:

  1. scores a wide test-split pool against the live endpoint, in batches
  2. picks a fixed number of admissions from each risk band
  3. rewrites the `demo_cohort` BigQuery table to exactly those admissions
  4. assigns deterministic synthetic names and clinical descriptors
  5. writes data/demo_cohort.json for the Django loader

Selection rules (from BUILD_GUIDE section 14):

  - test split only. A demo patient the model trained on proves nothing.
  - 20-40 patients spanning the decision boundary, including cases just below
    it, not only clear highs and clear lows.
  - fixture patients pinned in regardless, so Tier 1 keeps passing.
  - deterministic: same input, same cohort, same names, every run.

> Rebuilding the table invalidates the Feature Store sync. Re-run
> setup_feature_store.py --sync afterwards, or live demos will serve a stale
> cohort while BigQuery serves the new one.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from google.cloud import bigquery

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from mcp_server.config import COHORT_TABLE_FQN, PROJECT, TABLE_FQN  # noqa: E402
from mcp_server.endpoint import endpoint  # noqa: E402
from mcp_server.features import to_vector  # noqa: E402
from mcp_server.features.manifest import feature_order  # noqa: E402

FIXTURE_PATH = HARNESS_ROOT / "tests" / "fixtures" / "expected.json"
OUTPUT_PATH = HARNESS_ROOT / "data" / "demo_cohort.json"

# Bands are expressed against the operating threshold rather than hardcoded, so
# a recalibrated threshold reshapes the cohort instead of silently invalidating
# it. (lower, upper, how many, label)
def bands(threshold: float) -> list[tuple[float, float, int, str]]:
    return [
        (0.00, threshold - 0.06, 4, "clear low"),
        (threshold - 0.06, threshold - 0.03, 5, "low"),
        (threshold - 0.03, threshold, 6, "just below threshold"),
        (threshold, threshold + 0.03, 6, "just above threshold"),
        (threshold + 0.03, threshold + 0.13, 6, "moderate"),
        (threshold + 0.13, 1.01, 4, "high"),
    ]


# Era-appropriate pools, because a 90-year-old named Madison is a tell that
# nobody thought about it. Sex comes from the real record; only the name is
# invented.
FIRST_NAMES = {
    ("F", "older"): ["Margaret", "Dorothy", "Eleanor", "Ruth", "Joan", "Barbara",
                     "Shirley", "Doris", "Betty", "Marjorie", "Gloria", "Rosemary"],
    ("F", "middle"): ["Linda", "Susan", "Deborah", "Karen", "Patricia", "Cynthia",
                      "Sandra", "Denise", "Yvonne", "Paula", "Theresa", "Loretta"],
    ("F", "younger"): ["Ashley", "Jasmine", "Megan", "Brittany", "Alicia", "Chelsea",
                       "Danielle", "Kayla", "Erica", "Monique", "Sierra", "Vanessa"],
    ("M", "older"): ["Harold", "Walter", "Eugene", "Raymond", "Clarence", "Herbert",
                     "Stanley", "Leonard", "Arthur", "Melvin", "Chester", "Vernon"],
    ("M", "middle"): ["Gregory", "Dennis", "Randall", "Curtis", "Bruce", "Alan",
                      "Terrence", "Douglas", "Roger", "Neil", "Lamar", "Vincent"],
    ("M", "younger"): ["Tyler", "Devin", "Brandon", "Jordan", "Marcus", "Trevor",
                       "Dustin", "Corey", "Andre", "Shane", "Jared", "Malik"],
}

SURNAMES = [
    "Ellison", "Whitfield", "Barnhart", "Castellano", "Okafor", "Delgado",
    "Lindqvist", "Ferraro", "Hollingsworth", "Nakamura", "Boyle", "Mensah",
    "Vasquez", "Kowalski", "Abernathy", "Rasmussen", "Guerrero", "Tran",
    "Bellamy", "Sokolov", "Ibrahim", "Prentice", "Marchetti", "Duval",
    "Yeager", "Ashford", "Cardoso", "Nyland", "Petrov", "Ramachandran",
    "Fontaine", "Stroud", "Ocampo", "Berkowitz", "Lindgren", "Achebe",
]

ADMISSION_TYPES = {
    "admission_type_ew_emer": "emergency",
    "admission_type_eu_obs": "emergency observation",
    "admission_type_obs_admit": "observation",
    "admission_type_urgent": "urgent",
    "admission_type_direct_emer": "direct emergency",
    "admission_type_ambulatory_obs": "ambulatory observation",
    "admission_type_direct_obs": "direct observation",
}

DISCHARGE_LOCATIONS = {
    "discharge_location_home": "home",
    "discharge_location_home_health": "home with services",
    "discharge_location_snf": "skilled nursing",
    "discharge_location_rehab": "rehab",
    "discharge_location_ltac": "long-term care",
    "discharge_location_hospice": "hospice",
    "discharge_location_psych": "psychiatric care",
    "discharge_location_ama": "against medical advice",
    "discharge_location_assisted_living": "assisted living",
}


def _rank(hadm_id: int, salt: str) -> int:
    """Stable pseudo-random ordering key.

    hashlib, not the builtin `hash()`: string hashing is salted per process, so
    `hash()` would reshuffle the cohort and rename every patient on each run.
    """
    return int.from_bytes(hashlib.sha256(f"{salt}:{hadm_id}".encode()).digest()[:8], "big")


def _era(age: int) -> str:
    if age >= 68:
        return "older"
    return "middle" if age >= 45 else "younger"


def _one_hot_label(row: dict, mapping: dict[str, str], default: str) -> str:
    for column, label in mapping.items():
        if row.get(column) == 1:
            return label
    return default


def _summary(row: dict) -> str:
    """Clinical descriptor, never outcome-based.

    "72F, emergency, 3 prior admissions" is a reason to click. "High risk case"
    hands over the answer the model is supposed to produce.
    """
    sex = "M" if row["gender"] == 1 else "F"
    parts = [f"{row['age']}{sex}", _one_hot_label(row, ADMISSION_TYPES, "elective") + " admission"]

    priors = int(row["prior_admission_count"])
    if priors:
        parts.append(f"{priors} prior admission{'s' if priors != 1 else ''}")
    if row.get("oncology_flag") == 1:
        parts.append("oncology history")
    if row.get("has_procedure") == 1:
        procedures = int(row["procedure_count"])
        parts.append(f"{procedures} procedure{'s' if procedures != 1 else ''}")

    parts.append(f"{row['index_los_days']:.0f}-day stay")
    discharge = _one_hot_label(row, DISCHARGE_LOCATIONS, "")
    if discharge:
        parts.append(f"discharged to {discharge}")
    return " \u00b7 ".join(parts)


def _assign_name(row: dict, used: set[str]) -> str:
    """Deterministic, sex- and era-matched, collision-free."""
    sex = "M" if row["gender"] == 1 else "F"
    firsts = FIRST_NAMES[(sex, _era(row["age"]))]
    digest = hashlib.sha256(f"name:{row['hadm_id']}".encode()).digest()
    first_i = int.from_bytes(digest[:4], "big")
    last_i = int.from_bytes(digest[4:8], "big")

    for bump in range(len(firsts) * len(SURNAMES)):
        name = (
            f"{firsts[(first_i + bump) % len(firsts)]} "
            f"{SURNAMES[(last_i + bump // len(firsts)) % len(SURNAMES)]}"
        )
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("Name pool exhausted; add more surnames.")


def _pinned_ids() -> list[int]:
    if not FIXTURE_PATH.exists():
        return []
    return [int(k) for k in json.loads(FIXTURE_PATH.read_text()).get("patients", {})]


def _score_pool(bq: bigquery.Client, pool_size: int, batch: int,
                pinned: list[int]) -> list[dict]:
    order = feature_order()
    rows = [
        dict(r)
        for r in bq.query(
            f"SELECT * FROM `{TABLE_FQN}` WHERE split_name = 'test' "
            f"ORDER BY FARM_FINGERPRINT(CAST(hadm_id AS STRING)) LIMIT {pool_size}"
        ).result()
    ]

    # Pinned patients are fetched without the split filter on purpose. The
    # fixture admission is a *validation* row, so a test-only pool can never
    # contain it and the pin would silently do nothing. Validation rows were
    # not trained on, so this does not leak memorised predictions into the
    # demo, but it is the one documented exception to the test-split rule.
    known = {r["hadm_id"] for r in rows}
    if pinned:
        extra = [
            dict(r)
            for r in bq.query(
                f"SELECT * FROM `{TABLE_FQN}` WHERE hadm_id IN UNNEST(@ids)",
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", pinned)]
                ),
            ).result()
            if r["hadm_id"] not in known
        ]
        for row in extra:
            print(f"  pinned {row['hadm_id']} added from split={row['split_name']!r}")
        rows.extend(extra)

    print(f"Scoring {len(rows)} admissions in batches of {batch}...")

    ep = endpoint()
    scored = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        preds = ep.predict(instances=[to_vector(r, order) for r in chunk]).predictions
        if len(preds) != len(chunk):
            sys.exit(f"Endpoint returned {len(preds)} predictions for {len(chunk)} instances.")
        for row, pred in zip(chunk, preds):
            row["probability"] = round(float(pred["probability"]), 6)
            row["threshold"] = float(pred["threshold"])
            scored.append(row)
    print(f"  scored {len(scored)}")
    return scored


def _select(scored: list[dict], pinned: list[int]) -> list[dict]:
    threshold = scored[0]["threshold"]
    by_id = {r["hadm_id"]: r for r in scored}
    chosen: dict[int, dict] = {i: by_id[i] for i in pinned if i in by_id}

    print(f"\nThreshold {threshold}. Selecting by band:")
    for low, high, count, label in bands(threshold):
        candidates = sorted(
            (r for r in scored
             if low <= r["probability"] < high and r["hadm_id"] not in chosen),
            key=lambda r: _rank(r["hadm_id"], label),
        )
        # Alternate sex while candidates allow, so the cohort does not end up
        # visibly lopsided once it carries names.
        picked, want_male = [], True
        while len(picked) < count and candidates:
            match = next((c for c in candidates if (c["gender"] == 1) == want_male), None)
            match = match or candidates[0]
            candidates.remove(match)
            picked.append(match)
            want_male = not want_male

        for row in picked:
            chosen[row["hadm_id"]] = row
        flag = "" if len(picked) == count else f"  (only {len(picked)} available)"
        print(f"  {label:22} [{low:.2f}, {high:.2f})  {len(picked)}{flag}")

    return sorted(chosen.values(), key=lambda r: r["probability"])


def _rebuild_table(bq: bigquery.Client, ids: list[int]) -> None:
    job = bq.query(
        f"CREATE OR REPLACE TABLE `{COHORT_TABLE_FQN}` AS "
        f"SELECT * FROM `{TABLE_FQN}` WHERE hadm_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", ids)]
        ),
    )
    job.result()
    table = bq.get_table(COHORT_TABLE_FQN)
    if table.num_rows != len(ids):
        sys.exit(f"Cohort table has {table.num_rows} rows, expected {len(ids)}.")
    print(f"\nRewrote {COHORT_TABLE_FQN}: {table.num_rows} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool-size", type=int, default=400,
                        help="test-split admissions to score before selecting (default 400)")
    parser.add_argument("--batch", type=int, default=50,
                        help="instances per endpoint request (default 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="select and print, but do not touch BigQuery or write the artifact")
    args = parser.parse_args()

    bq = bigquery.Client(project=PROJECT)
    pinned = _pinned_ids()
    print(f"Pinned from fixture: {pinned or 'none'}")

    scored = _score_pool(bq, args.pool_size, args.batch, pinned)
    cohort = _select(scored, pinned)

    used: set[str] = set()
    patients = []
    for row in cohort:
        patients.append(
            {
                "hadm_id": row["hadm_id"],
                "display_name": _assign_name(row, used),
                "age": int(row["age"]),
                "sex": "M" if row["gender"] == 1 else "F",
                "summary": _summary(row),
                "split_name": row["split_name"],
            }
        )

    print(f"\n{'hadm_id':>10} {'prob':>7} {'name':<26} {'summary'}")
    for row, patient in zip(cohort, patients):
        print(f"{patient['hadm_id']:>10} {row['probability']:>7.4f} "
              f"{patient['display_name']:<26} {patient['summary']}")

    probs = [r["probability"] for r in cohort]
    threshold = cohort[0]["threshold"]
    below = sum(1 for p in probs if p < threshold)
    print(f"\n{len(cohort)} patients | {below} below / {len(probs) - below} at or above "
          f"{threshold} | range {min(probs):.4f}-{max(probs):.4f}")
    print(f"sex: M={sum(1 for p in patients if p['sex'] == 'M')} "
          f"F={sum(1 for p in patients if p['sex'] == 'F')} | "
          f"age {min(p['age'] for p in patients)}-{max(p['age'] for p in patients)}")

    missing = set(pinned) - {p["hadm_id"] for p in patients}
    if missing:
        sys.exit(f"Fixture patients missing from cohort: {sorted(missing)}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    _rebuild_table(bq, [p["hadm_id"] for p in patients])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"patients": patients}, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(HARNESS_ROOT.parents[1])}")
    print("\nNext: re-sync the Feature Store (setup_feature_store.py --sync) — the "
          "online store still holds the previous cohort.")


if __name__ == "__main__":
    main()
