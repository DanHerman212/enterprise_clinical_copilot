#!/usr/bin/env python
"""Verify the loaded hybrid_features_v2 table."""
from google.cloud import bigquery

PROJECT = "trim-icon-498815-a0"
T = f"{PROJECT}.readmission.hybrid_features_v2"

c = bigquery.Client(project=PROJECT)

n = list(c.query(f"SELECT COUNT(*) AS n FROM `{T}`").result())[0]["n"]
print(f"rows: {n}")

pos = list(c.query(f"SELECT SUM(readmission_30d) AS p FROM `{T}`").result())[0]["p"]
print(f"positives: {int(pos)}/{n} ({pos / n * 100:.1f}%)")

print("\n-- numeric features (min / median / max / null-count) --")
for col in [
    "age", "index_los_days", "medication_count", "prior_admission_count",
    "rbc_min", "rdw_max", "hemoglobin_min", "sodium_min", "monocytes_min",
    "medication_order_count", "procedure_count", "prior_inpatient_days",
    "recent_ed_visits",
]:
    row = list(c.query(
        f"SELECT MIN({col}) AS mn, MAX({col}) AS mx, "
        f"COUNTIF({col} IS NULL) AS nn, "
        f"ROUND(APPROX_QUANTILES({col}, 100)[OFFSET(50)], 3) AS med "
        f"FROM `{T}`"
    ).result())[0]
    mn = row["mn"]
    mx = row["mx"]
    med = row["med"]
    nn = row["nn"]
    print(f"{col:26s} min={str(mn):>8} med={str(med):>8} max={str(mx):>8} null={nn}")

print("\n-- categorical coverage (sum / n) --")
for col in [
    "race_white", "race_black", "race_hispanic", "race_asian", "race_unknown",
    "insurance_medicare", "insurance_medicaid", "insurance_private",
    "admission_type_ew_emer", "admission_type_urgent", "admission_type_direct_emer",
    "discharge_location_home", "discharge_location_home_health",
    "discharge_location_snf", "discharge_location_rehab",
    "oncology_flag", "gender",
]:
    row = list(c.query(
        f"SELECT ROUND(SUM({col}), 1) AS s, COUNT(*) AS n FROM `{T}`"
    ).result())[0]
    print(f"{col:30s} {row['s']:>5}/{row['n']} ({row['s'] / row['n'] * 100:.0f}%)")

print("\n-- history fields (donor-imputed) --")
for col in ["prior_admission_count", "prior_inpatient_days", "recent_ed_visits"]:
    row = list(c.query(
        f"SELECT MIN({col}) AS mn, MAX({col}) AS mx, "
        f"ROUND(APPROX_QUANTILES({col}, 100)[OFFSET(50)], 2) AS med, "
        f"COUNTIF({col} > 0) AS gt0 FROM `{T}`"
    ).result())[0]
    print(f"{col:26s} min={row['mn']} med={row['med']} max={row['mx']} >0={row['gt0']}/{n}")
