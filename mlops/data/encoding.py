"""
encoding — single source of truth for the static, leakage-free feature encoding.

The decoupled serving pattern (pre-built XGBoost container + Vertex Explainable
AI) requires the model to consume a fixed-order **numeric** vector. All
categorical encoding is therefore pushed out of Python and into a static
BigQuery view (``definitions/marts/analytics_dataset_encoded.sqlx``) that is
GENERATED from the definitions in this module, so three things can never drift
apart:

  1. the BigQuery one-hot SQL,
  2. the training/serving feature order (the array layout the model consumes),
  3. the one-hot -> parent ``groups`` map used to aggregate Sampled Shapley
     attributions back to human-readable parent features.

Design (see the locked serving-architecture decisions):
  * Numerics pass through unchanged — NULLs preserved for XGBoost native missing
    handling. No imputation, so no train-derived statistic ever leaves the DB
    (zero train/serve skew).
  * Binary flags (gender, has_procedure, oncology_flag) stay single 0/1 columns.
  * Multi-level categoricals are one-hot encoded with an explicit column per
    known level PLUS a ``<parent>_unknown`` catch-all that captures NULL and any
    level unseen at authoring time (leak-free and robust to new raw levels).
  * ``race`` is bucketed into OMB groups (mirrors
    ``pipelines.components.fairness_audit._omb_race``) both to tame cardinality
    and to align one-hot columns with the fairness slices.

Regenerate the view after editing this module::

    python -m src.encoding --emit-sql   > /tmp/encoded_select.sql
    python -m src.encoding --emit-order         # sanity-check feature_order
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Passthrough identity / split / label columns (not features)
# ---------------------------------------------------------------------------
ID_COLUMNS = ["subject_id", "hadm_id"]
SPLIT_COLUMN = "split_name"
LABEL_COLUMN = "readmission_30d"
PASSTHROUGH_COLUMNS = ID_COLUMNS + [SPLIT_COLUMN, LABEL_COLUMN]

# ---------------------------------------------------------------------------
# Numeric features — passed through unchanged, NULLs preserved
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age",
    "prior_admission_count",
    "prior_inpatient_days",
    "recent_ed_visits",
    "index_los_days",
    "procedure_count",
    "medication_count",
    "medication_order_count",
    "rbc_last",
    "rbc_min",
    "rdw_max",
    "monocytes_min",
    "hemoglobin_min",
    "sodium_last",
    "sodium_max",
    "sodium_min",
]

# ---------------------------------------------------------------------------
# Binary features — single 0/1 column each (parent name == output column)
# Each maps to a SQL expression producing INT64 in {0, 1}.
# ---------------------------------------------------------------------------
BINARY_FEATURES: dict[str, str] = {
    # gender is raw 'M'/'F'; encode as 1 iff male. Single column keeps the
    # feature interpretable and its Shapley value maps straight to "gender".
    "gender": "CAST(gender = 'M' AS INT64)",
    "has_procedure": "CAST(has_procedure AS INT64)",
    "oncology_flag": "CAST(oncology_flag AS INT64)",
}

# ---------------------------------------------------------------------------
# Multi-level categoricals — one-hot with an explicit column per known level
# plus a `<parent>_unknown` catch-all (NULL or any unseen raw level).
# ---------------------------------------------------------------------------

# race is bucketed first (raw MIMIC values -> OMB group), then one-hot on the
# bucket. Order mirrors fairness_audit._omb_race (Hispanic checked before
# Black/White). The 'unknown' bucket IS the catch-all (Other/Unknown + NULL).
RACE_BUCKETS: list[tuple[str, str]] = [
    # (bucket_slug, human_label)
    ("white", "White"),
    ("black", "Black or African American"),
    ("hispanic", "Hispanic or Latino"),
    ("asian", "Asian"),
    ("amind", "American Indian or Alaska Native"),
    ("nhpi", "Native Hawaiian or Pacific Islander"),
]

# Direct value-match categoricals: (raw_value, column_slug).
ONEHOT_DIRECT: dict[str, list[tuple[str, str]]] = {
    "admission_type": [
        ("EW EMER.", "ew_emer"),
        ("EU OBSERVATION", "eu_obs"),
        ("OBSERVATION ADMIT", "obs_admit"),
        ("URGENT", "urgent"),
        ("DIRECT EMER.", "direct_emer"),
        ("AMBULATORY OBSERVATION", "ambulatory_obs"),
        ("DIRECT OBSERVATION", "direct_obs"),
    ],
    "discharge_location": [
        ("HOME", "home"),
        ("HOME HEALTH CARE", "home_health"),
        ("SKILLED NURSING FACILITY", "snf"),
        ("REHAB", "rehab"),
        ("CHRONIC/LONG TERM ACUTE CARE", "ltac"),
        ("HOSPICE", "hospice"),
        ("AGAINST ADVICE", "ama"),
        ("PSYCH FACILITY", "psych"),
        ("ASSISTED LIVING", "assisted_living"),
    ],
    "insurance": [
        ("Medicare", "medicare"),
        ("Medicaid", "medicaid"),
        ("Private", "private"),
        ("Other", "other"),
    ],
}

# Parent order for one-hot blocks in the feature vector (race first).
ONEHOT_PARENTS = ["race"] + list(ONEHOT_DIRECT.keys())


def _sql_str(value: str) -> str:
    """SQL string literal with single quotes escaped."""
    return "'" + value.replace("'", "''") + "'"


def race_bucket_sql(col: str = "race") -> str:
    """SQL CASE expression mapping a raw race string to an OMB bucket slug.

    Mirrors ``fairness_audit._omb_race`` exactly (same precedence) so the
    one-hot columns line up with the fairness audit slices. Anything that does
    not match a defined bucket (incl. NULL / 'OTHER' / 'UNKNOWN') falls to
    ``'unknown'``.
    """
    u = f"UPPER({col})"
    return (
        "CASE\n"
        f"    WHEN {u} LIKE '%HISPANIC%' OR {u} LIKE '%LATINO%' "
        f"OR {u} LIKE '%SOUTH AMERICAN%' THEN 'hispanic'\n"
        f"    WHEN {u} LIKE '%BLACK%' OR {u} LIKE '%AFRICAN%' THEN 'black'\n"
        f"    WHEN {u} LIKE '%ASIAN%' THEN 'asian'\n"
        f"    WHEN {u} LIKE '%AMERICAN INDIAN%' OR {u} LIKE '%ALASKA NATIVE%' "
        "THEN 'amind'\n"
        f"    WHEN {u} LIKE '%NATIVE HAWAIIAN%' OR {u} LIKE '%PACIFIC ISLANDER%' "
        "THEN 'nhpi'\n"
        f"    WHEN {u} LIKE '%WHITE%' THEN 'white'\n"
        "    ELSE 'unknown'\n"
        "  END"
    )


def _onehot_columns(parent: str) -> list[str]:
    """Ordered one-hot output column names for one categorical parent."""
    if parent == "race":
        cols = [f"race_{slug}" for slug, _ in RACE_BUCKETS]
    else:
        cols = [f"{parent}_{slug}" for _, slug in ONEHOT_DIRECT[parent]]
    return cols + [f"{parent}_unknown"]


def feature_order() -> list[str]:
    """The exact numeric column layout the model consumes (array order).

    numerics -> binaries -> one-hot blocks (race, admission_type,
    discharge_location, insurance).
    """
    order = list(NUMERIC_FEATURES)
    order += list(BINARY_FEATURES.keys())
    for parent in ONEHOT_PARENTS:
        order += _onehot_columns(parent)
    return order


def groups() -> dict[str, list[str]]:
    """Parent feature -> list of its column(s) in ``feature_order``.

    Numerics and binaries are singletons; each categorical maps to its one-hot
    block. Sampled Shapley attributions are aggregated to the parent by summing
    the attributions of the columns in each group (valid by Shapley
    additivity).
    """
    g: dict[str, list[str]] = {name: [name] for name in NUMERIC_FEATURES}
    g.update({name: [name] for name in BINARY_FEATURES})
    for parent in ONEHOT_PARENTS:
        g[parent] = _onehot_columns(parent)
    return g


def group_map() -> dict[str, str]:
    """Inverse of ``groups``: each column -> its parent feature."""
    return {col: parent for parent, cols in groups().items() for col in cols}


# Name of the intermediate CTE column holding the race bucket slug.
RACE_BUCKET_COL = "race_bucket"


def select_expressions() -> list[str]:
    """SQL ``<expr> AS <col>`` list, in ``feature_order`` (features only).

    race one-hot columns reference the ``race_bucket`` column produced by the
    ``bucketed`` CTE in :func:`build_select_sql` (computed once, not inlined).
    """
    exprs: list[str] = []

    # Numerics pass through unchanged (NULL preserved).
    for col in NUMERIC_FEATURES:
        exprs.append(f"{col} AS {col}")

    # Binary flags.
    for col, expr in BINARY_FEATURES.items():
        exprs.append(f"{expr} AS {col}")

    # race — one-hot on the pre-computed bucket column.
    for slug, _label in RACE_BUCKETS:
        exprs.append(
            f"CAST({RACE_BUCKET_COL} = '{slug}' AS INT64) AS race_{slug}"
        )
    exprs.append(
        f"CAST({RACE_BUCKET_COL} = 'unknown' AS INT64) AS race_unknown"
    )

    # Direct-match categoricals — one-hot per known level + unknown catch-all.
    for parent, levels in ONEHOT_DIRECT.items():
        known_literals = ", ".join(_sql_str(v) for v, _ in levels)
        for raw_value, slug in levels:
            exprs.append(
                f"CAST({parent} = {_sql_str(raw_value)} AS INT64) "
                f"AS {parent}_{slug}"
            )
        # unknown = NULL or any level not in the known set.
        exprs.append(
            f"CAST(({parent} IS NULL OR {parent} NOT IN ({known_literals})) "
            f"AS INT64) AS {parent}_unknown"
        )

    return exprs


def build_select_sql(source: str = '${ref("analytics_dataset")}') -> str:
    """Full query for the encoded view (race bucket computed once in a CTE).

    ``source`` defaults to the Dataform ``ref()`` for the analytics dataset.
    """
    cte = (
        "WITH bucketed AS (\n"
        "  SELECT\n"
        "    *,\n"
        f"    {race_bucket_sql('race')} AS {RACE_BUCKET_COL}\n"
        f"  FROM {source}\n"
        ")"
    )
    lines = [f"  {ID_COLUMNS[0]},", f"  {ID_COLUMNS[1]},"]
    lines += [f"  {SPLIT_COLUMN},"]
    for expr in select_expressions():
        lines.append(f"  {expr},")
    lines.append(f"  {LABEL_COLUMN}")
    body = "\n".join(lines)
    return f"{cte}\nSELECT\n{body}\nFROM bucketed"


def manifest() -> dict[str, object]:
    """Serving contract: feature order + one-hot -> parent group map."""
    return {"feature_order": feature_order(), "groups": groups()}


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Feature encoding source of truth.")
    parser.add_argument("--emit-sql", action="store_true", help="Print SELECT body.")
    parser.add_argument("--emit-order", action="store_true", help="Print feature_order.")
    parser.add_argument("--emit-manifest", action="store_true", help="Print manifest JSON.")
    args = parser.parse_args()

    if args.emit_sql:
        print(build_select_sql())
    if args.emit_order:
        for i, name in enumerate(feature_order()):
            print(f"{i:3d}  {name}")
        print(f"\n# {len(feature_order())} columns")
    if args.emit_manifest:
        print(json.dumps(manifest(), indent=2))
