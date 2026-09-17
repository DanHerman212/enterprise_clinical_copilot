"""
load_data — query the ONE-HOT ENCODED BigQuery view and write model-ready
parquet splits, plus the serving manifest.

The heavy lifting (categorical encoding, missingness policy) now lives in a
static, leakage-free BigQuery view (``analytics_dataset_encoded``, generated
from :mod:`mlops.data.encoding`). This component is therefore a plain projection: it
selects the numeric ``feature_order`` columns, splits them into train/val/test
parquet, and emits the ``manifest.json`` serving contract (feature order +
one-hot -> parent group map). No imputer, no in-code encoding — so there is no
train/serve skew.
"""

import json
import re

import pandas as pd
from google.cloud import bigquery
from kfp import dsl

from mlops.data import encoding
from ._image import TRAINING_IMAGE, component

# ECC-63: identifiers (table ref, column names) cannot be bound as query
# parameters, so runtime pipeline params are validated against a strict shape
# instead; split-name VALUES are bound as query parameters below. Feature
# columns are not runtime inputs — they come from encoding.feature_order(),
# the code-owned manifest contract.
_TABLE_REF_RE = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validated_table_ref(ref: str) -> str:
    if not _TABLE_REF_RE.fullmatch(ref):
        raise ValueError(
            f"full_table_ref is not a valid project.dataset.table reference: {ref!r}"
        )
    return ref


def _validated_ident(name: str, param: str) -> str:
    if not _IDENT_RE.fullmatch(name):
        raise ValueError(f"{param} is not a valid column identifier: {name!r}")
    return name


def assert_patient_disjoint(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
) -> None:
    """Hard-fail if any patient's admissions straddle two splits (ECC-64).

    Everything downstream — the test AUCPR gate, the stability check, the
    fairness audit — assumes patient-level disjointness but only the upstream
    `split_name` column enforces it. If it ever regresses, every gate is
    leakage-contaminated while reporting PASS. One set intersection per pair
    is cheap insurance against that silent failure.
    """
    ids = {
        "train": set(train_df[id_col]),
        "val": set(val_df[id_col]),
        "test": set(test_df[id_col]),
    }
    leaked = {
        f"{a}/{b}": ids[a] & ids[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        if ids[a] & ids[b]
    }
    if leaked:
        detail = "; ".join(
            f"{pair}: {len(overlap)} shared {id_col}s (e.g. {sorted(overlap)[:3]})"
            for pair, overlap in leaked.items()
        )
        raise ValueError(
            f"Patient leakage across splits — {detail}. The upstream "
            f"'{id_col}' split assignment is broken; every downstream gate "
            "would be contaminated. Refusing to emit training data."
        )


def _data_reference(
    client: bigquery.Client,
    table_ref: str,
    split_rows: dict[str, int],
) -> dict:
    """What the table was when this run read it.

    Read from the table's own metadata rather than recomputed, because that is
    the only version of the answer that is true *at the moment of the read*: this
    table is rebuilt in place by Dataform under the same name, so the name alone
    says nothing about which rows a model was fitted on. Row count and
    modification time are what BigQuery already knows without a scan.

    It is not a content digest: a rebuild that changed values but kept the row
    count looks identical here. A digest is the honest next step and costs a
    second scan of the table, so it is left as a deliberate choice rather than
    smuggled in as a column of the query this component has to run anyway.
    """
    table = client.get_table(table_ref)
    return {
        "table": table_ref,
        "row_count": table.num_rows,
        "last_modified_utc": table.modified.isoformat() if table.modified else None,
        "observed_rows_by_split": split_rows,
    }


def run_load_data(
    *,
    project_id: str,
    full_table_ref: str,
    label_col: str,
    split_col: str,
    id_col: str,
    train_split: str,
    val_split: str,
    test_split: str,
    x_train_path: str,
    y_train_path: str,
    x_val_path: str,
    y_val_path: str,
    x_test_path: str,
    y_test_path: str,
    groups_train_path: str,
    groups_val_path: str,
    manifest_path: str,
) -> None:
    """Load the encoded splits, write parquet, and emit the serving manifest.

    ``full_table_ref`` must point at the ONE-HOT ENCODED view. Also emits
    ``groups_train`` (the train-split ``id_col``, e.g. subject_id) so HPO can run
    patient-grouped cross-validation without leaking a patient's admissions
    across folds.
    """
    feature_order = encoding.feature_order()

    full_table_ref = _validated_table_ref(full_table_ref)
    id_col = _validated_ident(id_col, "id_col")
    label_col = _validated_ident(label_col, "label_col")
    split_col = _validated_ident(split_col, "split_col")

    client = bigquery.Client(project=project_id)

    cols = ", ".join(feature_order)
    sql = f"""
        SELECT {id_col}, {cols}, {label_col}, {split_col}
        FROM `{full_table_ref}`
        WHERE {split_col} IN UNNEST(@splits)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "splits", "STRING", [train_split, val_split, test_split]
            )
        ]
    )
    df = client.query(sql, job_config=job_config).result().to_dataframe()

    def _split(name: str) -> pd.DataFrame:
        return df[df[split_col] == name].reset_index(drop=True)

    train_df, val_df, test_df = _split(train_split), _split(val_split), _split(test_split)
    assert_patient_disjoint(train_df, val_df, test_df, id_col)

    # Read before the splits are consumed, so the reference describes the state
    # the run actually saw rather than whatever the table holds by the time
    # registration happens.
    data_reference = _data_reference(
        client,
        full_table_ref,
        {
            train_split: len(train_df),
            val_split: len(val_df),
            test_split: len(test_df),
        },
    )

    def _xy(frame: pd.DataFrame):
        # All feature columns are already numeric; coerce to float64 so NULLs
        # arrive as NaN for XGBoost native missing handling (never nullable Int64).
        X = frame[feature_order].astype("float64")
        y = frame[label_col].astype(int)
        return X, y

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    X_train.to_parquet(x_train_path, index=False)
    pd.DataFrame(y_train).to_parquet(y_train_path, index=False)
    X_val.to_parquet(x_val_path, index=False)
    pd.DataFrame(y_val).to_parquet(y_val_path, index=False)
    X_test.to_parquet(x_test_path, index=False)
    pd.DataFrame(y_test).to_parquet(y_test_path, index=False)

    train_df[[id_col]].to_parquet(groups_train_path, index=False)
    val_df[[id_col]].to_parquet(groups_val_path, index=False)

    # Serving contract: feature order (array layout) + one-hot -> parent map for
    # aggregating Sampled Shapley attributions. Single source of truth.
    #
    # The data reference rides along in the same file (ECC-72): the manifest is
    # copied into the serving bundle and covered by its checksums, so what the
    # model was fitted on travels with the model and is verified at load rather
    # than living only in a job log.
    with open(manifest_path, "w") as f:
        json.dump({**encoding.manifest(), "data": data_reference}, f, indent=2)

    print(
        f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape} "
        f"({len(feature_order)} numeric features)"
    )
    print(
        f"  Data: {data_reference['table']} — "
        f"{data_reference['row_count']} rows, last modified "
        f"{data_reference['last_modified_utc']}"
    )


@component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["google-cloud-bigquery", "pandas>=2,<3", "pyarrow>=14,<25"],
)
def load_data(
    project_id: str,
    training_table: dsl.Input[dsl.Dataset],
    label_col: str,
    split_col: str,
    train_split: str,
    val_split: str,
    test_split: str,
    x_train: dsl.Output[dsl.Dataset],
    y_train: dsl.Output[dsl.Dataset],
    x_val: dsl.Output[dsl.Dataset],
    y_val: dsl.Output[dsl.Dataset],
    x_test: dsl.Output[dsl.Dataset],
    y_test: dsl.Output[dsl.Dataset],
    groups_train: dsl.Output[dsl.Dataset],
    groups_val: dsl.Output[dsl.Dataset],
    manifest: dsl.Output[dsl.Artifact],
    id_col: str = "subject_id",
):
    """KFP component: load the encoded splits and emit the serving manifest."""
    from mlops.training.components.load_data import run_load_data

    # The artifact is the single source of the table reference: the pipeline
    # names the table once, the importer turns that into the lineage artifact,
    # and this step reads the table back out of it rather than taking a second,
    # separate parameter that could disagree. The scheme comes off here, and
    # run_load_data validates the rest exactly as it would a direct parameter.
    run_load_data(
        project_id=project_id,
        full_table_ref=training_table.uri.removeprefix("bq://"),
        label_col=label_col, split_col=split_col, id_col=id_col,
        train_split=train_split, val_split=val_split, test_split=test_split,
        x_train_path=x_train.path, y_train_path=y_train.path,
        x_val_path=x_val.path, y_val_path=y_val.path,
        x_test_path=x_test.path, y_test_path=y_test.path,
        groups_train_path=groups_train.path,
        groups_val_path=groups_val.path,
        manifest_path=manifest.path,
    )
