"""Phase D — TFDV batch validation runner.

Generates per-split statistics, infers a schema from TRAIN, validates VAL
and TEST against that schema, and writes all artifacts (protos + Facets
HTML + a JSON summary) to ``$OUT_DIR``. Designed to run inside the
``docker/validation`` image; see ``scripts/run_validation.sh`` for the
host-side wrapper.

Inputs (env vars):
    BQ_PROJECT       GCP project for the BigQuery client (billing).
    FEATURES_TABLE   Fully-qualified `project.dataset.table` for cohort_features.
    OUT_DIR          Directory inside the container where artifacts are written
                     (bind-mounted to artifacts/validation/ on the host).

Outputs written to ``$OUT_DIR``:
    train_stats.pb, val_stats.pb, test_stats.pb           — DatasetFeatureStatisticsList protos
    schema.pbtxt                                          — schema inferred from TRAIN
    val_anomalies.pbtxt, test_anomalies.pbtxt             — anomaly reports
    train_vs_val.html, train_vs_test.html, all_splits.html — Facets stats viewers
    summary.json                                          — machine-readable rollup
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import tensorflow_data_validation as tfdv
from google.cloud import bigquery
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import anomalies_pb2, schema_pb2


# Columns that are identifiers, the label, or the split itself. They should
# not be validated as features.
EXCLUDE_COLS = {"subject_id", "hadm_id", "label", "split"}


def load_split(client: bigquery.Client, table: str, split: str) -> pd.DataFrame:
    """Pull one split of cohort_features into a pandas DataFrame."""
    sql = f"SELECT * EXCEPT(subject_id, hadm_id) FROM `{table}` WHERE split = @split"
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("split", "STRING", split)]
        ),
    )
    df = job.result().to_dataframe(create_bqstorage_client=True)
    # Drop the split column itself before stats generation.
    return df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])


def write_facets_html(out_path: Path, lhs_name: str, lhs, rhs_name=None, rhs=None) -> None:
    """Render a Facets Overview HTML page comparing one or two stats sets."""
    # tfdv.utils.display_util builds the Facets HTML used by the notebook
    # widget. We render it standalone so the host notebook can iframe it.
    from tensorflow_data_validation.utils import display_util

    html = display_util.get_statistics_html(
        lhs_statistics=lhs,
        lhs_name=lhs_name,
        rhs_statistics=rhs,
        rhs_name=rhs_name,
    )
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    project = os.environ["BQ_PROJECT"]
    table = os.environ["FEATURES_TABLE"]
    out_dir = Path(os.environ.get("OUT_DIR", "/artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_validation] project={project}")
    print(f"[run_validation] table={table}")
    print(f"[run_validation] out_dir={out_dir}")

    client = bigquery.Client(project=project)

    # 1. Load each split.
    splits = {}
    for name in ("train", "val", "test"):
        print(f"[run_validation] loading split={name} ...")
        splits[name] = load_split(client, table, name)
        print(f"[run_validation]   shape={splits[name].shape}")

    # 2. Generate statistics per split.
    stats = {}
    for name, df in splits.items():
        print(f"[run_validation] generating statistics for {name} ...")
        stats[name] = tfdv.generate_statistics_from_dataframe(df)
        tfdv.write_stats_text(stats[name], str(out_dir / f"{name}_stats.pb"))

    # 3. Infer schema from TRAIN only.
    print("[run_validation] inferring schema from train ...")
    schema = tfdv.infer_schema(statistics=stats["train"])
    tfdv.write_schema_text(schema, str(out_dir / "schema.pbtxt"))

    # 4. Validate VAL and TEST against the TRAIN schema.
    anomaly_counts = {}
    for name in ("val", "test"):
        print(f"[run_validation] validating {name} against train schema ...")
        anomalies = tfdv.validate_statistics(statistics=stats[name], schema=schema)
        (out_dir / f"{name}_anomalies.pbtxt").write_text(
            text_format.MessageToString(anomalies), encoding="utf-8"
        )
        anomaly_counts[name] = len(anomalies.anomaly_info)
        print(f"[run_validation]   {name}: {anomaly_counts[name]} anomalies")

    # 5. Render Facets HTML for interactive inspection in the host notebook.
    print("[run_validation] rendering Facets HTML ...")
    write_facets_html(out_dir / "train_vs_val.html", "train", stats["train"], "val", stats["val"])
    write_facets_html(out_dir / "train_vs_test.html", "train", stats["train"], "test", stats["test"])
    write_facets_html(out_dir / "train_only.html", "train", stats["train"])

    # 6. Machine-readable summary.
    summary = {
        "features_table": table,
        "row_counts": {name: int(df.shape[0]) for name, df in splits.items()},
        "n_columns": int(splits["train"].shape[1]),
        "schema_features": [f.name for f in schema.feature],
        "anomaly_counts": anomaly_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[run_validation] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
