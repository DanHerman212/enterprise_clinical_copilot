"""Tests for the real-time serving predictor (pipeline module 7, Option 2).

The one property that matters: the serving path reproduces the TRAINING
encoding exactly, so online predictions equal what the model would produce on
the training-encoded frame. Also covers artifact round-trip and leakage-free
handling of categories unseen at training time.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from pipelines.components.data import fit_imputer, prepare_splits
from pipelines.serving.predictor import Predictor, predict_from_records

LABEL = "readmission_30d"


def _train(train_df, val_df, test_df, selected_features, cat_features):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer,
        selected_features=selected_features,
        cat_features=cat_features,
        label_col=LABEL,
    )
    model = XGBClassifier(n_estimators=8, max_depth=3, enable_categorical=True,
                          tree_method="hist", random_state=42)
    model.fit(out["X_train"], out["y_train"])
    schema = {"feature_order": out["feature_order"], "cat_categories": out["cat_categories"]}
    return imputer, model, schema, out


def test_serving_matches_training_encoding(
    train_df, val_df, test_df, selected_features, cat_features
):
    imputer, model, schema, out = _train(
        train_df, val_df, test_df, selected_features, cat_features
    )
    # Raw test records (as an online caller would send them — no label).
    records = test_df.drop(columns=[LABEL]).to_dict(orient="records")

    served = predict_from_records(records, imputer=imputer, schema=schema, model=model)
    expected = model.predict_proba(out["X_test"])[:, 1]

    assert np.allclose(served, expected)


def test_predictor_artifact_roundtrip(
    tmp_path, train_df, val_df, test_df, selected_features, cat_features
):
    imputer, model, schema, out = _train(
        train_df, val_df, test_df, selected_features, cat_features
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    joblib.dump(model, bundle / "model.joblib")
    joblib.dump(imputer, bundle / "imputer.joblib")
    (bundle / "schema.json").write_text(json.dumps(schema))

    predictor = Predictor()
    predictor.load(str(bundle))
    records = test_df.drop(columns=[LABEL]).to_dict(orient="records")
    preds = predictor.predict(records)

    expected = model.predict_proba(out["X_test"])[:, 1]
    assert np.allclose(preds, expected)


def test_unseen_category_does_not_crash(
    train_df, val_df, test_df, selected_features, cat_features
):
    imputer, model, schema, _ = _train(
        train_df, val_df, test_df, selected_features, cat_features
    )
    # gender "U" never appeared in train -> must encode to NaN, still predict.
    record = {
        "age": 61, "discharge_location": "HOME", "insurance": "Medicare",
        "gender": "U", "glucose_last": 101.0,
    }
    preds = predict_from_records([record], imputer=imputer, schema=schema, model=model)
    assert len(preds) == 1
    assert np.isfinite(preds[0])


def test_assemble_serving_bundle(tmp_path):
    from pipelines.components.register_model import assemble_serving_bundle

    model_src = tmp_path / "m.joblib"
    imputer_src = tmp_path / "i.joblib"
    schema_src = tmp_path / "s.json"
    model_src.write_text("MODEL")
    imputer_src.write_text("IMPUTER")
    schema_src.write_text('{"feature_order": [], "cat_categories": {}}')

    bundle = tmp_path / "bundle"
    assemble_serving_bundle(
        model_path=str(model_src),
        imputer_path=str(imputer_src),
        schema_path=str(schema_src),
        bundle_dir=str(bundle),
    )
    assert (bundle / "model.joblib").read_text() == "MODEL"
    assert (bundle / "imputer.joblib").read_text() == "IMPUTER"
    assert (bundle / "schema.json").exists()
