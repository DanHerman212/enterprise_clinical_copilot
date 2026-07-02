"""Tests for the custom serving container's HTTP app (module 7 serving infra).

Exercises the Vertex custom-container contract: a health route returning 200
and a predict route that accepts {"instances": [...]} and returns
{"predictions": [...]}, loading the serving bundle from MODEL_DIR / AIP_STORAGE_URI.
"""

import json
import os
import warnings

import joblib
import numpy as np
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from pipelines.components.data import fit_imputer, prepare_splits

LABEL = "readmission_30d"


def _build_bundle(bundle_dir, train_df, val_df, test_df, selected_features, cat_features):
    imputer = fit_imputer(train_df)
    out = prepare_splits(
        train_df, val_df, test_df,
        imputer=imputer, selected_features=selected_features,
        cat_features=cat_features, label_col=LABEL,
    )
    model = XGBClassifier(n_estimators=8, max_depth=3, enable_categorical=True,
                          tree_method="hist", random_state=42)
    model.fit(out["X_train"], out["y_train"])
    schema = {"feature_order": out["feature_order"], "cat_categories": out["cat_categories"]}

    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, bundle_dir / "model.joblib")
    joblib.dump(imputer, bundle_dir / "imputer.joblib")
    (bundle_dir / "schema.json").write_text(json.dumps(schema))
    return out


def test_health_and_predict(
    tmp_path, monkeypatch, train_df, val_df, test_df, selected_features, cat_features
):
    bundle = tmp_path / "bundle"
    out = _build_bundle(
        bundle, train_df, val_df, test_df, selected_features, cat_features
    )
    monkeypatch.setenv("MODEL_DIR", str(bundle))

    from pipelines.serving.app import app

    records = test_df.drop(columns=[LABEL]).to_dict(orient="records")
    # Missing values arrive over JSON as null, not NaN.
    records = [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        for r in records
    ]
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

        resp = client.post("/predict", json={"instances": records})
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert len(preds) == len(records)

        # Served predictions must match the in-process model on encoded data.
        model = joblib.load(bundle / "model.joblib")
        expected = model.predict_proba(out["X_test"])[:, 1]
        assert np.allclose(preds, expected)
