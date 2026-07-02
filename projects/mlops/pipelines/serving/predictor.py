"""
predictor — real-time inference for the readmission model (Option 2).

The trained XGBoost model uses native categorical features, so predictions are
only valid if the online request is encoded with the *exact* same schema used
in training. This predictor guarantees that by reusing the single shared
``encode_frame`` function and the fitted imputer, plus the persisted schema
(feature order + ordered category levels).

Serving bundle layout (a GCS directory)::

    model.joblib     # fitted XGBClassifier (sklearn wrapper)
    imputer.joblib   # MissingnessImputer fit on train
    schema.json      # {"feature_order": [...], "cat_categories": {col: [levels]}}

``Predictor`` follows the Vertex AI Custom Prediction Routine interface
(``load`` + ``predict``) but has no hard dependency on the CPR base class, so
it stays unit-testable.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd

from pipelines.components.data import encode_frame


def predict_from_records(
    records: list[dict],
    *,
    imputer,
    schema: dict,
    model,
) -> list[float]:
    """Return P(readmission) for raw feature records, encoded like training."""
    df = pd.DataFrame(records)
    imputed = imputer.transform(df)
    X = encode_frame(
        imputed,
        feature_order=schema["feature_order"],
        cat_categories=schema["cat_categories"],
    )
    return model.predict_proba(X)[:, 1].tolist()


class Predictor:
    """Vertex-CPR-style predictor: ``load`` a serving bundle, then ``predict``."""

    def __init__(self) -> None:
        self._model = None
        self._imputer = None
        self._schema: dict | None = None

    def load(self, artifacts_dir: str) -> None:
        """Load model, imputer, and schema from a serving-bundle directory."""
        self._model = joblib.load(os.path.join(artifacts_dir, "model.joblib"))
        self._imputer = joblib.load(os.path.join(artifacts_dir, "imputer.joblib"))
        with open(os.path.join(artifacts_dir, "schema.json")) as f:
            self._schema = json.load(f)

    def predict(self, instances: list[dict]) -> list[float]:
        """Score a batch of raw feature records."""
        if self._model is None:
            raise RuntimeError("Predictor.load() must be called before predict().")
        return predict_from_records(
            instances,
            imputer=self._imputer,
            schema=self._schema,
            model=self._model,
        )
