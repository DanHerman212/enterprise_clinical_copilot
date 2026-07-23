"""
ReadmissionPredictor — Vertex AI Custom Prediction Routine (CPR).

Serves the native XGBoost booster behind a Vertex AI endpoint and returns, in a
single response, the calibrated probability, the operating-threshold decision,
and exact native-TreeSHAP feature attributions aggregated to parent features.

Because the routine parses the request itself, it sidesteps the pre-built
container's JSON limitations: missing values may be sent as JSON ``null`` (→ NaN,
XGBoost's native missing), and instances may be positional lists or named dicts.

Request  : {"instances": [[f0, f1, ... f48], ...]}  (null allowed for missing)
        or {"instances": [{"age": 90, "sodium_min": null, ...}, ...]}
Response : {"predictions": [{"probability", "prediction", "threshold",
                             "base_value", "attributions", "top_factors"}, ...]}
"""

import json
import os

import numpy as np
import xgboost as xgb
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

TOP_K = 10


class ReadmissionPredictor(Predictor):
    def load(self, artifacts_uri: str) -> None:
        """Download the serving bundle (model.bst + manifest.json [+ threshold.json])."""
        prediction_utils.download_model_artifacts(artifacts_uri)

        with open("manifest.json") as f:
            manifest = json.load(f)
        self._feature_order: list[str] = manifest["feature_order"]
        self._groups: dict[str, list[str]] = manifest.get("groups", {})

        self._threshold = 0.5
        if os.path.exists("threshold.json"):
            with open("threshold.json") as f:
                self._threshold = float(json.load(f).get("threshold", 0.5))

        self._booster = xgb.Booster()
        self._booster.load_model("model.bst")

    def preprocess(self, prediction_input: dict) -> np.ndarray:
        """Build a float32 matrix in feature order; JSON null -> NaN (missing)."""
        instances = prediction_input["instances"]
        rows = []
        for inst in instances:
            if isinstance(inst, dict):
                raw = [inst.get(c) for c in self._feature_order]
            else:
                raw = list(inst)
            rows.append([np.nan if v is None else float(v) for v in raw])
        return np.asarray(rows, dtype=np.float32)

    def predict(self, instances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dm = xgb.DMatrix(instances, feature_names=self._feature_order)
        probs = self._booster.predict(dm)
        contribs = self._booster.predict(dm, pred_contribs=True)
        return probs, contribs

    def postprocess(self, prediction_results: tuple[np.ndarray, np.ndarray]) -> dict:
        probs, contribs = prediction_results
        predictions = []
        for i in range(len(probs)):
            row = contribs[i]
            base = float(row[-1])
            by_index = {name: float(row[j]) for j, name in enumerate(self._feature_order)}
            if self._groups:
                attributions = {
                    parent: sum(by_index.get(col, 0.0) for col in cols)
                    for parent, cols in self._groups.items()
                }
            else:
                attributions = by_index
            top = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_K]
            prob = float(probs[i])
            predictions.append(
                {
                    "probability": prob,
                    "prediction": int(prob >= self._threshold),
                    "threshold": self._threshold,
                    "base_value": base,
                    "attributions": attributions,
                    "top_factors": [
                        {"feature": name, "attribution": val} for name, val in top
                    ],
                }
            )
        return {"predictions": predictions}
