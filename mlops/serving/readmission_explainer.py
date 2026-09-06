"""
readmission_explainer — native TreeSHAP serving glue for the readmission model.

The Vertex AI endpoint (pre-built XGBoost container) returns a calibrated
probability. Feature attributions are computed here, client-side, with XGBoost's
native TreeSHAP (``booster.predict(..., pred_contribs=True)``) — exact (not
sampled), low-latency, and free of any managed-XAI platform dependency.

One-hot column contributions are aggregated back to parent features using the
``manifest.json`` groups; this is valid because SHAP values are additive.

Consumed by:
  - ``scripts/smoke_test.py`` (endpoint prediction + local explanation)
  - the agent harness ``predict_readmission`` tool (same glue)

Note on parity: attributions are in margin (logit) space, matching XGBoost's
standard SHAP for ``objective=binary:logistic``. The sign indicates the
direction of risk; the magnitude is the additive logit contribution.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import xgboost as xgb


@dataclass
class Explanation:
    probability: float
    base_value: float
    attributions: dict[str, float]  # parent feature -> logit contribution

    def top(self, k: int = 10) -> list[tuple[str, float]]:
        return sorted(
            self.attributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:k]


class ReadmissionExplainer:
    """Loads a serving bundle (model.bst + manifest.json) and explains predictions."""

    def __init__(self, model_path: str, manifest: dict):
        self.feature_order: list[str] = manifest["feature_order"]
        self.groups: dict[str, list[str]] = manifest.get("groups", {})
        self.booster = xgb.Booster()
        self.booster.load_model(model_path)

    # -- constructors --------------------------------------------------------
    @classmethod
    def from_local(cls, model_path: str, manifest_path: str) -> "ReadmissionExplainer":
        with open(manifest_path) as f:
            return cls(model_path, json.load(f))

    @classmethod
    def from_gcs(
        cls, bundle_uri: str, cache_dir: str | None = None
    ) -> "ReadmissionExplainer":
        """Download model.bst + manifest.json from a GCS serving bundle (cached)."""
        bundle_uri = bundle_uri.rstrip("/")
        cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "readmission_bundle"
        )
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "model.bst")
        manifest_path = os.path.join(cache_dir, "manifest.json")
        for name, dest in (("model.bst", model_path), ("manifest.json", manifest_path)):
            if not os.path.exists(dest):
                subprocess.check_call(
                    ["gsutil", "-q", "cp", f"{bundle_uri}/{name}", dest]
                )
        return cls.from_local(model_path, manifest_path)

    # -- inference -----------------------------------------------------------
    def _dmatrix(self, features: list[float]) -> xgb.DMatrix:
        row = np.asarray([features], dtype=np.float32)
        return xgb.DMatrix(row, feature_names=self.feature_order)

    def predict_proba(self, features: list[float]) -> float:
        return float(self.booster.predict(self._dmatrix(features))[0])

    def explain(self, features: list[float]) -> Explanation:
        dm = self._dmatrix(features)
        prob = float(self.booster.predict(dm)[0])
        # pred_contribs -> [n_features + 1]; last element is the bias/base value.
        contribs = self.booster.predict(dm, pred_contribs=True)[0]
        base = float(contribs[-1])
        by_index = {name: float(contribs[i]) for i, name in enumerate(self.feature_order)}

        if self.groups:
            parent = {
                parent_name: sum(by_index.get(col, 0.0) for col in cols)
                for parent_name, cols in self.groups.items()
            }
        else:
            parent = by_index

        return Explanation(probability=prob, base_value=base, attributions=parent)
