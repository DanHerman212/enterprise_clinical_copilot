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
                             "base_value", "attributions", "attribution_units",
                             "top_factors"}, ...]}

Attributions are exact native TreeSHAP contributions (`pred_contribs=True` on a
`binary:logistic` booster) and are therefore in LOG-ODDS (margin) space, not
probability deltas — declared per prediction as `attribution_units`.
"""

import hashlib
import json
import math
import os

import numpy as np
import xgboost as xgb
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

TOP_K = 10
# Cap the per-request batch: attributions are O(rows × trees × features), so an
# unbounded batch is a one-request DoS on the endpoint.
MAX_BATCH = 100
ATTRIBUTION_UNITS = "log_odds"


def _verify_bundle_checksums() -> None:
    """ECC-61: refuse to serve a bundle whose files don't match their SHA-256.

    download_model_artifacts writes the bundle files into the working dir. The
    bundle's checksums.json (written by register_model.assemble_serving_bundle)
    is the expected digest of each file; a missing manifest OR any mismatch
    fails startup loudly rather than serving a tampered/corrupted artifact.
    """
    if not os.path.exists("checksums.json"):
        raise RuntimeError(
            "checksums.json missing from the serving bundle — refusing to serve "
            "an unverifiable artifact (ECC-61). Re-register the model."
        )
    with open("checksums.json") as f:
        checksums = json.load(f)
    for name, expected in checksums.items():
        if not os.path.exists(name):
            raise RuntimeError(f"serving bundle missing {name!r}")
        with open(name, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"serving bundle checksum mismatch for {name!r}: "
                f"expected {expected}, got {actual}"
            )


class ReadmissionPredictor(Predictor):
    def load(self, artifacts_uri: str) -> None:
        """Download the serving bundle (model.bst + manifest.json + threshold.json)."""
        prediction_utils.download_model_artifacts(artifacts_uri)
        _verify_bundle_checksums()

        # Which model this container is serving, stamped into the environment by
        # deploy_cpr.py from the provenance record it deployed. Returned with
        # every prediction so the answer carries the identity of the model that
        # produced it, instead of the caller inferring one from the registry.
        self._model_version = os.environ.get("MODEL_VERSION", "")
        print(f"  serving model_version: {self._model_version or '<unstamped>'}")

        with open("manifest.json") as f:
            manifest = json.load(f)
        self._feature_order: list[str] = manifest["feature_order"]
        self._feature_set = set(self._feature_order)
        self._groups: dict[str, list[str]] = manifest.get("groups", {})

        # Fail loudly — never fall back to 0.5 (ECC-68). The shipped threshold
        # is a recall-weighted F2 optimum, typically well below 0.5; a silent
        # default would flip many readmit decisions to negative with no signal.
        if not os.path.exists("threshold.json"):
            raise RuntimeError(
                "threshold.json missing from the serving bundle — the operating "
                "threshold is part of the serving contract; refusing to start "
                "with an implicit 0.5 default."
            )
        with open("threshold.json") as f:
            self._threshold = float(json.load(f)["threshold"])

        self._booster = xgb.Booster()
        self._booster.load_model("model.bst")

    def preprocess(self, prediction_input: dict) -> np.ndarray:
        """Validate + build a float32 matrix in feature order; null -> NaN.

        Every instance is validated (ECC-60): unknown dict keys are rejected
        instead of silently becoming NaN (XGBoost treats NaN as "missing" and
        would return a confident, silently wrong probability for a typo'd
        key), a dict that omits a declared feature is rejected for the same
        reason, positional lists must match the feature count, values must be
        finite numbers or null.
        """
        instances = prediction_input.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("Request must contain a non-empty 'instances' list.")
        if len(instances) > MAX_BATCH:
            raise ValueError(
                f"Batch size {len(instances)} exceeds the maximum of {MAX_BATCH}."
            )
        n = len(self._feature_order)
        rows = []
        for i, inst in enumerate(instances):
            if isinstance(inst, dict):
                unknown = set(inst) - self._feature_set
                if unknown:
                    raise ValueError(
                        f"Instance {i}: unknown feature keys "
                        f"{sorted(unknown)[:5]} — check the manifest "
                        "feature_order (a typo'd key would silently be "
                        "treated as missing)."
                    )
                # A null VALUE is a legitimate missing measurement; an ABSENT
                # KEY is a caller whose vocabulary differs from this bundle's,
                # and scoring it as missing is the silent wrong answer the
                # unknown-key check above exists to prevent.
                absent = [c for c in self._feature_order if c not in inst]
                if absent:
                    raise ValueError(
                        f"Instance {i}: missing feature keys {absent[:5]} — "
                        "send null for a value that is genuinely missing."
                    )
                raw = [inst.get(c) for c in self._feature_order]
            elif isinstance(inst, (list, tuple)):
                if len(inst) != n:
                    raise ValueError(
                        f"Instance {i}: expected {n} values in feature order, "
                        f"got {len(inst)}."
                    )
                raw = list(inst)
            else:
                raise ValueError(
                    f"Instance {i}: must be a dict of named features or a "
                    f"list of {n} values."
                )
            row = []
            for name, v in zip(self._feature_order, raw):
                if v is None:
                    row.append(np.nan)
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Instance {i}, feature '{name}': expected a number "
                        f"or null, got {type(v).__name__}."
                    )
                value = float(v)
                if not math.isfinite(value):
                    raise ValueError(
                        f"Instance {i}, feature '{name}': non-finite values "
                        "are not accepted; send null for missing."
                    )
                row.append(value)
            rows.append(row)
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
                    # TreeSHAP on binary:logistic is margin-space (ECC-73) —
                    # these are NOT probability deltas.
                    "attribution_units": ATTRIBUTION_UNITS,
                    "model_version": self._model_version,
                    "top_factors": [
                        {"feature": name, "attribution": val} for name, val in top
                    ],
                }
            )
        return {"predictions": predictions}
