"""Tests for Cluster L gate-integrity fixes.

Pins:
  * ECC-64: run_load_data hard-fails when a patient's admissions straddle two
    splits (train/val/test subject-id disjointness);
  * ECC-60: the CPR predictor validates every instance — unknown dict keys,
    wrong-length lists, non-finite/non-numeric values, and oversized batches
    are rejected instead of silently becoming NaN;
  * ECC-68: the CPR predictor refuses to start without threshold.json;
  * ECC-73: attributions declare their units (log-odds).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.components.load_data import assert_patient_disjoint


# ---------------------------------------------------------------------------
# ECC-64 — patient disjointness across splits
# ---------------------------------------------------------------------------

def _frame(ids):
    return pd.DataFrame({"subject_id": ids, "age": [50] * len(ids)})


def test_disjoint_splits_pass():
    assert_patient_disjoint(
        _frame([1, 2]), _frame([3]), _frame([4, 5]), "subject_id"
    )


def test_train_test_overlap_hard_fails():
    with pytest.raises(ValueError, match="Patient leakage"):
        assert_patient_disjoint(
            _frame([1, 2]), _frame([3]), _frame([2, 4]), "subject_id"
        )


def test_val_test_overlap_hard_fails():
    with pytest.raises(ValueError, match="val/test"):
        assert_patient_disjoint(
            _frame([1]), _frame([2, 3]), _frame([3]), "subject_id"
        )


# ---------------------------------------------------------------------------
# CPR predictor (ECC-60, ECC-68, ECC-73) — loaded from file, no package init
# ---------------------------------------------------------------------------

pytest.importorskip("google.cloud.aiplatform.prediction")

_PREDICTOR_PATH = (
    Path(__file__).resolve().parents[1] / "serving" / "cpr" / "predictor.py"
)
_spec = importlib.util.spec_from_file_location("cpr_predictor", _PREDICTOR_PATH)
cpr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpr)

_FEATURES = ["age", "sodium_min", "glucose_last"]


def _predictor(threshold=0.2):
    p = cpr.ReadmissionPredictor.__new__(cpr.ReadmissionPredictor)
    p._feature_order = list(_FEATURES)
    p._feature_set = set(_FEATURES)
    p._groups = {}
    p._threshold = threshold
    return p


def test_valid_dict_and_list_instances_build_the_matrix():
    p = _predictor()
    out = p.preprocess({"instances": [
        {"age": 70, "sodium_min": None, "glucose_last": 120.5},
        [65, 138, 99],
    ]})
    assert out.shape == (2, 3)
    assert np.isnan(out[0, 1])
    assert out[1, 0] == pytest.approx(65.0)


def test_unknown_dict_key_is_rejected_not_silently_nan():
    p = _predictor()
    with pytest.raises(ValueError, match="unknown feature keys"):
        p.preprocess({"instances": [{"age": 70, "sodiun_min": 138}]})


def test_wrong_length_list_is_rejected():
    p = _predictor()
    with pytest.raises(ValueError, match="expected 3 values"):
        p.preprocess({"instances": [[70, 138]]})


def test_non_finite_and_non_numeric_values_are_rejected():
    p = _predictor()
    with pytest.raises(ValueError, match="non-finite"):
        p.preprocess({"instances": [[70, float("inf"), 99]]})
    with pytest.raises(ValueError, match="expected a number"):
        p.preprocess({"instances": [[70, "138", 99]]})
    with pytest.raises(ValueError, match="expected a number"):
        p.preprocess({"instances": [[70, True, 99]]})


def test_batch_size_is_capped():
    p = _predictor()
    batch = [[70, 138, 99]] * (cpr.MAX_BATCH + 1)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        p.preprocess({"instances": batch})


def test_load_refuses_a_bundle_without_threshold(tmp_path, monkeypatch):
    """ECC-68: no silent 0.5 fallback for a recall-weighted F2 threshold."""
    monkeypatch.setattr(
        cpr.prediction_utils, "download_model_artifacts", lambda uri: None
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"feature_order": _FEATURES, "groups": {}})
    )
    p = cpr.ReadmissionPredictor()
    with pytest.raises(RuntimeError, match="threshold.json missing"):
        p.load("gs://unused")


def test_response_declares_attribution_units():
    """ECC-73: TreeSHAP on binary:logistic is log-odds, and says so."""
    p = _predictor(threshold=0.2)
    probs = np.array([0.3])
    contribs = np.array([[0.1, -0.2, 0.05, -1.5]])  # 3 features + base value
    out = p.postprocess((probs, contribs))
    pred = out["predictions"][0]
    assert pred["attribution_units"] == "log_odds"
    assert pred["prediction"] == 1
    assert pred["base_value"] == pytest.approx(-1.5)
