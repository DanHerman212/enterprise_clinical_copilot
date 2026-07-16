"""Tests for train_final (pipeline module 4).

Pins the corrected contract:
  * the final model is fit on the COMBINED train+val set with the HPO params;
  * the returned metric is the combined-set fit metric (for logging/overfit
    sanity), NOT a val AUCPR computed on data the model just trained on;
  * the saved model is loadable and honours the supplied hyperparameters.

The unbiased estimate is produced later on the hold-out test set by
evaluate_test; that is intentionally out of scope here.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

from pipelines.components.train_final import run_train_final

CAT = ["gender"]


def _frame(rng, n):
    gender = pd.Categorical(rng.choice(["M", "F"], n), categories=["M", "F"])
    return pd.DataFrame(
        {
            "age": rng.normal(60, 10, n),
            "glucose_last": rng.normal(100, 15, n),
            "gender": gender,
        }
    )


def _setup(tmp_path, n_train=120, n_val=40, n_estimators=8):
    rng = np.random.RandomState(0)
    X_train, X_val = _frame(rng, n_train), _frame(rng, n_val)
    y_train = pd.Series(rng.randint(0, 2, n_train), name="readmission_30d")
    y_val = pd.Series(rng.randint(0, 2, n_val), name="readmission_30d")
    # Patient-grouped val: assign each row to one of ~10 patients.
    g_val = pd.Series(np.repeat(np.arange(10), n_val // 10)[:n_val], name="subject_id")

    paths = {}
    for key, obj in [
        ("x_train", X_train), ("x_val", X_val),
        ("y_train", pd.DataFrame(y_train)), ("y_val", pd.DataFrame(y_val)),
        ("groups_val", pd.DataFrame(g_val)),
    ]:
        p = tmp_path / f"{key}.parquet"
        obj.to_parquet(p, index=False)
        paths[key] = str(p)

    best_params = {
        "n_estimators": n_estimators, "max_depth": 3, "random_state": 42,
        "enable_categorical": True, "n_jobs": 1,
    }
    bp_path = tmp_path / "best_params.json"
    bp_path.write_text(json.dumps(best_params))
    paths["best_params"] = str(bp_path)
    paths["model"] = str(tmp_path / "final_model.joblib")
    paths["curve"] = str(tmp_path / "training_curve.json")
    return paths, X_train, X_val, y_train, y_val, best_params


def test_returned_metric_is_combined_fit_not_val(tmp_path):
    paths, X_train, X_val, y_train, y_val, _ = _setup(tmp_path)
    returned = run_train_final(
        x_train_path=paths["x_train"], y_train_path=paths["y_train"],
        x_val_path=paths["x_val"], y_val_path=paths["y_val"],
        best_params_path=paths["best_params"], cat_features=CAT,
        model_artifact_path=paths["model"],
        training_curve_path=paths["curve"],
        groups_val_path=paths["groups_val"],
    )
    # Rebuild the combined DESIGN matrix the model was actually trained on
    # (train + 80% val, matching the patient-grouped monitor split logic).
    model = joblib.load(paths["model"])
    from sklearn.model_selection import GroupShuffleSplit
    g_val = pd.read_parquet(paths["groups_val"]).iloc[:, 0].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_pos, _ = next(gss.split(np.zeros(len(g_val)), groups=g_val))
    mask = np.zeros(len(g_val), dtype=bool); mask[train_pos] = True
    X_val_train = X_val.iloc[mask]; y_val_train = y_val.iloc[mask]
    X_all = pd.concat([X_train, X_val_train], ignore_index=True)
    y_all = pd.concat([y_train, y_val_train], ignore_index=True)
    expected = float(average_precision_score(y_all, model.predict_proba(X_all)[:, 1]))
    assert returned == expected


def test_model_trained_on_all_rows(tmp_path):
    paths, X_train, X_val, y_train, y_val, best_params = _setup(tmp_path)
    run_train_final(
        x_train_path=paths["x_train"], y_train_path=paths["y_train"],
        x_val_path=paths["x_val"], y_val_path=paths["y_val"],
        best_params_path=paths["best_params"], cat_features=CAT,
        model_artifact_path=paths["model"],
        training_curve_path=paths["curve"],
        groups_val_path=paths["groups_val"],
    )
    combined = joblib.load(paths["model"])

    # Train an otherwise-identical model on train ONLY. If the pipeline model
    # had ignored the val rows, the two would predict identically; they must
    # differ, proving the val split participated in the final fit.
    from xgboost import XGBClassifier

    train_only = XGBClassifier(**best_params).fit(X_train, y_train)

    p_combined = combined.predict_proba(X_train)[:, 1]
    p_train_only = train_only.predict_proba(X_train)[:, 1]
    assert not np.allclose(p_combined, p_train_only)



def test_uses_supplied_hyperparameters(tmp_path):
    paths, *_ = _setup(tmp_path, n_estimators=5)
    run_train_final(
        x_train_path=paths["x_train"], y_train_path=paths["y_train"],
        x_val_path=paths["x_val"], y_val_path=paths["y_val"],
        best_params_path=paths["best_params"], cat_features=CAT,
        model_artifact_path=paths["model"],
        training_curve_path=paths["curve"],
        groups_val_path=paths["groups_val"],
    )
    model = joblib.load(paths["model"])
    assert model.n_estimators == 5


def test_model_artifact_saved(tmp_path):
    paths, *_ = _setup(tmp_path)
    run_train_final(
        x_train_path=paths["x_train"], y_train_path=paths["y_train"],
        x_val_path=paths["x_val"], y_val_path=paths["y_val"],
        best_params_path=paths["best_params"], cat_features=CAT,
        model_artifact_path=paths["model"],
        training_curve_path=paths["curve"],
        groups_val_path=paths["groups_val"],
    )
    from pathlib import Path
    assert Path(paths["model"]).exists()


def test_training_curve_artifact_saved_with_per_round_scores(tmp_path):
    # Larger dataset so the monitor split gets both classes (AUCPR needs ≥1 pos).
    paths, *_ = _setup(tmp_path, n_train=200, n_val=100, n_estimators=30)
    run_train_final(
        x_train_path=paths["x_train"], y_train_path=paths["y_train"],
        x_val_path=paths["x_val"], y_val_path=paths["y_val"],
        best_params_path=paths["best_params"], cat_features=CAT,
        model_artifact_path=paths["model"],
        training_curve_path=paths["curve"],
        groups_val_path=paths["groups_val"],
    )
    curve = json.loads(open(paths["curve"]).read())
    train_keys = [k for k in curve if k.startswith("train_")]
    monitor_keys = [k for k in curve if k.startswith("monitor_")]
    assert len(train_keys) >= 1, f"no train_* curve keys: {list(curve.keys())}"
    assert len(monitor_keys) >= 1
    assert len(curve[train_keys[0]]) >= 1
    assert all(isinstance(v, float) for v in curve[train_keys[0]])
