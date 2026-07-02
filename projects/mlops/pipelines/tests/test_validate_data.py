"""Tests for the data-validation gate (pipeline module 3).

Pins the real Evidently drift gate behavior:
  * no drift -> passes (returns True), no exception;
  * drift beyond ``max_drifted_share`` -> HARD-FAIL (raises), by default;
  * ``fail_on_drift=False`` -> returns False instead of raising;
  * real Evidently HTML reports are written (not the old placeholder stub).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from pipelines.components.validate_data import run_validate_data


def _write(df: pd.DataFrame, path) -> str:
    df.to_parquet(path, index=False)
    return str(path)


def _make_frame(rng, *, shift: float = 0.0, n: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": rng.normal(60 + shift, 5, n),
            "glucose_last": rng.normal(100 + shift * 4, 10, n),
            "prior_admission_count": rng.poisson(2, n).astype(float),
        }
    )


def test_no_drift_passes(tmp_path):
    rng = np.random.RandomState(0)
    ref = _make_frame(rng)
    cur = ref.copy()  # identical -> zero drift
    result = run_validate_data(
        x_train_path=_write(ref, tmp_path / "train.parquet"),
        x_val_path=_write(cur, tmp_path / "val.parquet"),
        drift_report_html=str(tmp_path / "drift.html"),
        quality_report_html=str(tmp_path / "quality.html"),
        max_drifted_share=0.2,
    )
    assert result is True


def test_drift_hard_fails(tmp_path):
    rng = np.random.RandomState(1)
    ref = _make_frame(rng)
    cur = _make_frame(rng, shift=10.0)  # strong distribution shift on all cols
    with pytest.raises(ValueError):
        run_validate_data(
            x_train_path=_write(ref, tmp_path / "train.parquet"),
            x_val_path=_write(cur, tmp_path / "val.parquet"),
            drift_report_html=str(tmp_path / "drift.html"),
            quality_report_html=str(tmp_path / "quality.html"),
            max_drifted_share=0.2,
        )


def test_drift_does_not_raise_when_override(tmp_path):
    rng = np.random.RandomState(2)
    ref = _make_frame(rng)
    cur = _make_frame(rng, shift=10.0)
    result = run_validate_data(
        x_train_path=_write(ref, tmp_path / "train.parquet"),
        x_val_path=_write(cur, tmp_path / "val.parquet"),
        drift_report_html=str(tmp_path / "drift.html"),
        quality_report_html=str(tmp_path / "quality.html"),
        max_drifted_share=0.2,
        fail_on_drift=False,
    )
    assert result is False


def test_real_html_reports_written(tmp_path):
    rng = np.random.RandomState(3)
    ref = _make_frame(rng)
    cur = ref.copy()
    drift_html = tmp_path / "drift.html"
    quality_html = tmp_path / "quality.html"
    run_validate_data(
        x_train_path=_write(ref, tmp_path / "train.parquet"),
        x_val_path=_write(cur, tmp_path / "val.parquet"),
        drift_report_html=str(drift_html),
        quality_report_html=str(quality_html),
        max_drifted_share=0.2,
    )
    for path in (drift_html, quality_html):
        assert path.exists() and path.stat().st_size > 0
        text = path.read_text()
        # Real Evidently output, not the old "Skipped in test" placeholder.
        assert "Skipped" not in text
        assert "html" in text.lower()
