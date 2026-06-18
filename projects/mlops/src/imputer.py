"""
Lightweight imputer that applies the missingness policy CSV.

Reads the policy file, fits on train only, and transforms any DataFrame.
Used by feature selection and training scripts — the same logic will be
containerized into the Vertex AI pipeline component later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MISSINGNESS_POLICY_CSV


class MissingnessImputer:
    """Fit on train, transform any split — no leakage.

    Policy rules (from missingness_policy.csv):
      - no_missing                        → pass through
      - impute_mode                       → fill with training-set mode
      - constant_unknown                  → fill with literal "Unknown"
      - missing_indicator+impute_median   → add binary _was_measured column
                                            + fill NULLs with training median
    """

    def __init__(self, policy_path: str | Path | None = None):
        self.policy_path = Path(policy_path or MISSINGNESS_POLICY_CSV)
        self.policy_: pd.DataFrame | None = None
        self.mode_values_: dict[str, object] = {}
        self.median_values_: dict[str, float] = {}
        self.indicator_cols_: list[str] = []
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None) -> "MissingnessImputer":
        """Learn imputation values from *train* data only.

        Accepts optional ``y`` for sklearn Pipeline compatibility (ignored).
        """
        policy = pd.read_csv(self.policy_path)
        self.policy_ = policy.set_index("column")

        for _, row in policy.iterrows():
            col = row["column"]
            pol = row["policy"]
            if col not in X.columns:
                continue

            if pol == "impute_mode":
                self.mode_values_[col] = X[col].mode().iloc[0]

            elif pol == "missing_indicator+impute_median":
                self.indicator_cols_.append(col)
                self.median_values_[col] = X[col].median()

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputations. Returns a new DataFrame."""
        if not self.fitted_:
            raise RuntimeError("Imputer must be fit before transform.")

        X = X.copy()

        for _, row in self.policy_.iterrows():
            col = row.name  # index is column name (set_index in fit)
            pol = row["policy"]
            if col not in X.columns:
                continue

            if pol == "no_missing":
                pass

            elif pol == "impute_mode":
                if col in self.mode_values_:
                    X[col] = X[col].fillna(self.mode_values_[col])

            elif pol == "constant_unknown":
                X[col] = X[col].fillna("Unknown")

            elif pol == "missing_indicator+impute_median":
                indicator_name = f"{col}_was_measured"
                X[indicator_name] = (~X[col].isna()).astype(np.int64)
                if col in self.median_values_:
                    X[col] = X[col].fillna(self.median_values_[col])

        return X

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit on X, then transform X.

        Accepts optional ``y`` and ``**fit_params`` for sklearn Pipeline
        compatibility (passed through to ``fit``, then ignored).
        """
        return self.fit(X, y=y).transform(X)

    @property
    def n_features_before(self) -> int:
        return len(self.policy_) if self.policy_ is not None else 0

    @property
    def n_features_after(self) -> int:
        return self.n_features_before + len(self.indicator_cols_)

    @property
    def indicator_columns(self) -> list[str]:
        return [f"{c}_was_measured" for c in self.indicator_cols_]
