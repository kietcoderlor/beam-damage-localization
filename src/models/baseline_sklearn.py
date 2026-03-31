from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


POSITION_COLS = ["damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


@dataclass(frozen=True)
class RfBaselineConfig:
    random_state: int = 42
    n_estimators: int = 600
    max_depth: int | None = None
    min_samples_leaf: int = 1
    n_jobs: int = -1


class RfDamageBaseline:
    """
    Simple, dependency-light baseline:
    - RandomForestClassifier for num_damages
    - RandomForestRegressor (multi-output) for damage positions
    """

    def __init__(self, cfg: RfBaselineConfig | None = None) -> None:
        self.cfg = cfg or RfBaselineConfig()

        self.num_damages_clf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            class_weight="balanced_subsample",
        )

        base_reg = RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        self.pos_reg = MultiOutputRegressor(base_reg)

        self._is_fit = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        y_num = y_train["num_damages"].astype(int).to_numpy()
        y_pos = y_train[POSITION_COLS].astype(float).to_numpy()

        # fill missing slots with -1 for training convenience
        y_pos_filled = np.where(np.isfinite(y_pos), y_pos, -1.0)

        self.num_damages_clf.fit(X_train.to_numpy(), y_num)
        self.pos_reg.fit(X_train.to_numpy(), y_pos_filled)
        self._is_fit = True

    def predict_num_damages(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Model is not fit yet.")
        return self.num_damages_clf.predict(X.to_numpy()).astype(int)

    def predict_positions(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Model is not fit yet.")
        pred = self.pos_reg.predict(X.to_numpy())
        return pd.DataFrame(pred, columns=POSITION_COLS)

