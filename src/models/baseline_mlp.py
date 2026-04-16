from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


POSITION_COLS = ["damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


@dataclass(frozen=True)
class MlpBaselineConfig:
    random_state: int = 42

    # Shared architecture (you can split later if needed)
    hidden_sizes: tuple[int, ...] = (128, 64)

    # Classifier
    max_iter_clf: int = 800
    early_stopping_clf: bool = True
    n_iter_no_change_clf: int = 20
    alpha: float = 1e-4

    # Regressor
    max_iter_reg: int = 1200
    early_stopping_reg: bool = True
    n_iter_no_change_reg: int = 30


class MlpDamageBaseline:
    """
    Baseline approach:
    - MLPClassifier for num_damages (multiclass)
    - MLPRegressor (multioutput) for damage_pos_1..4 (masked in evaluation; model predicts all 4 slots)
    """

    def __init__(self, cfg: MlpBaselineConfig | None = None) -> None:
        self.cfg = cfg or MlpBaselineConfig()

        self.num_damages_clf = MLPClassifier(
            hidden_layer_sizes=self.cfg.hidden_sizes,
            activation="relu",
            solver="adam",
            alpha=self.cfg.alpha,
            random_state=self.cfg.random_state,
            max_iter=self.cfg.max_iter_clf,
            early_stopping=self.cfg.early_stopping_clf,
            n_iter_no_change=self.cfg.n_iter_no_change_clf,
        )

        base_reg = MLPRegressor(
            hidden_layer_sizes=self.cfg.hidden_sizes,
            activation="relu",
            solver="adam",
            alpha=self.cfg.alpha,
            random_state=self.cfg.random_state,
            max_iter=self.cfg.max_iter_reg,
            early_stopping=self.cfg.early_stopping_reg,
            n_iter_no_change=self.cfg.n_iter_no_change_reg,
        )
        self.pos_reg = MultiOutputRegressor(base_reg)

        self._is_fit = False
        # MLP nhạy với scale, nên chuẩn hóa X và y_pos để ổn định huấn luyện.
        self._x_scaler = StandardScaler()
        self._pos_scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        y_num = y_train["num_damages"].astype(int).to_numpy()
        y_pos = y_train[POSITION_COLS].astype(float).to_numpy()

        # fill missing slots with -1 for training stability
        y_pos_filled = np.where(np.isfinite(y_pos), y_pos, -1.0)

        X_np = X_train.to_numpy()
        X_scaled = self._x_scaler.fit_transform(X_np)

        # Scale target regression to help convergence.
        y_pos_scaled = self._pos_scaler.fit_transform(y_pos_filled)

        self.num_damages_clf.fit(X_scaled, y_num)
        self.pos_reg.fit(X_scaled, y_pos_scaled)

        self._is_fit = True

    def predict_num_damages(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Model is not fit yet.")
        X_scaled = self._x_scaler.transform(X.to_numpy())
        return self.num_damages_clf.predict(X_scaled).astype(int)

    def predict_positions(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Model is not fit yet.")
        X_scaled = self._x_scaler.transform(X.to_numpy())
        pred_scaled = self.pos_reg.predict(X_scaled)
        pred = self._pos_scaler.inverse_transform(pred_scaled)
        return pd.DataFrame(pred, columns=POSITION_COLS)

