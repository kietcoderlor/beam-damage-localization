from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBClassifier, XGBRegressor


POSITION_COLS = ["damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


@dataclass(frozen=True)
class XgbBaselineConfig:
    random_state: int = 42
    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    tree_method: str = "hist"


class XgbDamageBaseline:
    """
    Baseline approach:
    - classify num_damages (0/1/2/4)
    - regress damage_pos_1..4 (masked in evaluation; model predicts all 4 slots)
    """

    def __init__(self, cfg: XgbBaselineConfig | None = None) -> None:
        self.cfg = cfg or XgbBaselineConfig()

        self.num_damages_clf = XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_lambda=self.cfg.reg_lambda,
            random_state=self.cfg.random_state,
            tree_method=self.cfg.tree_method,
            n_jobs=0,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )

        base_reg = XGBRegressor(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_lambda=self.cfg.reg_lambda,
            random_state=self.cfg.random_state,
            tree_method=self.cfg.tree_method,
            n_jobs=0,
            objective="reg:squarederror",
        )
        self.pos_reg = MultiOutputRegressor(base_reg)

        self._is_fit = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> None:
        y_num = y_train["num_damages"].astype(int).to_numpy()
        y_pos = y_train[POSITION_COLS].astype(float).to_numpy()

        # Fill missing position slots with -1 for training stability.
        # NOTE: evaluation still masks by true NaNs; this is just a training convenience.
        y_pos_filled = np.where(np.isfinite(y_pos), y_pos, -1.0)

        if X_val is not None and y_val is not None:
            y_num_val = y_val["num_damages"].astype(int).to_numpy()
            self.num_damages_clf.fit(
                X_train.to_numpy(),
                y_num,
                eval_set=[(X_val.to_numpy(), y_num_val)],
                verbose=False,
            )
        else:
            self.num_damages_clf.fit(X_train.to_numpy(), y_num, verbose=False)

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
        pred_df = pd.DataFrame(pred, columns=POSITION_COLS)
        return pred_df

