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
        self._class_values: np.ndarray | None = None  # original labels (e.g., [0,1,2,4])
        self._class_pos_reg: dict[int, MultiOutputRegressor] = {}
        self._valid_position_values: np.ndarray | None = None

    def _make_pos_regressor(self) -> MultiOutputRegressor:
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
        return MultiOutputRegressor(base_reg)

    @staticmethod
    def _snap_to_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        if grid.size == 0:
            return values
        snapped = values.copy()
        for i, v in enumerate(values):
            snapped[i] = float(grid[np.argmin(np.abs(grid - v))])
        return snapped

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> None:
        y_num_raw_train = y_train["num_damages"].astype(int).to_numpy()
        y_num_raw_val = None
        if X_val is not None and y_val is not None:
            y_num_raw_val = y_val["num_damages"].astype(int).to_numpy()

        # XGBoost classifier (multi) expects consistent class set across train/val.
        # Our labels are not guaranteed to be continuous (may be [0,1,2,4] etc),
        # so we remap them to contiguous indices for training.
        if y_num_raw_val is None:
            class_values = np.unique(y_num_raw_train)
        else:
            class_values = np.unique(np.concatenate([y_num_raw_train, y_num_raw_val]))
        self._class_values = class_values

        # Since class_values is sorted, we can use searchsorted for mapping.
        y_num_train_enc = np.searchsorted(class_values, y_num_raw_train).astype(int)
        y_num_val_enc = None
        if y_num_raw_val is not None:
            y_num_val_enc = np.searchsorted(class_values, y_num_raw_val).astype(int)

        y_pos = y_train[POSITION_COLS].astype(float).to_numpy()
        finite_positions = y_pos[np.isfinite(y_pos)]
        if finite_positions.size > 0:
            self._valid_position_values = np.unique(np.asarray(finite_positions, dtype=float))
        else:
            self._valid_position_values = None

        # Fill missing position slots with -1 for training stability.
        # NOTE: evaluation still masks by true NaNs; this is just a training convenience.
        y_pos_filled = np.where(np.isfinite(y_pos), y_pos, -1.0)

        if X_val is not None and y_val is not None:
            self.num_damages_clf.fit(
                X_train.to_numpy(),
                y_num_train_enc,
                eval_set=[(X_val.to_numpy(), y_num_val_enc)],
                verbose=False,
            )
        else:
            self.num_damages_clf.fit(X_train.to_numpy(), y_num_train_enc, verbose=False)

        self.pos_reg.fit(X_train.to_numpy(), y_pos_filled)

        # Mixture-of-experts: class-conditional position regressors.
        # This helps each regressor focus on topology of a specific damage-count class.
        self._class_pos_reg = {}
        for cls in np.unique(y_num_raw_train):
            cls_mask = y_num_raw_train == cls
            # Keep a minimal support threshold to avoid unstable overfitting on tiny classes.
            if int(np.sum(cls_mask)) < 3:
                continue
            reg = self._make_pos_regressor()
            reg.fit(X_train.to_numpy()[cls_mask], y_pos_filled[cls_mask])
            self._class_pos_reg[int(cls)] = reg

        self._is_fit = True

    def predict_num_damages(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Model is not fit yet.")
        if self._class_values is None:
            raise RuntimeError("Internal class mapping is missing.")
        pred_enc = self.num_damages_clf.predict(X.to_numpy()).astype(int)
        # Map back to original labels.
        return self._class_values[pred_enc]

    def predict_positions(
        self,
        X: pd.DataFrame,
        y_num_pred: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("Model is not fit yet.")
        x_np = X.to_numpy()
        if y_num_pred is None:
            y_num_pred = self.predict_num_damages(X)
        y_num_pred = np.asarray(y_num_pred, dtype=int)

        # Default to global regressor and selectively replace with class-specific experts.
        pred = self.pos_reg.predict(x_np)
        for cls, reg in self._class_pos_reg.items():
            cls_mask = y_num_pred == cls
            if not np.any(cls_mask):
                continue
            pred[cls_mask] = reg.predict(x_np[cls_mask])

        # Physics-aware post-processing:
        # - clip invalid negatives (except sentinel -1)
        # - enforce sorted positions
        # - optional snap to known valid position grid
        processed = np.asarray(pred, dtype=float)
        for i in range(processed.shape[0]):
            row = processed[i]
            row = np.where(row < 0.0, -1.0, row)
            valid_mask = row >= 0.0
            if np.any(valid_mask):
                valid_vals = np.sort(row[valid_mask])
                if self._valid_position_values is not None:
                    valid_vals = self._snap_to_grid(valid_vals, self._valid_position_values)
                k = int(min(max(y_num_pred[i], 0), len(valid_vals)))
                row_out = np.full_like(row, np.nan, dtype=float)
                if k > 0:
                    row_out[:k] = valid_vals[:k]
                processed[i] = row_out
            else:
                processed[i] = np.full_like(row, np.nan, dtype=float)

        pred_df = pd.DataFrame(processed, columns=POSITION_COLS)
        return pred_df

