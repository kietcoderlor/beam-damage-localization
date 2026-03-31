from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


POSITION_COLS = ["damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


@dataclass(frozen=True)
class DamageMetrics:
    num_damages_accuracy: float
    num_damages_f1_macro: float
    pos_mae_overall: float
    pos_rmse_overall: float
    pos_mae_per_slot: dict[str, float]
    pos_rmse_per_slot: dict[str, float]


def _masked_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Compute MAE/RMSE over entries where y_true is finite.
    """
    mask = np.isfinite(y_true)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return float(mae), float(rmse)


def compute_damage_metrics(
    y_num_true: np.ndarray,
    y_num_pred: np.ndarray,
    y_pos_true: pd.DataFrame,
    y_pos_pred: pd.DataFrame,
) -> DamageMetrics:
    """
    Metrics for:
    - num_damages (multiclass classification)
    - damage positions (4-slot regression, with NaNs masked out)
    """
    num_acc = float(accuracy_score(y_num_true, y_num_pred))
    num_f1_macro = float(f1_score(y_num_true, y_num_pred, average="macro"))

    pos_true = y_pos_true[POSITION_COLS].astype(float).to_numpy()
    pos_pred = y_pos_pred[POSITION_COLS].astype(float).to_numpy()

    overall_mae, overall_rmse = _masked_regression_metrics(pos_true, pos_pred)

    mae_per: dict[str, float] = {}
    rmse_per: dict[str, float] = {}
    for i, col in enumerate(POSITION_COLS):
        mae_i, rmse_i = _masked_regression_metrics(pos_true[:, i], pos_pred[:, i])
        mae_per[col] = mae_i
        rmse_per[col] = rmse_i

    return DamageMetrics(
        num_damages_accuracy=num_acc,
        num_damages_f1_macro=num_f1_macro,
        pos_mae_overall=overall_mae,
        pos_rmse_overall=overall_rmse,
        pos_mae_per_slot=mae_per,
        pos_rmse_per_slot=rmse_per,
    )

