from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from src.features.baseline_features import build_baseline_feature_matrix  # noqa: E402
from src.models.baseline_sklearn import POSITION_COLS  # noqa: E402


def _masked_row_pos_mae(y_true_row: np.ndarray, y_pred_row: np.ndarray) -> float:
    mask = np.isfinite(y_true_row)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true_row[mask] - y_pred_row[mask])))


def _print_ascii_safe(text: str) -> None:
    # Avoid cp1252 console crashes on Vietnamese text in scenario names.
    print(text.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    split_path = PROCESSED_DIR / "test.csv"
    artifact_path = OUTPUT_DIR / "baseline_rf" / "artifact.joblib"
    out_dir = OUTPUT_DIR / "baseline_rf"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(split_path)
    test_df = test_df[test_df["num_modes_found"] == 4].copy().reset_index(drop=True)

    artifact = joblib.load(artifact_path)
    feat_cfg = artifact["feature_config"]
    model = artifact["model"]

    X_test = build_baseline_feature_matrix(test_df, feat_cfg)
    y_num_true = test_df["num_damages"].astype(int).to_numpy()
    y_pos_true = test_df[POSITION_COLS].astype(float)

    y_num_pred = model.predict_num_damages(X_test)
    y_pos_pred = model.predict_positions(X_test)

    labels = sorted(np.unique(np.concatenate([y_num_true, y_num_pred])))
    cm = confusion_matrix(y_num_true, y_num_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    print("\n=== TEST confusion matrix (num_damages) ===")
    print(cm_df.to_string())
    cm_df.to_csv(out_dir / "test_confusion_matrix.csv", index=True, encoding="utf-8-sig")

    report = classification_report(y_num_true, y_num_pred, digits=4, zero_division=0)
    print("\n=== TEST classification report (num_damages) ===")
    print(report)
    (out_dir / "test_classification_report.txt").write_text(report, encoding="utf-8")

    pos_true_np = y_pos_true.to_numpy()
    pos_pred_np = y_pos_pred[POSITION_COLS].to_numpy()

    rows = []
    for idx in range(len(test_df)):
        mae_row = _masked_row_pos_mae(pos_true_np[idx], pos_pred_np[idx])
        rows.append(
            {
                "config_id": test_df.loc[idx, "config_id"],
                "scenario_name": test_df.loc[idx, "scenario_name"],
                "num_damages_true": int(y_num_true[idx]),
                "num_damages_pred": int(y_num_pred[idx]),
                "num_damages_correct": int(y_num_true[idx] == y_num_pred[idx]),
                "position_mae_masked": mae_row,
            }
        )
    err_df = pd.DataFrame(rows)

    print("\n=== Position MAE by true num_damages (TEST) ===")
    by_cls = (
        err_df.groupby("num_damages_true", dropna=False)["position_mae_masked"]
        .agg(["count", "mean", "std", "max"])
        .sort_index()
    )
    print(by_cls.to_string(float_format=lambda x: f"{x:.4f}"))
    by_cls.to_csv(out_dir / "test_position_error_by_class.csv", encoding="utf-8-sig")

    top_err = err_df.sort_values("position_mae_masked", ascending=False).head(15)
    print("\n=== Top 15 worst position errors (TEST) ===")
    _print_ascii_safe(top_err.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    top_err.to_csv(out_dir / "test_top15_position_errors.csv", index=False, encoding="utf-8-sig")

    print("\nSaved analysis files:")
    print(f" - {out_dir / 'test_confusion_matrix.csv'}")
    print(f" - {out_dir / 'test_classification_report.txt'}")
    print(f" - {out_dir / 'test_position_error_by_class.csv'}")
    print(f" - {out_dir / 'test_top15_position_errors.csv'}")


if __name__ == "__main__":
    main()

