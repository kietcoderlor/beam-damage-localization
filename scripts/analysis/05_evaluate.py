from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate project root.")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from src.eval.evaluate import print_damage_metrics  # noqa: E402
from src.eval.metrics import compute_damage_metrics  # noqa: E402
from src.features.baseline_features import (  # noqa: E402
    BaselineFeatureConfig,
    build_baseline_feature_matrix,
)
from src.features.physics_features import (  # noqa: E402
    PhysicsFeatureConfig,
    build_physics_feature_matrix,
)
from src.features.wavelet_features import (  # noqa: E402
    WaveletFeatureConfig,
    build_wavelet_feature_matrix,
)


TARGET_COLS = ["num_damages", "damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    return df[df["num_modes_found"] == 4].copy().reset_index(drop=True)


def _as_metrics_dict(m) -> dict:
    return {
        "num_damages_accuracy": m.num_damages_accuracy,
        "num_damages_f1_macro": m.num_damages_f1_macro,
        "pos_mae_overall": m.pos_mae_overall,
        "pos_rmse_overall": m.pos_rmse_overall,
        "pos_mae_per_slot": m.pos_mae_per_slot,
        "pos_rmse_per_slot": m.pos_rmse_per_slot,
    }


def _build_features(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    feature_builder = artifact.get("feature_builder", "baseline")
    feature_columns = artifact.get("feature_columns")

    if feature_builder == "baseline":
        feat_cfg = artifact["feature_config"]
        X = build_baseline_feature_matrix(df, feat_cfg)
    elif feature_builder == "cnn_raw_modal":
        X = df.copy()
    elif feature_builder == "baseline+wavelet+physics":
        feat_cfg = artifact["feature_config"]
        baseline_cfg = feat_cfg["baseline_config"]
        wavelet_cfg = feat_cfg["wavelet_config"]
        physics_cfg = feat_cfg["physics_config"]

        X_baseline = build_baseline_feature_matrix(df, BaselineFeatureConfig(**baseline_cfg))
        X_wavelet = build_wavelet_feature_matrix(df, WaveletFeatureConfig(**wavelet_cfg))
        X_physics = build_physics_feature_matrix(df, PhysicsFeatureConfig(**physics_cfg))

        freq_cols = [c for c in X_baseline.columns if c.startswith("freq_mode_")]
        X_wavelet = X_wavelet.drop(columns=[c for c in freq_cols if c in X_wavelet.columns], errors="ignore")
        X_physics = X_physics.drop(columns=[c for c in freq_cols if c in X_physics.columns], errors="ignore")
        X = pd.concat([X_baseline, X_wavelet, X_physics], axis=1)
    else:
        raise ValueError(f"Unsupported feature_builder: {feature_builder}")

    if feature_columns is not None:
        X = X[feature_columns]
    return X


def evaluate_split(title: str, model, X: pd.DataFrame, y: pd.DataFrame) -> dict:
    y_num_true = y["num_damages"].astype(int).to_numpy()
    y_num_pred = model.predict_num_damages(X)
    y_pos_pred = model.predict_positions(X)
    m = compute_damage_metrics(
        y_num_true=y_num_true,
        y_num_pred=y_num_pred,
        y_pos_true=y[TARGET_COLS[1:]],
        y_pos_pred=y_pos_pred,
    )
    print_damage_metrics(title, m)
    return _as_metrics_dict(m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="baseline_rf")
    args = parser.parse_args()

    run_dir = OUTPUT_DIR / args.run_name
    artifact_path = run_dir / "artifact.joblib"
    artifact = joblib.load(artifact_path)

    model = artifact["model"]

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    X_train = _build_features(train_df, artifact)
    X_val = _build_features(val_df, artifact)
    X_test = _build_features(test_df, artifact)

    y_train = train_df[TARGET_COLS].copy()
    y_val = val_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    metrics = {
        "TRAIN": evaluate_split("TRAIN", model, X_train, y_train),
        "VAL": evaluate_split("VAL", model, X_val, y_val),
        "TEST": evaluate_split("TEST", model, X_test, y_test),
    }

    out_path = run_dir / "metrics_summary.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

