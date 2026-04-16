from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
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

from src.config.settings import OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED  # noqa: E402
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
from src.models.baseline_xgb import (  # noqa: E402
    XgbBaselineConfig,
    XgbDamageBaseline,
)


TARGET_COLS = ["num_damages", "damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    return df[df["num_modes_found"] == 4].copy().reset_index(drop=True)


def _build_advanced_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    baseline_cfg = BaselineFeatureConfig(
        resample_len=32,
        include_freq=True,
        include_mode_stats=True,
        include_resampled_vectors=True,
    )
    wavelet_cfg = WaveletFeatureConfig(
        wavelet="db4",
        level=3,
        include_freq=False,  # avoid duplicate freq columns
        include_raw_energy_ratios=True,
        include_wavelet_stats=True,
    )
    physics_cfg = PhysicsFeatureConfig(
        include_freq=False,  # avoid duplicate freq columns
        include_gradient_stats=True,
        include_curvature_stats=True,
        include_shape_descriptors=True,
    )

    X_baseline = build_baseline_feature_matrix(df, baseline_cfg)
    X_wavelet = build_wavelet_feature_matrix(df, wavelet_cfg)
    X_physics = build_physics_feature_matrix(df, physics_cfg)

    # Remove duplicated frequency columns from advanced blocks if present.
    freq_cols = [c for c in X_baseline.columns if c.startswith("freq_mode_")]
    X_wavelet = X_wavelet.drop(columns=[c for c in freq_cols if c in X_wavelet.columns], errors="ignore")
    X_physics = X_physics.drop(columns=[c for c in freq_cols if c in X_physics.columns], errors="ignore")

    X = pd.concat([X_baseline, X_wavelet, X_physics], axis=1)
    cfg_dict = {
        "baseline_config": asdict(baseline_cfg),
        "wavelet_config": asdict(wavelet_cfg),
        "physics_config": asdict(physics_cfg),
    }
    return X, cfg_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", type=str, default="baseline_xgb_advanced")
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--tree-method", type=str, default="hist")
    args = parser.parse_args()

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    print("Loaded splits:")
    print(" - train:", train_df.shape)
    print(" - val:  ", val_df.shape)
    print(" - test: ", test_df.shape)

    X_train, feature_cfg = _build_advanced_feature_matrix(train_df)
    X_val, _ = _build_advanced_feature_matrix(val_df)
    X_test, _ = _build_advanced_feature_matrix(test_df)

    y_train = train_df[TARGET_COLS].copy()
    y_val = val_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    cfg = XgbBaselineConfig(
        random_state=RANDOM_SEED,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        tree_method=args.tree_method,
    )
    model = XgbDamageBaseline(cfg)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def eval_split(name: str, X: pd.DataFrame, y: pd.DataFrame) -> None:
        y_num_true = y["num_damages"].astype(int).to_numpy()
        y_num_pred = model.predict_num_damages(X)
        y_pos_pred = model.predict_positions(X)
        m = compute_damage_metrics(
            y_num_true=y_num_true,
            y_num_pred=y_num_pred,
            y_pos_true=y[TARGET_COLS[1:]],
            y_pos_pred=y_pos_pred,
        )
        print_damage_metrics(name, m)

    eval_split("TRAIN", X_train, y_train)
    eval_split("VAL", X_val, y_val)
    eval_split("TEST", X_test, y_test)

    out_dir = OUTPUT_DIR / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "artifact.joblib"
    joblib.dump(
        {
            "feature_config": feature_cfg,
            "model_config": asdict(cfg),
            "model": model,
            "feature_columns": X_train.columns.tolist(),
            "feature_builder": "baseline+wavelet+physics",
        },
        artifact_path,
    )
    print(f"\nSaved artifact to: {artifact_path}")


if __name__ == "__main__":
    main()

