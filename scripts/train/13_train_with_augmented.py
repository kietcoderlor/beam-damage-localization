"""
Retrain XGBoost advanced model on augmented dataset (class-0 augmented).

Loads train_augmented.csv + val_augmented.csv, evaluates on original
val.csv and test.csv so results are comparable to baseline runs.

Usage:
    python scripts/train/13_train_with_augmented.py
    python scripts/train/13_train_with_augmented.py --output-name xgb_aug_noise_v2
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path


def _find_project_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "src").exists():
            return candidate
    raise RuntimeError("Cannot find project root.")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.config.settings import OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED  # noqa: E402
from src.eval.evaluate import print_damage_metrics  # noqa: E402
from src.eval.metrics import compute_damage_metrics  # noqa: E402
from src.features.baseline_features import BaselineFeatureConfig, build_baseline_feature_matrix  # noqa: E402
from src.features.physics_features import PhysicsFeatureConfig, build_physics_feature_matrix  # noqa: E402
from src.features.wavelet_features import WaveletFeatureConfig, build_wavelet_feature_matrix  # noqa: E402
from src.models.baseline_xgb import XgbBaselineConfig, XgbDamageBaseline  # noqa: E402


TARGET_COLS = ["num_damages", "damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


def _load_split(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/data/03_augment_class0.py first."
        )
    df = pd.read_csv(path)
    return df[df["num_modes_found"] == 4].copy().reset_index(drop=True)


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    baseline_cfg = BaselineFeatureConfig(
        resample_len=32,
        include_freq=True,
        include_mode_stats=True,
        include_resampled_vectors=True,
    )
    wavelet_cfg = WaveletFeatureConfig(
        wavelet="db4",
        level=3,
        include_freq=False,
        include_raw_energy_ratios=True,
        include_wavelet_stats=True,
    )
    physics_cfg = PhysicsFeatureConfig(
        include_freq=False,
        include_gradient_stats=True,
        include_curvature_stats=True,
        include_shape_descriptors=True,
    )

    X_b = build_baseline_feature_matrix(df, baseline_cfg)
    X_w = build_wavelet_feature_matrix(df, wavelet_cfg)
    X_p = build_physics_feature_matrix(df, physics_cfg)

    freq_cols = [c for c in X_b.columns if c.startswith("freq_mode_")]
    X_w = X_w.drop(columns=[c for c in freq_cols if c in X_w.columns], errors="ignore")
    X_p = X_p.drop(columns=[c for c in freq_cols if c in X_p.columns], errors="ignore")

    X = pd.concat([X_b, X_w, X_p], axis=1)
    cfg_dict = {
        "baseline_config": asdict(baseline_cfg),
        "wavelet_config": asdict(wavelet_cfg),
        "physics_config": asdict(physics_cfg),
    }
    return X, cfg_dict


def _eval_split(name: str, model: XgbDamageBaseline, X: pd.DataFrame, y: pd.DataFrame) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", type=str, default="xgb_aug_noise")
    parser.add_argument("--train-csv", type=str, default="train_augmented.csv",
                        help="Augmented train split filename in data/processed/ (default: train_augmented.csv)")
    parser.add_argument("--val-csv", type=str, default="val_augmented.csv",
                        help="Augmented val split filename in data/processed/ (default: val_augmented.csv)")
    parser.add_argument("--test-csv", type=str, default="test.csv",
                        help="Test split filename in data/processed/ (default: test.csv)")
    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    args = parser.parse_args()

    # Train and validate on augmented splits (class 0 augmented)
    train_df = _load_split(args.train_csv)
    val_aug_df = _load_split(args.val_csv)

    # Evaluate on the original val/test (no synthetic data) for fair comparison
    val_orig_df = _load_split("val.csv")
    test_df = _load_split(args.test_csv)

    print("Loaded splits:")
    print(f"  train_augmented:  {train_df.shape}  | class-0: {(train_df['num_damages']==0).sum()}")
    print(f"  val_augmented:    {val_aug_df.shape}  | class-0: {(val_aug_df['num_damages']==0).sum()}")
    print(f"  val_original:     {val_orig_df.shape}")
    print(f"  test:             {test_df.shape}")

    X_train, feature_cfg = _build_features(train_df)
    X_val_aug, _ = _build_features(val_aug_df)
    X_val_orig, _ = _build_features(val_orig_df)
    X_test, _ = _build_features(test_df)

    y_train = train_df[TARGET_COLS].copy()
    y_val_aug = val_aug_df[TARGET_COLS].copy()
    y_val_orig = val_orig_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    cfg = XgbBaselineConfig(
        random_state=RANDOM_SEED,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
    )
    model = XgbDamageBaseline(cfg)

    # Fit using augmented val for early stopping signal
    model.fit(X_train, y_train, X_val=X_val_aug, y_val=y_val_aug)

    print("\n--- Evaluation ---")
    _eval_split("TRAIN (augmented)", model, X_train, y_train)
    _eval_split("VAL   (augmented)", model, X_val_aug, y_val_aug)
    _eval_split("VAL   (original) ", model, X_val_orig, y_val_orig)
    _eval_split(f"TEST ({args.test_csv})", model, X_test, y_test)

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
            "augmentation": "gaussian_noise_class0",
        },
        artifact_path,
    )
    print(f"\nSaved artifact -> {artifact_path}")


if __name__ == "__main__":
    main()
