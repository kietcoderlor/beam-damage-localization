from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import sys

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
from src.models.baseline_xgb import XgbBaselineConfig, XgbDamageBaseline  # noqa: E402


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

    X_baseline = build_baseline_feature_matrix(df, baseline_cfg)
    X_wavelet = build_wavelet_feature_matrix(df, wavelet_cfg)
    X_physics = build_physics_feature_matrix(df, physics_cfg)

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


def _choice_maybe_scalar(rng: np.random.Generator, seq: list) -> object:
    v = rng.choice(seq)
    return v.item() if isinstance(v, np.generic) else v


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": int(_choice_maybe_scalar(rng, [200, 300, 450, 600, 800])),
        "learning_rate": float(_choice_maybe_scalar(rng, [0.03, 0.05, 0.08, 0.1])),
        "max_depth": int(_choice_maybe_scalar(rng, [3, 4, 5, 6])),
        "subsample": float(_choice_maybe_scalar(rng, [0.8, 0.9, 1.0])),
        "colsample_bytree": float(_choice_maybe_scalar(rng, [0.7, 0.8, 0.9, 1.0])),
        "reg_lambda": float(_choice_maybe_scalar(rng, [1.0, 2.0, 3.0, 5.0])),
        "tree_method": "hist",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-name", type=str, default="tuned_xgb_advanced")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--refit-on-train-val", action="store_true")
    args = parser.parse_args()

    alpha = float(args.alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0,1], got {alpha}")

    out_dir = OUTPUT_DIR / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    X_train, feature_cfg = _build_advanced_feature_matrix(train_df)
    X_val, _ = _build_advanced_feature_matrix(val_df)
    X_test, _ = _build_advanced_feature_matrix(test_df)

    y_train = train_df[TARGET_COLS].copy()
    y_val = val_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    rng = np.random.default_rng(args.seed)

    results: list[dict] = []
    best: dict | None = None
    best_score: float | None = None

    for t in range(args.n_trials):
        params = _sample_params(rng)
        cfg = XgbBaselineConfig(random_state=args.seed, **params)
        model = XgbDamageBaseline(cfg)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        y_num_true = y_val["num_damages"].astype(int).to_numpy()
        y_num_pred = model.predict_num_damages(X_val)
        y_pos_pred = model.predict_positions(X_val)
        m = compute_damage_metrics(
            y_num_true=y_num_true,
            y_num_pred=y_num_pred,
            y_pos_true=y_val[TARGET_COLS[1:]],
            y_pos_pred=y_pos_pred,
        )

        f1 = float(m.num_damages_f1_macro)
        mae = float(m.pos_mae_overall)
        mae_score = 1.0 / (1.0 + mae)
        score = alpha * f1 + (1.0 - alpha) * mae_score

        row = {
            "trial": t,
            "score": score,
            "num_damages_f1_macro": f1,
            "num_damages_accuracy": float(m.num_damages_accuracy),
            "pos_mae_overall": mae,
            "pos_rmse_overall": float(m.pos_rmse_overall),
            **params,
        }
        results.append(row)

        if best_score is None or score > best_score:
            best_score = score
            best = row
            print(f"[{t+1}/{args.n_trials}] New best: f1_macro={f1:.4f}, pos_mae={mae:.4f}, score={score:.4f}")

    results_df = pd.DataFrame(results).sort_values(by=["score"], ascending=False)
    results_path = out_dir / "tuning_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    if best is None:
        raise RuntimeError("No tuning results produced.")

    best_params = {k: best[k] for k in ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "reg_lambda", "tree_method"]}
    best_cfg = XgbBaselineConfig(random_state=args.seed, **best_params)
    best_model = XgbDamageBaseline(best_cfg)

    if args.refit_on_train_val:
        trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        X_fit, _ = _build_advanced_feature_matrix(trainval_df)
        y_fit = trainval_df[TARGET_COLS].copy()
        best_model.fit(X_fit, y_fit)
    else:
        best_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def eval_split(title: str, X: pd.DataFrame, y: pd.DataFrame) -> None:
        y_num_true = y["num_damages"].astype(int).to_numpy()
        y_num_pred = best_model.predict_num_damages(X)
        y_pos_pred = best_model.predict_positions(X)
        m = compute_damage_metrics(
            y_num_true=y_num_true,
            y_num_pred=y_num_pred,
            y_pos_true=y[TARGET_COLS[1:]],
            y_pos_pred=y_pos_pred,
        )
        print_damage_metrics(title, m)

    print("\n=== Best hyperparameters ===")
    print(best_params)
    print("\n=== Evaluation (TRAIN/VAL/TEST) ===")
    eval_split("TRAIN", X_train, y_train)
    eval_split("VAL", X_val, y_val)
    eval_split("TEST", X_test, y_test)

    artifact_path = out_dir / "artifact.joblib"
    joblib.dump(
        {
            "feature_config": feature_cfg,
            "model_config": asdict(best_cfg),
            "model": best_model,
            "feature_columns": X_train.columns.tolist(),
            "feature_builder": "baseline+wavelet+physics",
            "best_val_row": {k: best[k] for k in ["score", "num_damages_f1_macro", "pos_mae_overall", "pos_rmse_overall"]},
        },
        artifact_path,
    )
    print(f"\nSaved tuned XGB advanced artifact to: {artifact_path}")


if __name__ == "__main__":
    main()

