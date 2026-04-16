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
from src.models.baseline_sklearn import RfBaselineConfig, RfDamageBaseline  # noqa: E402


TARGET_COLS = ["num_damages", "damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    # chỉ giữ các sample đủ 4 mode
    df = df[df["num_modes_found"] == 4].copy().reset_index(drop=True)
    return df


def _choice_maybe_scalar(rng: np.random.Generator, seq: list) -> object:
    v = rng.choice(seq)
    # rng.choice may return numpy scalars (need .item()) or plain Python objects (no .item()).
    return v.item() if isinstance(v, np.generic) else v


def _sample_params(rng: np.random.Generator) -> dict:
    # Keep values realistic for this small dataset.
    # Note: max_features accepts both categorical strings and numeric proportions.
    n_estimators = int(_choice_maybe_scalar(rng, [300, 450, 600, 750, 900, 1200]))
    max_depth = _choice_maybe_scalar(rng, [None, 6, 8, 10, 12, 16, 20, 24])
    if max_depth is not None:
        max_depth = int(max_depth)
    max_features = _choice_maybe_scalar(rng, ["sqrt", "log2", 0.25, 0.5, 0.75, None])
    min_samples_leaf = int(_choice_maybe_scalar(rng, [1, 2, 3, 4, 5, 8, 10]))
    min_samples_split = int(_choice_maybe_scalar(rng, [2, 3, 4, 5, 6, 8, 10]))
    bootstrap = bool(_choice_maybe_scalar(rng, [True, False]))
    class_weight = _choice_maybe_scalar(rng, ["balanced_subsample", None])

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
        "class_weight": class_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-name", type=str, default="tuned_rf")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Trade-off between num_damages macro-F1 and position MAE (0..1).",
    )
    parser.add_argument("--refit-on-train-val", action="store_true", help="Refit best model on train+val before final test.")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    # Feature extraction is deterministic; do it once for speed.
    feat_cfg = BaselineFeatureConfig(
        resample_len=32,
        include_freq=True,
        include_mode_stats=True,
        include_resampled_vectors=True,
    )
    X_train = build_baseline_feature_matrix(train_df, feat_cfg)
    X_val = build_baseline_feature_matrix(val_df, feat_cfg)
    X_test = build_baseline_feature_matrix(test_df, feat_cfg)

    y_train = train_df[TARGET_COLS].copy()
    y_val = val_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    rng = np.random.default_rng(args.seed)
    alpha = float(args.alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0,1], got {alpha}")

    results: list[dict] = []
    best: dict | None = None
    best_key: float | None = None  # higher score is better

    for t in range(args.n_trials):
        params = _sample_params(rng)
        cfg = RfBaselineConfig(random_state=args.seed, **params)
        model = RfDamageBaseline(cfg)
        model.fit(X_train, y_train)

        y_num_true = y_val["num_damages"].astype(int).to_numpy()
        y_num_pred = model.predict_num_damages(X_val)
        y_pos_pred = model.predict_positions(X_val)

        m = compute_damage_metrics(
            y_num_true=y_num_true,
            y_num_pred=y_num_pred,
            y_pos_true=y_val[TARGET_COLS[1:]],
            y_pos_pred=y_pos_pred,
        )

        row = {
            "trial": t,
            "num_damages_f1_macro": m.num_damages_f1_macro,
            "num_damages_accuracy": m.num_damages_accuracy,
            "pos_mae_overall": m.pos_mae_overall,
            "pos_rmse_overall": m.pos_rmse_overall,
            **params,
        }
        results.append(row)

        f1 = float(m.num_damages_f1_macro)
        mae = float(m.pos_mae_overall)
        # Convert MAE to a score in (0,1] so it is comparable to F1.
        # - lower MAE => higher mae_score
        mae_score = 1.0 / (1.0 + mae)
        score = alpha * f1 + (1.0 - alpha) * mae_score

        row["mae_score"] = mae_score
        row["score"] = score

        if best_key is None or score > best_key:
            best_key = score
            best = row
            print(
                f"[{t+1}/{args.n_trials}] New best: f1_macro={f1:.4f}, pos_mae={mae:.4f}, score={score:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(by=["score"], ascending=[False])
    results_path = out_dir / "tuning_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    if best is None:
        raise RuntimeError("No tuning results produced.")

    # Prepare best config (remove non-config fields).
    best_params = {k: best[k] for k in ["n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf", "bootstrap", "class_weight"]}
    best_cfg = RfBaselineConfig(random_state=args.seed, **best_params)

    # Refit best model.
    if args.refit_on_train_val:
        trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        X_train_fit = build_baseline_feature_matrix(trainval_df, feat_cfg)
        y_train_fit = trainval_df[TARGET_COLS].copy()
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    best_model = RfDamageBaseline(best_cfg)
    best_model.fit(X_train_fit, y_train_fit)

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

    # Save artifact.
    artifact_path = out_dir / "artifact.joblib"
    joblib.dump(
        {
            "feature_config": feat_cfg,
            "model_config": asdict(best_cfg),
            "model": best_model,
            "feature_columns": X_train.columns.tolist(),
            "best_val_row": {k: best[k] for k in ["num_damages_f1_macro", "pos_mae_overall", "num_damages_accuracy", "pos_rmse_overall"]},
        },
        artifact_path,
    )
    print(f"\nSaved tuned RF artifact to: {artifact_path}")

    # Quick best summary file for convenience.
    best_summary_path = out_dir / "best_summary.txt"
    best_summary_path.write_text(
        "\n".join(
            [
                "Best hyperparameters:",
                str(best_params),
                "",
                "Best validation metrics:",
                f"num_damages_f1_macro={best['num_damages_f1_macro']:.6f}",
                f"num_damages_accuracy={best['num_damages_accuracy']:.6f}",
                f"pos_mae_overall={best['pos_mae_overall']:.6f}",
                f"pos_rmse_overall={best['pos_rmse_overall']:.6f}",
                "",
                f"tuning_results.csv: {results_path}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

