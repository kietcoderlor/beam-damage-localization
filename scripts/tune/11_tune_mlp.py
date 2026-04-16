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
from src.features.baseline_features import BaselineFeatureConfig, build_baseline_feature_matrix  # noqa: E402
from src.models.baseline_mlp import MlpBaselineConfig, MlpDamageBaseline  # noqa: E402


TARGET_COLS = ["num_damages", "damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    return df[df["num_modes_found"] == 4].copy().reset_index(drop=True)


def _choice_maybe_scalar(rng: np.random.Generator, seq: list) -> object:
    v = rng.choice(seq)
    return v.item() if isinstance(v, np.generic) else v


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "hidden_sizes": (
            int(_choice_maybe_scalar(rng, [64, 96, 128, 192])),
            int(_choice_maybe_scalar(rng, [32, 48, 64, 96])),
        ),
        "alpha": float(_choice_maybe_scalar(rng, [1e-5, 1e-4, 5e-4, 1e-3])),
        "max_iter_clf": int(_choice_maybe_scalar(rng, [400, 600, 800, 1000])),
        "max_iter_reg": int(_choice_maybe_scalar(rng, [600, 800, 1000, 1400])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-name", type=str, default="tuned_mlp_balanced")
    parser.add_argument("--alpha-score", type=float, default=0.5)
    parser.add_argument("--refit-on-train-val", action="store_true")
    args = parser.parse_args()

    score_alpha = float(args.alpha_score)
    if not (0.0 <= score_alpha <= 1.0):
        raise ValueError(f"--alpha-score must be in [0,1], got {score_alpha}")

    out_dir = OUTPUT_DIR / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

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
    best = None
    best_score = None
    results: list[dict] = []

    for t in range(args.n_trials):
        params = _sample_params(rng)
        cfg = MlpBaselineConfig(random_state=args.seed, **params)
        model = MlpDamageBaseline(cfg)
        model.fit(X_train, y_train)

        m = compute_damage_metrics(
            y_num_true=y_val["num_damages"].astype(int).to_numpy(),
            y_num_pred=model.predict_num_damages(X_val),
            y_pos_true=y_val[TARGET_COLS[1:]],
            y_pos_pred=model.predict_positions(X_val),
        )
        f1 = float(m.num_damages_f1_macro)
        mae = float(m.pos_mae_overall)
        mae_score = 1.0 / (1.0 + mae)
        score = score_alpha * f1 + (1.0 - score_alpha) * mae_score
        row = {
            "trial": t,
            "score": score,
            "num_damages_f1_macro": f1,
            "num_damages_accuracy": float(m.num_damages_accuracy),
            "pos_mae_overall": mae,
            "pos_rmse_overall": float(m.pos_rmse_overall),
            "hidden_sizes": str(params["hidden_sizes"]),
            "alpha": params["alpha"],
            "max_iter_clf": params["max_iter_clf"],
            "max_iter_reg": params["max_iter_reg"],
        }
        results.append(row)
        if best_score is None or score > best_score:
            best_score = score
            best = row
            print(f"[{t+1}/{args.n_trials}] New best: f1_macro={f1:.4f}, pos_mae={mae:.4f}, score={score:.4f}")

    if best is None:
        raise RuntimeError("No tuning results produced.")

    pd.DataFrame(results).sort_values(by=["score"], ascending=False).to_csv(
        out_dir / "tuning_results.csv", index=False, encoding="utf-8-sig"
    )

    hidden_sizes = tuple(int(x.strip()) for x in best["hidden_sizes"].strip("()").split(",") if x.strip())
    best_cfg = MlpBaselineConfig(
        random_state=args.seed,
        hidden_sizes=hidden_sizes,
        alpha=float(best["alpha"]),
        max_iter_clf=int(best["max_iter_clf"]),
        max_iter_reg=int(best["max_iter_reg"]),
    )

    best_model = MlpDamageBaseline(best_cfg)
    if args.refit_on_train_val:
        trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        X_fit = build_baseline_feature_matrix(trainval_df, feat_cfg)
        y_fit = trainval_df[TARGET_COLS].copy()
        best_model.fit(X_fit, y_fit)
    else:
        best_model.fit(X_train, y_train)

    def eval_split(title: str, X: pd.DataFrame, y: pd.DataFrame) -> None:
        m = compute_damage_metrics(
            y_num_true=y["num_damages"].astype(int).to_numpy(),
            y_num_pred=best_model.predict_num_damages(X),
            y_pos_true=y[TARGET_COLS[1:]],
            y_pos_pred=best_model.predict_positions(X),
        )
        print_damage_metrics(title, m)

    print("\n=== Best hyperparameters ===")
    print(asdict(best_cfg))
    print("\n=== Evaluation (TRAIN/VAL/TEST) ===")
    eval_split("TRAIN", X_train, y_train)
    eval_split("VAL", X_val, y_val)
    eval_split("TEST", X_test, y_test)

    joblib.dump(
        {
            "feature_config": feat_cfg,
            "model_config": asdict(best_cfg),
            "model": best_model,
            "feature_columns": X_train.columns.tolist(),
            "best_val_row": {k: best[k] for k in ["score", "num_damages_f1_macro", "pos_mae_overall", "pos_rmse_overall"]},
        },
        out_dir / "artifact.joblib",
    )


if __name__ == "__main__":
    main()

