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
from src.models.cnn1d import Cnn1dBaselineConfig, Cnn1dDamageBaseline  # noqa: E402


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
        "learning_rate": float(_choice_maybe_scalar(rng, [3e-4, 5e-4, 1e-3])),
        "weight_decay": float(_choice_maybe_scalar(rng, [1e-5, 1e-4, 5e-4])),
        "hidden_dim": int(_choice_maybe_scalar(rng, [32, 64, 96, 128])),
        "batch_size": int(_choice_maybe_scalar(rng, [16, 32, 64])),
        "max_epochs": int(_choice_maybe_scalar(rng, [60, 90, 120])),
        "patience": int(_choice_maybe_scalar(rng, [8, 12, 16])),
        "cls_loss_weight": float(_choice_maybe_scalar(rng, [1.0, 1.5, 2.0])),
        "reg_loss_weight": float(_choice_maybe_scalar(rng, [0.5, 1.0, 1.5])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-name", type=str, default="tuned_cnn1d_balanced")
    parser.add_argument("--alpha-score", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
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

    y_train = train_df[TARGET_COLS].copy()
    y_val = val_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    rng = np.random.default_rng(args.seed)
    best = None
    best_score = None
    results: list[dict] = []

    for t in range(args.n_trials):
        params = _sample_params(rng)
        cfg = Cnn1dBaselineConfig(random_state=args.seed, device=args.device, **params)
        model = Cnn1dDamageBaseline(cfg)
        model.fit(train_df, y_train, X_val=val_df, y_val=y_val)

        m = compute_damage_metrics(
            y_num_true=y_val["num_damages"].astype(int).to_numpy(),
            y_num_pred=model.predict_num_damages(val_df),
            y_pos_true=y_val[TARGET_COLS[1:]],
            y_pos_pred=model.predict_positions(val_df),
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
            **params,
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

    best_cfg = Cnn1dBaselineConfig(
        random_state=args.seed,
        device=args.device,
        batch_size=int(best["batch_size"]),
        max_epochs=int(best["max_epochs"]),
        learning_rate=float(best["learning_rate"]),
        weight_decay=float(best["weight_decay"]),
        patience=int(best["patience"]),
        hidden_dim=int(best["hidden_dim"]),
        cls_loss_weight=float(best["cls_loss_weight"]),
        reg_loss_weight=float(best["reg_loss_weight"]),
    )
    best_model = Cnn1dDamageBaseline(best_cfg)
    if args.refit_on_train_val:
        trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        best_model.fit(trainval_df, trainval_df[TARGET_COLS].copy())
    else:
        best_model.fit(train_df, y_train, X_val=val_df, y_val=y_val)

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
    eval_split("TRAIN", train_df, y_train)
    eval_split("VAL", val_df, y_val)
    eval_split("TEST", test_df, y_test)

    joblib.dump(
        {
            "feature_config": asdict(best_cfg),
            "model_config": asdict(best_cfg),
            "model": best_model,
            "feature_builder": "cnn_raw_modal",
            "best_val_row": {k: best[k] for k in ["score", "num_damages_f1_macro", "pos_mae_overall", "pos_rmse_overall"]},
        },
        out_dir / "artifact.joblib",
    )


if __name__ == "__main__":
    main()

