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
from src.models.cnn1d import Cnn1dBaselineConfig, Cnn1dDamageBaseline  # noqa: E402


TARGET_COLS = ["num_damages", "damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    return df[df["num_modes_found"] == 4].copy().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", type=str, default="baseline_cnn1d")
    parser.add_argument("--resample-len", type=int, default=191)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    print("Loaded splits:")
    print(" - train:", train_df.shape)
    print(" - val:  ", val_df.shape)
    print(" - test: ", test_df.shape)

    cfg = Cnn1dBaselineConfig(
        random_state=RANDOM_SEED,
        resample_len=args.resample_len,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    model = Cnn1dDamageBaseline(cfg)
    model.fit(train_df, train_df[TARGET_COLS].copy(), X_val=val_df, y_val=val_df[TARGET_COLS].copy())

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

    eval_split("TRAIN", train_df, train_df[TARGET_COLS].copy())
    eval_split("VAL", val_df, val_df[TARGET_COLS].copy())
    eval_split("TEST", test_df, test_df[TARGET_COLS].copy())

    out_dir = OUTPUT_DIR / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "artifact.joblib"
    joblib.dump(
        {
            "feature_config": asdict(cfg),
            "model_config": asdict(cfg),
            "model": model,
            "feature_builder": "cnn_raw_modal",
        },
        artifact_path,
    )
    print(f"\nSaved artifact to: {artifact_path}")


if __name__ == "__main__":
    main()

