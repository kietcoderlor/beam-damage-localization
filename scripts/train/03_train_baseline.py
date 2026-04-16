from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

import sys
from pathlib import Path

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
    df = df[df["num_modes_found"] == 4].copy().reset_index(drop=True)
    return df


def main() -> None:
    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    print("Loaded splits:")
    print(" - train:", train_df.shape)
    print(" - val:  ", val_df.shape)
    print(" - test: ", test_df.shape)

    # Features
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

    # Model (no extra deps)
    model = RfDamageBaseline(RfBaselineConfig(random_state=RANDOM_SEED))
    model.fit(X_train, y_train)

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

    # Save artifacts
    out_dir = OUTPUT_DIR / "baseline_rf"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "feature_config": feat_cfg,
            "model_config": model.cfg,
            "model": model,
            "feature_columns": X_train.columns.tolist(),
        },
        out_dir / "artifact.joblib",
    )

    print(f"\nSaved artifact to: {out_dir / 'artifact.joblib'}")


if __name__ == "__main__":
    main()

