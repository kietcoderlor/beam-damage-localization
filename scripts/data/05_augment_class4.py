"""
Augment class-4 (4-damage) samples using Gaussian noise and append them to
the existing augmented train/val/test splits produced by 03_augment_class0.py.

Run AFTER 03_augment_class0.py so that test_augmented.csv already exists.

Usage:
    python scripts/data/05_augment_class4.py
    python scripts/data/05_augment_class4.py --n-samples 50 --test-holdout 5 --val-holdout 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_project_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "src").exists():
            return candidate
    raise RuntimeError("Cannot find project root.")


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config.settings import PROCESSED_DIR, RANDOM_SEED  # noqa: E402
from src.data.augment import augment_by_class_gaussian  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment class-4 samples and append to augmented splits.")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Total synthetic class-4 samples to generate (default: 50)")
    parser.add_argument("--noise-std-frac", type=float, default=0.01,
                        help="Mode vector noise sigma as fraction of vector std (default: 0.01)")
    parser.add_argument("--freq-noise-std-frac", type=float, default=0.005,
                        help="Frequency noise sigma as fraction of freq value (default: 0.005)")
    parser.add_argument("--val-holdout", type=int, default=5,
                        help="Synthetic class-4 samples placed in val split (default: 5)")
    parser.add_argument("--test-holdout", type=int, default=5,
                        help="Synthetic class-4 samples placed in test split (default: 5)")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # ---- check prerequisite files exist ----
    for fname in ("train_augmented.csv", "val_augmented.csv", "test_augmented.csv"):
        p = PROCESSED_DIR / fname
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/data/03_augment_class0.py --test-holdout 5 first."
            )

    # ---- load real class-4 samples from full dataset ----
    dataset_path = PROCESSED_DIR / "scenario_dataset.csv"
    full_df = pd.read_csv(dataset_path)
    full_df = full_df[full_df["num_modes_found"] == 4].copy().reset_index(drop=True)

    class4_df = full_df[full_df["num_damages"] == 4].copy().reset_index(drop=True)
    print(f"Real class-4 samples found: {len(class4_df)}")

    if len(class4_df) == 0:
        raise RuntimeError("No real class-4 samples found. Cannot augment.")

    # ---- generate synthetic samples ----
    print(f"\nGenerating {args.n_samples} synthetic class-4 samples "
          f"(mode_noise_std_frac={args.noise_std_frac}, "
          f"freq_noise_std_frac={args.freq_noise_std_frac}) ...")

    synthetic_df = augment_by_class_gaussian(
        source_df=class4_df,
        target_class=4,
        n_samples=args.n_samples,
        freq_noise_std_frac=args.freq_noise_std_frac,
        mode_noise_std_frac=args.noise_std_frac,
        random_state=args.random_state,
    )

    print(f"Generated {len(synthetic_df)} synthetic class-4 rows.")

    # ---- split into test / val / train portions ----
    n_test = min(args.test_holdout, len(synthetic_df) // 5)
    n_val = min(args.val_holdout, (len(synthetic_df) - n_test) // 5)
    n_train = len(synthetic_df) - n_test - n_val

    synthetic_df = synthetic_df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    syn_test = synthetic_df.iloc[:n_test].copy()
    syn_val = synthetic_df.iloc[n_test:n_test + n_val].copy()
    syn_train = synthetic_df.iloc[n_test + n_val:].copy()

    print(f"Synthetic split: {n_train} -> train, {n_val} -> val, {n_test} -> test")

    # ---- append to existing augmented splits ----
    out_train = PROCESSED_DIR / "train_augmented.csv"
    out_val = PROCESSED_DIR / "val_augmented.csv"
    out_test = PROCESSED_DIR / "test_augmented.csv"

    def _append(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_rows], axis=0, ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
        combined.to_csv(path, index=False, encoding="utf-8-sig")
        return combined

    train_aug = _append(out_train, syn_train)
    val_aug = _append(out_val, syn_val)
    test_aug = _append(out_test, syn_test)

    print(f"\nUpdated augmented splits:")
    print(f"  {out_train}  ({len(train_aug)} rows)")
    print(f"  {out_val}    ({len(val_aug)} rows)")
    print(f"  {out_test}   ({len(test_aug)} rows)")

    # ---- class distribution summary ----
    print("\nClass distribution in train_augmented:")
    print(train_aug["num_damages"].value_counts(dropna=False).sort_index().to_string())
    print("\nClass distribution in val_augmented:")
    print(val_aug["num_damages"].value_counts(dropna=False).sort_index().to_string())
    print("\nClass distribution in test_augmented:")
    print(test_aug["num_damages"].value_counts(dropna=False).sort_index().to_string())

    print("\nDone. To retrain on the fully augmented dataset, run:")
    print("  python scripts/train/13_train_with_augmented.py "
          "--output-name xgb_aug_full --test-csv test_augmented.csv")


if __name__ == "__main__":
    main()
