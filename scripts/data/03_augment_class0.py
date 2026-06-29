"""
Augment class-0 (no damage) samples using Gaussian noise and write
augmented train/val splits.

Usage:
    python scripts/data/03_augment_class0.py
    python scripts/data/03_augment_class0.py --n-samples 80 --noise-std-frac 0.01 --val-holdout 10
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
from src.data.augment import augment_class0_gaussian, validate_synthetic_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment class-0 samples for beam damage dataset.")
    parser.add_argument("--n-samples", type=int, default=60,
                        help="Total synthetic class-0 samples to generate (default: 60)")
    parser.add_argument("--noise-std-frac", type=float, default=0.01,
                        help="Mode vector noise σ as fraction of vector std (default: 0.01)")
    parser.add_argument("--freq-noise-std-frac", type=float, default=0.005,
                        help="Frequency noise σ as fraction of freq value (default: 0.005)")
    parser.add_argument("--val-holdout", type=int, default=8,
                        help="How many synthetic samples to put in val split (default: 8)")
    parser.add_argument("--test-holdout", type=int, default=5,
                        help="How many synthetic samples to put in test split (default: 5)")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # ---- load real data ----
    dataset_path = PROCESSED_DIR / "scenario_dataset.csv"
    full_df = pd.read_csv(dataset_path)
    full_df = full_df[full_df["num_modes_found"] == 4].copy().reset_index(drop=True)

    class0_df = full_df[full_df["num_damages"] == 0].copy().reset_index(drop=True)
    print(f"Real class-0 samples found: {len(class0_df)}")
    print(f"Full dataset size: {len(full_df)} (all classes)")

    if len(class0_df) == 0:
        raise RuntimeError("No real class-0 samples found in scenario_dataset.csv. "
                           "Run 01_build_dataset.py first.")

    # ---- generate synthetic samples ----
    print(f"\nGenerating {args.n_samples} synthetic class-0 samples "
          f"(mode_noise_std_frac={args.noise_std_frac}, "
          f"freq_noise_std_frac={args.freq_noise_std_frac}) ...")

    synthetic_df = augment_class0_gaussian(
        class0_df=class0_df,
        n_samples=args.n_samples,
        freq_noise_std_frac=args.freq_noise_std_frac,
        mode_noise_std_frac=args.noise_std_frac,
        random_state=args.random_state,
    )

    # ---- validate ----
    report = validate_synthetic_batch(synthetic_df, real_sample=class0_df.iloc[0])
    n_ok = report["all_ok"].sum()
    print(f"\nValidation report ({n_ok}/{len(report)} samples passed all checks):")
    print(f"  freq_ordered:   {report['freq_ordered'].sum()}/{len(report)}")
    print(f"  freq_ratio_ok:  {report['freq_ratio_ok'].sum()}/{len(report)}")
    print(f"  mode_std_ok:    {report['mode_std_ok'].sum()}/{len(report)}")

    failed = report[~report["all_ok"]]
    if len(failed) > 0:
        print(f"  WARNING: {len(failed)} samples failed checks — they will be dropped.")
        synthetic_df = synthetic_df[report["all_ok"]].reset_index(drop=True)
        print(f"  Remaining after filter: {len(synthetic_df)}")

    # ---- split synthetic into train / val / test portions ----
    n_test = min(args.test_holdout, len(synthetic_df) // 8)
    n_val = min(args.val_holdout, (len(synthetic_df) - n_test) // 5)
    n_train = len(synthetic_df) - n_val - n_test

    # Shuffle before splitting
    synthetic_df = synthetic_df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    syn_test = synthetic_df.iloc[:n_test].copy()
    syn_val = synthetic_df.iloc[n_test:n_test + n_val].copy()
    syn_train = synthetic_df.iloc[n_test + n_val:].copy()

    print(f"\nSynthetic split: {n_train} -> train, {n_val} -> val, {n_test} -> test")

    # ---- load original train / val / test splits ----
    train_orig = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_orig = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_orig = pd.read_csv(PROCESSED_DIR / "test.csv")

    train_aug = pd.concat([train_orig, syn_train], axis=0, ignore_index=True)
    train_aug = train_aug.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    val_aug = pd.concat([val_orig, syn_val], axis=0, ignore_index=True)
    val_aug = val_aug.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    test_aug = pd.concat([test_orig, syn_test], axis=0, ignore_index=True)
    test_aug = test_aug.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    # ---- also write augmented full dataset (for reference) ----
    full_aug = pd.concat([full_df, synthetic_df], axis=0, ignore_index=True)

    # ---- save ----
    out_train = PROCESSED_DIR / "train_augmented.csv"
    out_val = PROCESSED_DIR / "val_augmented.csv"
    out_test = PROCESSED_DIR / "test_augmented.csv"
    out_full = PROCESSED_DIR / "scenario_dataset_augmented.csv"

    train_aug.to_csv(out_train, index=False, encoding="utf-8-sig")
    val_aug.to_csv(out_val, index=False, encoding="utf-8-sig")
    test_aug.to_csv(out_test, index=False, encoding="utf-8-sig")
    full_aug.to_csv(out_full, index=False, encoding="utf-8-sig")

    print(f"\nSaved:")
    print(f"  {out_train}  ({len(train_aug)} rows)")
    print(f"  {out_val}    ({len(val_aug)} rows)")
    print(f"  {out_test}   ({len(test_aug)} rows)")
    print(f"  {out_full}   ({len(full_aug)} rows)")

    # ---- class distribution summary ----
    print("\nClass distribution in train_augmented:")
    print(train_aug["num_damages"].value_counts(dropna=False).sort_index().to_string())
    print("\nClass distribution in val_augmented:")
    print(val_aug["num_damages"].value_counts(dropna=False).sort_index().to_string())
    print("\nClass distribution in test_augmented:")
    print(test_aug["num_damages"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
