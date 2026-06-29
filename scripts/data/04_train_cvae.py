"""
Train a Conditional VAE on all training data, then generate synthetic
class-0 (no damage) samples and optionally retrain the XGBoost model.

Usage:
    python scripts/data/04_train_cvae.py
    python scripts/data/04_train_cvae.py --n-samples 60 --epochs 500 --latent-dim 16
    python scripts/data/04_train_cvae.py --n-samples 60 --val-holdout 8

Outputs
-------
outputs/cvae_class0/
  cvae_model.pt              Trained CVAE weights
  synthetic_class0.csv       Generated class-0 rows
  train_loss.txt             Per-epoch loss

data/processed/
  train_cvae.csv             Original train + CVAE class-0 samples
  val_cvae.csv               Original val  + holdout CVAE class-0 samples
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
import torch  # noqa: E402

from src.config.settings import OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED  # noqa: E402
from src.data.augment import validate_synthetic_batch  # noqa: E402
from src.models.cvae import CVAE, CVAEConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CVAE and generate synthetic class-0 data.")
    parser.add_argument("--n-samples", type=int, default=60,
                        help="Synthetic class-0 samples to generate (default: 60)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=0.01)
    parser.add_argument("--anneal-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-holdout", type=int, default=8,
                        help="CVAE-generated samples placed into val split (default: 8)")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # ---- load training data ----
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    train_df = train_df[train_df["num_modes_found"] == 4].copy().reset_index(drop=True)

    val_orig_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    val_orig_df = val_orig_df[val_orig_df["num_modes_found"] == 4].copy().reset_index(drop=True)

    print(f"Training CVAE on {len(train_df)} samples "
          f"({(train_df['num_damages']==0).sum()} class-0) ...")

    # ---- build and train CVAE ----
    cfg = CVAEConfig(
        latent_dim=args.latent_dim,
        beta_max=args.beta_max,
        anneal_epochs=args.anneal_epochs,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    cvae = CVAE(cfg)
    losses = cvae.fit(train_df)

    # ---- save model ----
    out_dir = OUTPUT_DIR / "cvae_class0"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "cvae_model.pt"
    torch.save(
        {
            "state_dict": cvae.state_dict(),
            "config": cfg,
            "x_mean": cvae._x_mean,
            "x_std": cvae._x_std,
        },
        model_path,
    )
    (out_dir / "train_loss.txt").write_text("\n".join(f"{l:.8f}" for l in losses))
    print(f"CVAE saved -> {model_path}")
    print(f"Final training loss: {losses[-1]:.6f}")

    # ---- generate synthetic class-0 samples ----
    print(f"\nGenerating {args.n_samples} class-0 samples from CVAE ...")
    synthetic_df = cvae.generate_class0(n_samples=args.n_samples, random_state=args.random_state)

    # ---- validate ----
    class0_real = train_df[train_df["num_damages"] == 0]
    if len(class0_real) > 0:
        report = validate_synthetic_batch(synthetic_df, real_sample=class0_real.iloc[0])
        n_ok = report["all_ok"].sum()
        print(f"\nValidation ({n_ok}/{len(report)} passed all checks):")
        print(f"  freq_ordered:   {report['freq_ordered'].sum()}/{len(report)}")
        print(f"  freq_ratio_ok:  {report['freq_ratio_ok'].sum()}/{len(report)}")
        print(f"  mode_std_ok:    {report['mode_std_ok'].sum()}/{len(report)}")

        failed = report[~report["all_ok"]]
        if len(failed) > 0:
            print(f"  Dropping {len(failed)} samples that failed validation.")
            synthetic_df = synthetic_df[report["all_ok"]].reset_index(drop=True)
    else:
        print("(Skipping plausibility check — no real class-0 sample available)")

    # ---- save generated samples ----
    synth_path = out_dir / "synthetic_class0.csv"
    synthetic_df.to_csv(synth_path, index=False, encoding="utf-8-sig")
    print(f"Synthetic samples saved -> {synth_path}")

    # ---- build augmented train / val splits ----
    n_val = min(args.val_holdout, len(synthetic_df) // 5)
    synthetic_df = synthetic_df.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
    syn_val = synthetic_df.iloc[:n_val].copy()
    syn_train = synthetic_df.iloc[n_val:].copy()

    train_cvae = (
        pd.concat([train_df, syn_train], axis=0, ignore_index=True)
        .sample(frac=1.0, random_state=args.random_state)
        .reset_index(drop=True)
    )
    val_cvae = (
        pd.concat([val_orig_df, syn_val], axis=0, ignore_index=True)
        .sample(frac=1.0, random_state=args.random_state)
        .reset_index(drop=True)
    )

    train_cvae.to_csv(PROCESSED_DIR / "train_cvae.csv", index=False, encoding="utf-8-sig")
    val_cvae.to_csv(PROCESSED_DIR / "val_cvae.csv", index=False, encoding="utf-8-sig")

    print(f"\nAugmented splits saved:")
    print(f"  train_cvae.csv  ({len(train_cvae)} rows)")
    print(f"  val_cvae.csv    ({len(val_cvae)} rows)")

    print("\nClass distribution in train_cvae:")
    print(train_cvae["num_damages"].value_counts(dropna=False).sort_index().to_string())
    print("\nClass distribution in val_cvae:")
    print(val_cvae["num_damages"].value_counts(dropna=False).sort_index().to_string())

    print("\nDone. To retrain XGBoost with CVAE data, run:")
    print("  python scripts/train/13_train_with_augmented.py "
          "--output-name xgb_aug_cvae --train-csv train_cvae.csv --val-csv val_cvae.csv")


if __name__ == "__main__":
    main()
