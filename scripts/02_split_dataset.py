import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import PROCESSED_DIR  # noqa: E402
from src.data.split_dataset import split_dataset, print_split_summary  # noqa: E402


def main():
    input_path = PROCESSED_DIR / "scenario_dataset.csv"

    df = pd.read_csv(input_path)

    # chỉ giữ các sample đủ 4 mode
    df = df[df["num_modes_found"] == 4].copy().reset_index(drop=True)

    print("Full dataset (after num_modes_found==4 filter):", df.shape)
    if "num_damages" in df.columns:
        print("num_damages distribution (full):")
        print(df["num_damages"].value_counts(dropna=False).sort_index())

    # Rare-class handling for robust stratification:
    # class num_damages==0 currently has only 1 sample -> cannot be stratified.
    # Keep it in train so the model still sees this class, and stratify the rest.
    rare_df = df[df["num_damages"] == 0].copy().reset_index(drop=True)
    core_df = df[df["num_damages"] != 0].copy().reset_index(drop=True)

    train_core, val_df, test_df = split_dataset(
        core_df,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
        stratify_col="num_damages",
    )
    train_df = (
        pd.concat([train_core, rare_df], axis=0)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    print("\nRare-class handling:")
    print(f" - rare num_damages==0 rows forced into TRAIN: {len(rare_df)}")

    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"Saved train: {train_path}")
    print(f"Saved val:   {val_path}")
    print(f"Saved test:  {test_path}")

    print_split_summary("TRAIN", train_df)
    print_split_summary("VAL", val_df)
    print_split_summary("TEST", test_df)


if __name__ == "__main__":
    main()