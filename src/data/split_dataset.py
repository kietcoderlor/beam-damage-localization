from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if "config_id" not in df.columns:
        raise ValueError("Missing required column: config_id")

    if df["config_id"].duplicated().any():
        raise ValueError("config_id must be unique in scenario-level dataset")

    stratify_values = (
        df[stratify_col]
        if (stratify_col is not None and stratify_col in df.columns)
        else None
    )

    try:
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_values,
        )
    except ValueError as e:
        if stratify_values is None:
            raise
        counts = stratify_values.value_counts(dropna=False).sort_index()
        raise ValueError(
            f"Stratified split failed for stratify_col={stratify_col!r}. "
            f"Class counts:\n{counts.to_string()}\n\nOriginal error: {e}"
        ) from e

    temp_ratio = val_size + test_size
    val_ratio_within_temp = val_size / temp_ratio

    temp_stratify = (
        temp_df[stratify_col]
        if (stratify_col is not None and stratify_col in temp_df.columns)
        else None
    )

    try:
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_within_temp,
            random_state=random_state,
            shuffle=True,
            stratify=temp_stratify,
        )
    except ValueError as e:
        if temp_stratify is None:
            raise
        counts = temp_stratify.value_counts(dropna=False).sort_index()
        raise ValueError(
            f"Stratified split failed in temp split for stratify_col={stratify_col!r}. "
            f"Temp class counts:\n{counts.to_string()}\n\nOriginal error: {e}"
        ) from e

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print(f"Rows: {len(df)}")
    if "num_damages" in df.columns:
        print("num_damages distribution:")
        print(df["num_damages"].value_counts(dropna=False).sort_index())