from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


MODE_VECTOR_JSON_COLS = [
    "mode_1_vector_json",
    "mode_2_vector_json",
    "mode_3_vector_json",
    "mode_4_vector_json",
]

FREQ_COLS = ["freq_mode_1", "freq_mode_2", "freq_mode_3", "freq_mode_4"]


def _ensure_1d_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape={arr.shape}")
    return arr


def parse_mode_vector_json(value: str) -> np.ndarray:
    """
    Parse a JSON-encoded mode vector into a 1D float numpy array.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):  # type: ignore[unreachable]
        raise ValueError("Mode vector JSON is missing (NaN/None).")
    if not isinstance(value, str):
        raise TypeError(f"Mode vector JSON must be str, got {type(value)}")
    parsed = json.loads(value)
    return _ensure_1d_float_array(parsed)


def resample_1d(vec: np.ndarray, n: int) -> np.ndarray:
    """
    Deterministically resample a 1D vector to length n using linear interpolation.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    vec = _ensure_1d_float_array(vec)
    if len(vec) == 0:
        raise ValueError("Cannot resample empty vector")
    if len(vec) == n:
        return vec.astype(float, copy=False)

    x_old = np.linspace(0.0, 1.0, num=len(vec))
    x_new = np.linspace(0.0, 1.0, num=n)
    return np.interp(x_new, x_old, vec).astype(float)


def vector_basic_stats(vec: np.ndarray) -> dict[str, float]:
    vec = _ensure_1d_float_array(vec)
    abs_vec = np.abs(vec)
    return {
        "mean": float(np.mean(vec)),
        "std": float(np.std(vec)),
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "ptp": float(np.ptp(vec)),
        "abs_mean": float(np.mean(abs_vec)),
        "abs_max": float(np.max(abs_vec)),
        "l1": float(np.sum(abs_vec)),
        "l2": float(np.sqrt(np.sum(vec**2))),
        "energy": float(np.sum(vec**2)),
    }


@dataclass(frozen=True)
class BaselineFeatureConfig:
    resample_len: int = 32
    include_freq: bool = True
    include_mode_stats: bool = True
    include_resampled_vectors: bool = True


def build_baseline_feature_matrix(
    df: pd.DataFrame,
    cfg: BaselineFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Baseline features:
    - modal frequencies (4 scalars)
    - per-mode basic stats (mean/std/min/max/energy...)
    - per-mode resampled vectors (fixed length)

    IMPORTANT: This uses only per-sample information (no global fitting),
    keeping leakage risk low.
    """
    cfg = cfg or BaselineFeatureConfig()

    missing_cols = []
    if cfg.include_freq:
        missing_cols += [c for c in FREQ_COLS if c not in df.columns]
    if cfg.include_mode_stats or cfg.include_resampled_vectors:
        missing_cols += [c for c in MODE_VECTOR_JSON_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(set(missing_cols))}")

    out: dict[str, Iterable[float]] = {}

    if cfg.include_freq:
        for c in FREQ_COLS:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    if cfg.include_mode_stats or cfg.include_resampled_vectors:
        # parse all vectors once to avoid double json.loads work
        parsed_vectors: dict[str, list[np.ndarray]] = {c: [] for c in MODE_VECTOR_JSON_COLS}
        for col in MODE_VECTOR_JSON_COLS:
            for v in df[col].tolist():
                parsed_vectors[col].append(parse_mode_vector_json(v))

        if cfg.include_mode_stats:
            for mode_idx, col in enumerate(MODE_VECTOR_JSON_COLS, start=1):
                stats_rows = [vector_basic_stats(vec) for vec in parsed_vectors[col]]
                stats_df = pd.DataFrame(stats_rows)
                for stat_name in stats_df.columns:
                    out[f"mode_{mode_idx}_{stat_name}"] = stats_df[stat_name].astype(float)

        if cfg.include_resampled_vectors:
            for mode_idx, col in enumerate(MODE_VECTOR_JSON_COLS, start=1):
                resampled = [resample_1d(vec, cfg.resample_len) for vec in parsed_vectors[col]]
                mat = np.vstack(resampled)  # (n_samples, resample_len)
                for j in range(cfg.resample_len):
                    out[f"mode_{mode_idx}_v{j:02d}"] = mat[:, j].astype(float)

    X = pd.DataFrame(out)
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs produced in feature matrix; columns with NaNs: {nan_cols[:10]}")

    return X

