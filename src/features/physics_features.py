from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.baseline_features import (
    FREQ_COLS,
    MODE_VECTOR_JSON_COLS,
    parse_mode_vector_json,
)


def _zero_crossing_count(vec: np.ndarray) -> int:
    signs = np.sign(vec)
    return int(np.sum(signs[:-1] * signs[1:] < 0))


def _peak_count(vec: np.ndarray) -> int:
    # Simple local maxima count without extra deps.
    if len(vec) < 3:
        return 0
    return int(np.sum((vec[1:-1] > vec[:-2]) & (vec[1:-1] > vec[2:])))


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    abs_arr = np.abs(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "abs_mean": float(np.mean(abs_arr)),
        "abs_max": float(np.max(abs_arr)),
        "energy": float(np.sum(arr**2)),
    }


@dataclass(frozen=True)
class PhysicsFeatureConfig:
    include_freq: bool = True
    include_gradient_stats: bool = True
    include_curvature_stats: bool = True
    include_shape_descriptors: bool = True


def build_physics_feature_matrix(
    df: pd.DataFrame,
    cfg: PhysicsFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Physics-inspired features from mode shapes.

    Features include:
    - optional modal frequencies
    - gradient/slope statistics
    - curvature statistics (2nd derivative proxy)
    - zero-crossing count, peak count, crest factor, symmetry score
    """
    cfg = cfg or PhysicsFeatureConfig()

    missing_cols = []
    if cfg.include_freq:
        missing_cols += [c for c in FREQ_COLS if c not in df.columns]
    missing_cols += [c for c in MODE_VECTOR_JSON_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(set(missing_cols))}")

    out: dict[str, np.ndarray | pd.Series] = {}

    if cfg.include_freq:
        for c in FREQ_COLS:
            out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for mode_idx, col in enumerate(MODE_VECTOR_JSON_COLS, start=1):
        parsed = [parse_mode_vector_json(v) for v in df[col].tolist()]

        if cfg.include_gradient_stats:
            rows = [_stats(np.gradient(vec)) for vec in parsed]
            stats_df = pd.DataFrame(rows)
            for stat_name in stats_df.columns:
                out[f"mode_{mode_idx}_grad_{stat_name}"] = stats_df[stat_name].astype(float)

        if cfg.include_curvature_stats:
            rows = [_stats(np.gradient(np.gradient(vec))) for vec in parsed]
            stats_df = pd.DataFrame(rows)
            for stat_name in stats_df.columns:
                out[f"mode_{mode_idx}_curv_{stat_name}"] = stats_df[stat_name].astype(float)

        if cfg.include_shape_descriptors:
            zero_crossings = []
            peak_counts = []
            crest_factors = []
            symmetry_scores = []
            end_diff_abs = []

            for vec in parsed:
                abs_vec = np.abs(vec)
                rms = float(np.sqrt(np.mean(vec**2))) + 1e-12
                mirrored = vec[::-1]

                zero_crossings.append(float(_zero_crossing_count(vec)))
                peak_counts.append(float(_peak_count(vec)))
                crest_factors.append(float(np.max(abs_vec) / rms))
                symmetry_scores.append(float(np.mean(np.abs(vec - mirrored))))
                end_diff_abs.append(float(abs(vec[0] - vec[-1])))

            out[f"mode_{mode_idx}_zero_crossings"] = np.asarray(zero_crossings, dtype=float)
            out[f"mode_{mode_idx}_peak_count"] = np.asarray(peak_counts, dtype=float)
            out[f"mode_{mode_idx}_crest_factor"] = np.asarray(crest_factors, dtype=float)
            out[f"mode_{mode_idx}_symmetry_score"] = np.asarray(symmetry_scores, dtype=float)
            out[f"mode_{mode_idx}_end_diff_abs"] = np.asarray(end_diff_abs, dtype=float)

    X = pd.DataFrame(out)
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs produced in physics feature matrix; example columns: {nan_cols[:10]}")

    return X

