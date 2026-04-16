from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pywt

from src.features.baseline_features import (
    FREQ_COLS,
    MODE_VECTOR_JSON_COLS,
    parse_mode_vector_json,
)


def _coeff_stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    abs_arr = np.abs(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "abs_mean": float(np.mean(abs_arr)),
        "energy": float(np.sum(arr**2)),
        "abs_max": float(np.max(abs_arr)),
    }


@dataclass(frozen=True)
class WaveletFeatureConfig:
    wavelet: str = "db4"
    level: int = 3
    include_freq: bool = True
    include_raw_energy_ratios: bool = True
    include_wavelet_stats: bool = True


def build_wavelet_feature_matrix(
    df: pd.DataFrame,
    cfg: WaveletFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Wavelet-inspired features from each mode vector.

    Features:
    - optional modal frequencies
    - raw vector energy ratios (first/middle/last thirds)
    - discrete wavelet decomposition coefficient statistics per level

    IMPORTANT: Uses only per-sample transforms, so leakage risk stays low.
    """
    cfg = cfg or WaveletFeatureConfig()

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

    parsed_vectors: dict[str, list[np.ndarray]] = {c: [] for c in MODE_VECTOR_JSON_COLS}
    for col in MODE_VECTOR_JSON_COLS:
        for v in df[col].tolist():
            parsed_vectors[col].append(parse_mode_vector_json(v))

    for mode_idx, col in enumerate(MODE_VECTOR_JSON_COLS, start=1):
        vecs = parsed_vectors[col]

        if cfg.include_raw_energy_ratios:
            first_ratio = []
            middle_ratio = []
            last_ratio = []
            for vec in vecs:
                n = len(vec)
                cut1 = max(1, n // 3)
                cut2 = max(cut1 + 1, 2 * n // 3)
                e_total = float(np.sum(vec**2)) + 1e-12
                e_first = float(np.sum(vec[:cut1] ** 2))
                e_middle = float(np.sum(vec[cut1:cut2] ** 2))
                e_last = float(np.sum(vec[cut2:] ** 2))
                first_ratio.append(e_first / e_total)
                middle_ratio.append(e_middle / e_total)
                last_ratio.append(e_last / e_total)

            out[f"mode_{mode_idx}_energy_ratio_first"] = np.asarray(first_ratio, dtype=float)
            out[f"mode_{mode_idx}_energy_ratio_middle"] = np.asarray(middle_ratio, dtype=float)
            out[f"mode_{mode_idx}_energy_ratio_last"] = np.asarray(last_ratio, dtype=float)

        if cfg.include_wavelet_stats:
            wavelet = pywt.Wavelet(cfg.wavelet)
            for vec_idx, vec in enumerate(vecs):
                max_level = pywt.dwt_max_level(len(vec), wavelet.dec_len)
                if cfg.level > max_level:
                    raise ValueError(
                        f"Requested level={cfg.level} too high for len(vec)={len(vec)} and wavelet={cfg.wavelet}; max_level={max_level}"
                    )
                coeffs = pywt.wavedec(vec, wavelet=wavelet, level=cfg.level, mode="symmetric")
                # coeffs[0]=A_level, coeffs[1:]=D_level..D1
                if vec_idx == 0:
                    coeff_store: dict[str, list[dict[str, float]]] = {}
                for coeff_i, coeff in enumerate(coeffs):
                    band_name = f"A{cfg.level}" if coeff_i == 0 else f"D{cfg.level - coeff_i + 1}"
                    coeff_store.setdefault(band_name, []).append(_coeff_stats(coeff))

            for band_name, stats_rows in coeff_store.items():
                stats_df = pd.DataFrame(stats_rows)
                for stat_name in stats_df.columns:
                    out[f"mode_{mode_idx}_{band_name}_{stat_name}"] = stats_df[stat_name].astype(float)

    X = pd.DataFrame(out)
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs produced in wavelet feature matrix; example columns: {nan_cols[:10]}")

    return X

