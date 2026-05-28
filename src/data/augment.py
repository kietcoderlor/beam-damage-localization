from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.features.baseline_features import parse_mode_vector_json


_MODE_JSON_COLS = [
    "mode_1_vector_json",
    "mode_2_vector_json",
    "mode_3_vector_json",
    "mode_4_vector_json",
]
_FREQ_COLS = ["freq_mode_1", "freq_mode_2", "freq_mode_3", "freq_mode_4"]


def augment_class0_gaussian(
    class0_df: pd.DataFrame,
    n_samples: int = 60,
    freq_noise_std_frac: float = 0.005,
    mode_noise_std_frac: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic class-0 (no damage) samples by adding Gaussian noise
    to the real class-0 sample's modal frequencies and mode shape vectors.

    Physical motivation: measurement noise, sensor variation, environmental
    fluctuation are well-modelled as additive Gaussian noise in structural
    monitoring.

    Args:
        class0_df: DataFrame rows where num_damages == 0 (typically 1 row).
        n_samples: Number of synthetic samples to produce.
        freq_noise_std_frac: Noise σ as a fraction of each frequency value.
            e.g. 0.005 → σ = 0.5 % of the frequency.
        mode_noise_std_frac: Noise σ as a fraction of each mode vector's std.
            e.g. 0.01 → σ = 1 % of the vector's standard deviation.
        random_state: RNG seed for reproducibility.

    Returns:
        DataFrame with the same schema as scenario_dataset.csv containing
        n_samples synthetic class-0 rows.
    """
    if len(class0_df) == 0:
        raise ValueError("class0_df is empty — provide at least one real class-0 row.")

    rng = np.random.default_rng(random_state)
    template = class0_df.iloc[0]

    real_freqs = np.array([float(template[c]) for c in _FREQ_COLS])
    real_vecs = [parse_mode_vector_json(template[col]) for col in _MODE_JSON_COLS]

    synthetic_rows = []

    for i in range(n_samples):
        row: dict = {}

        row["config_id"] = f"synthetic_normal_{i + 1:03d}__pos_na_na_na_na__sev_na"
        row["scenario_name"] = f"synthetic_normal_{i + 1:03d}"
        row["num_damages"] = 0
        row["damage_pos_1"] = float("nan")
        row["damage_pos_2"] = float("nan")
        row["damage_pos_3"] = float("nan")
        row["damage_pos_4"] = float("nan")
        row["damage_severity"] = float("nan")
        row["num_modes_found"] = 4

        # --- noisy frequencies, sorted to preserve f1 < f2 < f3 < f4 ---
        noise = rng.normal(0.0, freq_noise_std_frac * real_freqs)
        noisy_freqs = np.sort(np.maximum(real_freqs + noise, real_freqs * 0.5))
        for j, col in enumerate(_FREQ_COLS):
            row[col] = float(noisy_freqs[j])

        # --- noisy mode shape vectors ---
        for vec, col in zip(real_vecs, _MODE_JSON_COLS):
            sigma = mode_noise_std_frac * float(np.std(vec))
            if sigma < 1e-12:
                sigma = mode_noise_std_frac * 1e-3  # flat-vector fallback
            noisy_vec = vec + rng.normal(0.0, sigma, size=vec.shape)
            row[col] = json.dumps(noisy_vec.tolist())

        synthetic_rows.append(row)

    return pd.DataFrame(synthetic_rows)


def augment_by_class_gaussian(
    source_df: pd.DataFrame,
    target_class: int,
    n_samples: int = 50,
    freq_noise_std_frac: float = 0.005,
    mode_noise_std_frac: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic samples for any damage class by adding Gaussian noise.

    For class 0: damage_pos_1..4 are NaN (same as real class-0 rows).
    For class 4 (or any other): damage_pos values are copied EXACTLY from the
    template row — positions are ground-truth labels and must not be perturbed.

    Templates are selected round-robin across all rows in source_df so that
    the synthetic batch spans the full diversity of real examples.

    Args:
        source_df: Real rows of the class to augment (all must share target_class).
        target_class: num_damages value (0, 1, 2, or 4).
        n_samples: Number of synthetic samples to produce.
        freq_noise_std_frac: Noise σ as fraction of each frequency value.
        mode_noise_std_frac: Noise σ as fraction of each mode vector's std.
        random_state: RNG seed.

    Returns:
        DataFrame with same schema as scenario_dataset.csv.
    """
    if len(source_df) == 0:
        raise ValueError("source_df is empty — provide at least one real row.")

    rng = np.random.default_rng(random_state)
    class_tag = f"class{target_class}"

    pos_cols = ["damage_pos_1", "damage_pos_2", "damage_pos_3", "damage_pos_4"]

    synthetic_rows = []
    for i in range(n_samples):
        template = source_df.iloc[i % len(source_df)]

        real_freqs = np.array([float(template[c]) for c in _FREQ_COLS])
        real_vecs = [parse_mode_vector_json(template[col]) for col in _MODE_JSON_COLS]

        row: dict = {}
        row["config_id"] = f"synthetic_{class_tag}_{i + 1:03d}"
        row["scenario_name"] = f"synthetic_{class_tag}_{i + 1:03d}"
        row["num_damages"] = target_class
        for col in pos_cols:
            row[col] = template[col]
        row["damage_severity"] = template.get("damage_severity", float("nan"))
        row["num_modes_found"] = 4

        noise = rng.normal(0.0, freq_noise_std_frac * real_freqs)
        noisy_freqs = np.sort(np.maximum(real_freqs + noise, real_freqs * 0.5))
        for j, col in enumerate(_FREQ_COLS):
            row[col] = float(noisy_freqs[j])

        for vec, col in zip(real_vecs, _MODE_JSON_COLS):
            sigma = mode_noise_std_frac * float(np.std(vec))
            if sigma < 1e-12:
                sigma = mode_noise_std_frac * 1e-3
            noisy_vec = vec + rng.normal(0.0, sigma, size=vec.shape)
            row[col] = json.dumps(noisy_vec.tolist())

        synthetic_rows.append(row)

    return pd.DataFrame(synthetic_rows)


def validate_synthetic_batch(
    synthetic_df: pd.DataFrame,
    real_sample: pd.Series,
) -> pd.DataFrame:
    """
    Run physical plausibility checks on every synthetic sample.

    Checks:
    - freq_ordered:    f1 < f2 < f3 < f4 (strictly increasing)
    - freq_ratio_ok:   frequency ratios fi/f1 within ±15 % of the real sample
    - mode_std_ok:     per-mode std(synthetic) / std(real) ∈ [0.8, 1.2]

    Returns a boolean DataFrame aligned with synthetic_df's index,
    plus an 'all_ok' column.
    """
    real_freqs = np.array([float(real_sample[c]) for c in _FREQ_COLS])
    real_vecs = [parse_mode_vector_json(real_sample[col]) for col in _MODE_JSON_COLS]
    real_ratios = real_freqs / real_freqs[0]

    records = []
    for _, row in synthetic_df.iterrows():
        freqs = np.array([float(row[c]) for c in _FREQ_COLS])

        freq_ordered = bool(np.all(np.diff(freqs) > 0))

        syn_ratios = freqs / freqs[0]
        freq_ratio_ok = bool(np.all(np.abs(syn_ratios - real_ratios) / real_ratios < 0.15))

        mode_std_ok = True
        for vec_real, col in zip(real_vecs, _MODE_JSON_COLS):
            vec_syn = parse_mode_vector_json(row[col])
            real_std = float(np.std(vec_real))
            if real_std > 1e-10:
                ratio = float(np.std(vec_syn)) / real_std
                if not (0.8 <= ratio <= 1.2):
                    mode_std_ok = False
                    break

        records.append(
            {
                "freq_ordered": freq_ordered,
                "freq_ratio_ok": freq_ratio_ok,
                "mode_std_ok": mode_std_ok,
                "all_ok": freq_ordered and freq_ratio_ok and mode_std_ok,
            }
        )

    return pd.DataFrame(records, index=synthetic_df.index)
