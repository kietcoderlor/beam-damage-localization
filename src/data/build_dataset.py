from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.data.load_excel import load_raw_excel


def _safe_float(value: Any):
    if pd.isna(value):
        return None
    return float(value)


def build_scenario_dataset() -> pd.DataFrame:
    beam_data = load_raw_excel()
    df = beam_data.data.copy()
    mode_shape_cols = beam_data.mode_shape_cols

    scenario_col = "Kịch bản hư hỏng"
    mode_col = "Mode dao động"
    freq_col = "Tần số dao động"

    damage_meta_cols = [
        "Vị trí hư hỏng 1 cách gối trái",
        "Vị trí hư hỏng 2 cách gối trái",
        "Vị trí hư hỏng 3 cách gối trái",
        "Vị trí hư hỏng 4 cách gối trái",
        "Độ giảm Mô đun đàn hồi (%)",
    ]

    # Group theo toàn bộ metadata hư hỏng, không group chỉ theo "Kịch bản 1"
    group_cols = [scenario_col] + damage_meta_cols

    scenario_rows = []

    for group_key, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values(mode_col)

        row_out = {}

        # metadata
        row_out["scenario_name"] = group_key[0]
        row_out["damage_pos_1"] = _safe_float(group_key[1])
        row_out["damage_pos_2"] = _safe_float(group_key[2])
        row_out["damage_pos_3"] = _safe_float(group_key[3])
        row_out["damage_pos_4"] = _safe_float(group_key[4])
        row_out["damage_severity"] = _safe_float(group_key[5])

        # đếm số vị trí hư thật sự
        damage_positions = [
            row_out["damage_pos_1"],
            row_out["damage_pos_2"],
            row_out["damage_pos_3"],
            row_out["damage_pos_4"],
        ]
        row_out["num_damages"] = sum(x is not None for x in damage_positions)

        # kiểm tra số mode
        row_out["num_modes_found"] = int(len(g))

        # tạo config id dễ đọc
        pos_str = "_".join(
            ["na" if p is None else f"{p:.2f}" for p in damage_positions]
        )
        sev_str = "na" if row_out["damage_severity"] is None else f"{row_out['damage_severity']:.2f}"
        row_out["config_id"] = f"{row_out['scenario_name']}__pos_{pos_str}__sev_{sev_str}"

        # lưu frequency + vector cho từng mode
        for _, mode_row in g.iterrows():
            mode_id = int(mode_row[mode_col])

            row_out[f"freq_mode_{mode_id}"] = float(mode_row[freq_col])

            mode_vector = [float(mode_row[col]) for col in mode_shape_cols]
            row_out[f"mode_{mode_id}_vector_json"] = json.dumps(mode_vector)

        scenario_rows.append(row_out)

    scenario_df = pd.DataFrame(scenario_rows)

    # sắp xếp cột cho dễ nhìn
    front_cols = [
        "config_id",
        "scenario_name",
        "num_damages",
        "damage_pos_1",
        "damage_pos_2",
        "damage_pos_3",
        "damage_pos_4",
        "damage_severity",
        "num_modes_found",
        "freq_mode_1",
        "freq_mode_2",
        "freq_mode_3",
        "freq_mode_4",
        "mode_1_vector_json",
        "mode_2_vector_json",
        "mode_3_vector_json",
        "mode_4_vector_json",
    ]

    existing_front_cols = [c for c in front_cols if c in scenario_df.columns]
    remaining_cols = [c for c in scenario_df.columns if c not in existing_front_cols]
    scenario_df = scenario_df[existing_front_cols + remaining_cols]

    return scenario_df


if __name__ == "__main__":
    scenario_df = build_scenario_dataset()
    print("Scenario dataset shape:", scenario_df.shape)
    print(scenario_df.head())