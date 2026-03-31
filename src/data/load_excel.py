from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd

from src.config.settings import RAW_EXCEL_PATH


@dataclass
class BeamExcelData:
    data: pd.DataFrame
    node_ids: List[int]
    node_coords: List[float]
    mode_shape_cols: List[str]


def load_raw_excel(path=RAW_EXCEL_PATH, sheet_name: str = 0) -> BeamExcelData:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # Dòng 1 và dòng 2 của Excel
    header_row_1 = raw.iloc[0]   # tọa độ điểm nút
    header_row_2 = raw.iloc[1]   # tên cột thật + số thứ tự node

    # 8 cột đầu là metadata
    meta_cols = []
    for i, x in enumerate(header_row_2.iloc[:8]):
        if pd.notna(x):
            meta_cols.append(str(x).strip())
        else:
            meta_cols.append(f"meta_{i}")

    # Từ cột thứ 9 trở đi là mode shape theo node
    node_ids_raw = header_row_2.iloc[8:]
    node_coords_raw = header_row_1.iloc[8:]

    valid_node_mask = node_ids_raw.notna() & node_coords_raw.notna()

    node_ids = [int(x) for x in node_ids_raw[valid_node_mask].tolist()]
    node_coords = [float(x) for x in node_coords_raw[valid_node_mask].tolist()]
    mode_shape_cols = [f"node_{nid}" for nid in node_ids]

    all_columns = meta_cols + mode_shape_cols

    # Dữ liệu thật bắt đầu từ dòng thứ 3
    data = raw.iloc[2:, : 8 + len(mode_shape_cols)].copy().reset_index(drop=True)
    data.columns = all_columns

    # Chuẩn hóa các cột metadata sang numeric nếu có
    numeric_meta_cols = [
        "Mode dao động",
        "Tần số dao động",
        "Vị trí hư hỏng 1 cách gối trái",
        "Vị trí hư hỏng 2 cách gối trái",
        "Vị trí hư hỏng 3 cách gối trái",
        "Vị trí hư hỏng 4 cách gối trái",
        "Độ giảm Mô đun đàn hồi (%)",
    ]

    for col in numeric_meta_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Chuẩn hóa toàn bộ cột mode shape sang numeric
    for col in mode_shape_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return BeamExcelData(
        data=data,
        node_ids=node_ids,
        node_coords=node_coords,
        mode_shape_cols=mode_shape_cols,
    )


if __name__ == "__main__":
    beam_data = load_raw_excel()
    df = beam_data.data

    print("Shape:", df.shape)
    print("Columns (first 12):", df.columns[:12].tolist())
    print("Node count:", len(beam_data.node_ids))
    print("First 5 node ids:", beam_data.node_ids[:5])
    print("First 5 node coords:", beam_data.node_coords[:5])
    print(df.head())