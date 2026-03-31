import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import PROCESSED_DIR  # noqa: E402

df = pd.read_csv(PROCESSED_DIR / "scenario_dataset.csv")

print("=== num_damages distribution ===")
print(df["num_damages"].value_counts(dropna=False).sort_index())

print("\n=== Rows with num_damages = 4 ===")
cols = [
    "config_id",
    "scenario_name",
    "damage_pos_1",
    "damage_pos_2",
    "damage_pos_3",
    "damage_pos_4",
    "damage_severity",
    "num_damages",
]
print(df.loc[df["num_damages"] == 4, cols].to_string(index=False))

print("\n=== Rows with num_damages = 0 ===")
print(df.loc[df["num_damages"] == 0, cols].to_string(index=False))