import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import PROCESSED_DIR  # noqa: E402
from src.data.build_dataset import build_scenario_dataset  # noqa: E402


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    scenario_df = build_scenario_dataset()

    out_csv_path = PROCESSED_DIR / "scenario_dataset.csv"
    out_xlsx_path = PROCESSED_DIR / "scenario_dataset.xlsx"

    scenario_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    scenario_df.to_excel(out_xlsx_path, index=False)

    print(f"Saved CSV to: {out_csv_path}")
    print(f"Saved XLSX to: {out_xlsx_path}")
    print("Shape:", scenario_df.shape)
    print(scenario_df.head())


if __name__ == "__main__":
    main()