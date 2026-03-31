from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"

RAW_EXCEL_NAME = "Data tổng 27 trường hợp.xlsx"
RAW_EXCEL_PATH = RAW_DIR / RAW_EXCEL_NAME

RANDOM_SEED = 42
N_MODES = 4