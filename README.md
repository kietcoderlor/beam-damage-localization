# Beam Damage Localization

Research codebase for multi-damage localization in beam structures using modal frequencies, mode shapes, and ML baselines.

## What this repo does

- Builds a scenario-level dataset from raw Excel simulation data
- Splits data into train/val/test at configuration level (to reduce leakage risk)
- Extracts baseline features from modal frequencies and mode vectors
- Trains a first baseline model (RandomForest)
- Produces evaluation metrics and error analysis outputs

## Current status

- End-to-end baseline pipeline is runnable
- Baseline artifact is saved in `outputs/baseline_rf/artifact.joblib`
- Evaluation and analysis reports are saved in `outputs/baseline_rf/`

## Quick start

### 1) Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2) Run pipeline

```bash
python scripts/data/01_build_dataset.py
python scripts/data/02_split_dataset.py
python scripts/train/03_train_baseline.py
python scripts/analysis/04_error_analysis.py
```

## Data convention (important)

- `1 raw row = 1 mode`
- `1 final sample = 1 full damage configuration (all 4 modes)`
- Splitting must happen at configuration level, never raw mode-row level
- Avoid leakage between train/val/test at all times

## Project structure

- `src/data/` - loading, reshaping, cleaning, splitting
- `src/features/` - feature extraction
- `src/models/` - baseline and model wrappers
- `src/eval/` - metrics and reporting helpers
- `scripts/` - runnable orchestration scripts
- `data/` - raw and processed data
- `outputs/` - training artifacts and analysis files
- `docs/` - project notes and rules

## Key outputs

- `data/processed/scenario_dataset.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `outputs/baseline_rf/artifact.joblib`
- `outputs/baseline_rf/test_classification_report.txt`
- `outputs/baseline_rf/test_confusion_matrix.csv`
- `outputs/baseline_rf/test_position_error_by_class.csv`
- `outputs/baseline_rf/test_top15_position_errors.csv`

## Notes

- The current dataset includes classes: `0`, `1`, `2`, and `4` damages.
- Class `0` is very rare, so split logic handles rare-class safety explicitly.
- If metrics look unusually high, check leakage first.
  .
