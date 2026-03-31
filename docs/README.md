# Beam Damage Localization Research Project

## Project overview
This project studies multi-damage localization in beam structures using modal features, wavelet-based signal analysis, and AI/ML models.

The current goal is to build a clean, reproducible research pipeline that starts from raw Excel simulation data and ends with train/validation/test datasets and baseline models.

## Current research direction
We are focusing on:

- beam damage localization
- multi-damage cases
- modal frequencies and mode shapes
- wavelet-inspired and physics-informed features
- AI baselines such as XGBoost, MLP, and later 1D-CNN

## Current dataset status
The raw Excel file has already been parsed successfully.

Important dataset conventions:

- 1 raw row = 1 mode
- 1 final sample = 1 full damage configuration with all 4 modes
- split must be done at configuration level, never at mode-row level
- current processed files:
  - `data/processed/scenario_dataset.csv`
  - `data/processed/train.csv`
  - `data/processed/val.csv`
  - `data/processed/test.csv`

## Key columns in the scenario dataset
Main metadata:

- `config_id`
- `scenario_name`
- `num_damages`
- `damage_pos_1`
- `damage_pos_2`
- `damage_pos_3`
- `damage_pos_4`
- `damage_severity`
- `num_modes_found`

Modal features:

- `freq_mode_1`
- `freq_mode_2`
- `freq_mode_3`
- `freq_mode_4`
- `mode_1_vector_json`
- `mode_2_vector_json`
- `mode_3_vector_json`
- `mode_4_vector_json`

## Important data note
The dataset includes:
- 0-damage case
- 1-damage case
- 2-damage case
- 4-damage case

So the data is not limited to only up to 3 damage locations.

## Project structure
- `src/data/` for loading, reshaping, splitting
- `src/features/` for physics and wavelet features
- `src/models/` for baselines and future models
- `src/eval/` for metrics and evaluation
- `scripts/` for execution scripts
- `docs/` for project documentation and Cursor context

## Immediate next steps
1. verify dataset integrity
2. create baseline feature extraction
3. train first baseline model
4. evaluate train/val/test split behavior
5. iterate toward publishable research design