# Cursor Project Prompt

You are helping with a research codebase for beam damage localization.

## Project context
This project studies multi-damage localization in beam structures using modal data, wavelet-inspired methods, and machine learning.

The raw dataset originally had:
- one row per mode
- modal frequencies
- mode shape values along beam nodes
- damage location columns
- damage severity

The raw Excel file has already been converted into a scenario-level dataset where:
- one row = one full damage configuration
- each row includes all 4 modal frequencies
- each row includes 4 mode vectors stored as JSON strings
- split must always happen at configuration level

## Important data facts
Current processed dataset columns include:
- `config_id`
- `scenario_name`
- `num_damages`
- `damage_pos_1`
- `damage_pos_2`
- `damage_pos_3`
- `damage_pos_4`
- `damage_severity`
- `num_modes_found`
- `freq_mode_1..4`
- `mode_1_vector_json..mode_4_vector_json`

Current class distribution includes:
- 0 damage
- 1 damage
- 2 damage
- 4 damage

So do not assume only up to 3 damage locations.

## Critical rules and coding standards

See `docs/CODE_RULES.md` for the full list. Key points:
- Never split by raw mode row — always by `config_id`
- If metrics look suspiciously high, check leakage first
- Baseline-first: verify data → verify split → baseline → evaluate → advanced

## Current pipeline status

Pipeline is complete through Phase 4. Best model: `xgb_advanced_moe_postprocess`.
See `docs/pipeline.md` for phase status and `docs/MODEL_COMPARISON.md` for results.

## Response style
When proposing code changes:
- explain exactly which file to edit
- provide complete code blocks
- keep steps sequential
- avoid vague advice
- mention risks if any assumption is being made