# Code Rules

## General principles
- Keep code simple, explicit, and easy to debug.
- Prefer small functions with clear responsibilities.
- Prioritize reproducibility over cleverness.
- Do not introduce unnecessary abstractions.
- Every change must preserve dataset integrity and avoid leakage.

## Style rules
- Use Python 3.10+ compatible code.
- Use type hints whenever practical.
- Use descriptive variable names.
- Avoid one-letter variable names except in very small local scopes.
- Write comments only where they add real clarity.
- Do not leave dead code commented out.

## Data handling rules
- Never split by raw mode rows.
- Always treat one full damage configuration as one sample.
- Never leak information across train/val/test.
- Any new feature extraction must operate only on information available at inference time.
- Be careful with using severity as an input feature unless the research design explicitly allows it.

## File responsibility rules
- `src/data/` should only handle loading, reshaping, cleaning, and splitting.
- `src/features/` should only handle feature engineering.
- `src/models/` should only define models and training helpers.
- `src/eval/` should only handle metrics and evaluation logic.
- `scripts/` should orchestrate tasks, not contain core business logic.

## Logging and outputs
- Print concise but useful progress messages.
- Save outputs to `outputs/` or `data/processed/` as appropriate.
- Do not overwrite important files silently unless the script is explicitly designed to do so.

## Research safety rules
- If a result looks too good, check for leakage first.
- If stratification fails, inspect class counts before changing logic.
- If label distribution changes unexpectedly, inspect raw grouped data before training.
- Do not trust metrics until dataset grouping and split logic are verified.

## Baseline-first rule
Before adding a more complex model:
1. verify data
2. verify split
3. create a simple baseline
4. create a reproducible evaluation
Only then move to more advanced models.