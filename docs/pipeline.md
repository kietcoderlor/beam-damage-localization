# Beam Damage Localization — Research Pipeline

> **Convention:** S = Small (≤1 day) · M = Medium (1–2 days) · L = Large (2+ days)
> **Status:** ✅ Done · ⚡ WIP · ⏳ Up Next · 🔒 Later

---

## Status Summary

| Phase | Title                         | Overall      |
| ----- | ----------------------------- | ------------ |
| **P0**  | Data Pipeline & Audit        | ✅ Complete  |
| **P1**  | Feature Engineering          | ✅ Complete  |
| **P2**  | Baseline Models              | ✅ Complete  |
| **P3**  | Evaluation & Error Analysis  | ✅ Complete  |
| **P4**  | Advanced Models & Tuning     | ✅ Complete  |
| **P4b** | Data Augmentation            | ✅ Complete  |
| **P5**  | Paper-Ready Analysis         | ⏳ Up Next   |

---

## Phase 0 — Data Pipeline & Audit ✅

| Task | Description | Done |
| ---- | ----------- | ---- |
| P0-1 | Excel loader + column normalisation | ✅ |
| P0-2 | Mode-level → scenario-level reshape (1 row = 1 config with 4 modes) | ✅ |
| P0-3 | Config-level train / val / test split (no leakage) | ✅ |
| P0-4 | Dataset integrity audit script | ✅ |

**Outputs:** `data/processed/scenario_dataset.csv`, `train.csv`, `val.csv`, `test.csv`

---

## Phase 1 — Feature Engineering ✅

| Task | Description | Done |
| ---- | ----------- | ---- |
| P1-1 | Baseline frequency feature vector (4 freqs) | ✅ |
| P1-2 | Mode shape concat features (4 × 191 dims) | ✅ |
| P1-3 | Wavelet-inspired features (`src/features/wavelet_features.py`) | ✅ |
| P1-4 | Physics-inspired features: gradient, curvature, zero-crossing, peak count (`src/features/physics_features.py`) | ✅ |
| P1-5 | Reusable feature matrix builder | ✅ |

---

## Phase 2 — Baseline Models ✅

| Task | Description | Done |
| ---- | ----------- | ---- |
| P2-1 | RandomForest baseline (`baseline_rf`) | ✅ |
| P2-2 | XGBoost baseline (`baseline_xgb`) | ✅ |
| P2-3 | MLP baseline (`baseline_mlp_smoke2`) | ✅ |
| P2-4 | CNN 1D baseline (`baseline_cnn1d_smoke`) | ✅ |
| P2-5 | XGBoost + advanced features (`baseline_xgb_advanced`) | ✅ |

**Best baseline:** `baseline_xgb_advanced` — `acc=0.9615`, `f1_macro=0.9653`, `pos_mae=0.4208`

---

## Phase 3 — Evaluation & Error Analysis ✅

| Task | Description | Done |
| ---- | ----------- | ---- |
| P3-1 | Metrics module: accuracy, F1 macro, pos MAE/RMSE (masked) | ✅ |
| P3-2 | Evaluation runner (`scripts/analysis/04_error_analysis.py`, `05_evaluate.py`) | ✅ |
| P3-3 | Confusion matrix + per-class breakdown | ✅ |
| P3-4 | Error case inspector: top-15 worst position errors | ✅ |
| P3-5 | Leakage sanity check | ✅ |
| P3-6 | Per-class P/R/F1 in metrics + `--test-csv` flag in all eval scripts *(28/05/2026)* | ✅ |

**Outputs per run:** `test_classification_report.txt`, `test_confusion_matrix.csv`, `test_position_error_by_class.csv`, `test_top15_position_errors.csv`, `metrics_summary.json`

---

## Phase 4 — Advanced Models & Tuning ✅

| Task | Description | Done |
| ---- | ----------- | ---- |
| P4-1 | Tuning RandomForest (`scripts/tune/06_tune_rf.py`) → `tuned_rf_balanced_refit` | ✅ |
| P4-2 | Tuning XGBoost baseline (`scripts/tune/07_tune_xgb.py`) → `tuned_xgb_balanced` | ✅ |
| P4-3 | Tuning XGBoost advanced (`scripts/tune/10_tune_xgb_advanced.py`) → `tuned_xgb_advanced_balanced` | ✅ |
| P4-4 | Class-conditional position regressor + postprocess/snap → `xgb_advanced_moe_postprocess` | ✅ |
| P4-5 | Tuning CNN 1D → `tuned_cnn1d_balanced` | ✅ |
| P4-6 | Tuning MLP → `tuned_mlp_balanced` | ✅ |

**Best overall:** `xgb_advanced_moe_postprocess` — `acc=0.9615`, `f1_macro=0.9653`, `pos_mae=0.1217`

See `docs/MODEL_COMPARISON.md` for full ranking.

---

## Phase 4b — Data Augmentation ✅ *(hoàn thành 28/05/2026)*

| Task | Description | Done |
| ---- | ----------- | ---- |
| P4b-1 | Gaussian noise augment class 0 — `scripts/data/03_augment_class0.py` (60 samples, 60/60 pass) | ✅ |
| P4b-2 | CVAE conditional generative model — `src/models/cvae.py`, `scripts/data/04_train_cvae.py` | ✅ |
| P4b-3 | Generalized augment function `augment_by_class_gaussian()` — `src/data/augment.py` | ✅ |
| P4b-4 | Augment class 4 + append vào splits — `scripts/data/05_augment_class4.py` (50 samples) | ✅ |
| P4b-5 | Thêm `--test-holdout` vào `03_augment_class0.py`, tạo `test_augmented.csv` | ✅ |
| P4b-6 | Retrain trên full augmented data → `xgb_aug_full` | ✅ |

**Augmented splits (28/05/2026):**

| File | Rows | Class 0 | Class 1 | Class 2 | Class 4 |
|------|------|---------|---------|---------|---------|
| `train_augmented.csv` | 289 | 48 (syn) | 58 | 136 | 47 (7 real + 40 syn) |
| `val_augmented.csv` | 38 | 8 (syn) | 7 | 17 | 6 (1 real + 5 syn) |
| `test_augmented.csv` | 36 | 5 (syn) | 7 | 18 | 6 (1 real + 5 syn) |

**Kết quả `xgb_aug_full` trên `test_augmented.csv`:** `acc=0.9444`, `f1_macro=0.9452`, `pos_mae=0.0997`

Per-class: Class 0 R=1.00 · Class 1 R=0.71 · Class 2 R=1.00 · Class 4 R=1.00

> Class 0 và class 4 synthetic trong test là sanity check, không phải blind test trên real measurements.

---

## Phase 5 — Paper-Ready Analysis ⏳

| Task | Description | Done |
| ---- | ----------- | ---- |
| P5-1 | Ablation study: bỏ class-conditional regressor / bỏ postprocess, so sánh MAE/F1 | – |
| P5-2 | Multi-seed stability check (chạy nhiều random seed để kiểm tra độ ổn định) | – |
| P5-3 | Final result table (LaTeX + CSV): all models × all metrics, train/val/test | – |
| P5-4 | Figures: prediction error plots, confusion matrices, mode shape overlays | – |
| P5-5 | Comparison notes vs reference paper (flag differences explicitly) | – |
| P5-6 | Reproducibility checklist: `requirements.txt`, seed lock, README run instructions | – |

**Immediate next step:** P5-1 — ablation study cho `xgb_advanced_moe_postprocess`; P5-x — cải thiện recall class 1 (hiện 71%).

---

## Critical Rules (carry forward to every phase)

1. **Never split by raw mode row** — split only by `config_id`
2. **Check leakage first** if any result looks too good
3. **Scripts stay thin** — core logic lives in `src/`
4. **Baseline before complexity** — do not jump to CNN until XGBoost is evaluated
5. **Save everything** — predictions, checkpoints, metrics, figures
6. **Flag assumptions explicitly** — do not silently assume comparability with reference paper
