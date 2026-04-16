# Beam Damage Localization — Research Pipeline

> **Convention:** S = Small (≤1 day) · M = Medium (1–2 days) · L = Large (2+ days)
> **Status:** ✅ Done · ⚡ WIP · ⏳ Up Next · 🔒 Later

---

## Status Summary

| Phase | Title                        | Backend / Scripts     | Overall   |
| ----- | ---------------------------- | --------------------- | --------- |
| **P0**  | Data Pipeline & Audit      | Done (4/4)            | ✅ Complete |
| **P1**  | Feature Engineering        | Not started (0/6)     | ⏳ Up Next |
| **P2**  | Baseline Models            | Not started (0/5)     | ⏳ Up Next |
| **P3**  | Evaluation & Error Analysis | Not started (0/5)    | ⏳ Up Next |
| **P4**  | Advanced Models & Ablation | Not started (0/5)     | 🔒 Later   |
| **P5**  | Paper-Ready Analysis       | Not started (0/4)     | 🔒 Later   |

---

## Phase 0 — Data Pipeline & Audit ✅

_Foundation. Must be complete before any feature or model work._

| ID    | Task                                              | Scope | Deps | Done |
| ----- | ------------------------------------------------- | ----- | ---- | ---- |
| P0-1  | Excel loader + column normalisation               | M     | —    | ✅   |
| P0-2  | Mode-level → scenario-level reshape (1 row = 1 config with 4 modes) | M | — | ✅ |
| P0-3  | Config-level train / val / test split (no leakage) | S    | P0-2 | ✅   |
| P0-4  | Dataset integrity audit script (`scripts/audit_dataset.py`) | S | P0-3 | ✅ |

**Definition of done:** `audit_dataset.py` passes all checks — no leakage, all JSON vectors parse to length 191, no NaN in freq columns, label distribution visible across splits.

---

## Phase 1 — Feature Engineering ⏳

_Build a reusable, versioned feature matrix from scenario-level data. No model training yet._

| ID    | Task                                              | Scope | Deps       | Done |
| ----- | ------------------------------------------------- | ----- | ---------- | ---- |
| P1-1  | Parse JSON mode vectors → numpy arrays (`src/features/vector_utils.py`) | S | P0-4 | – |
| P1-2  | Frequency feature vector (4 freqs, normalised by healthy baseline or z-score) | S | P1-1 | – |
| P1-3  | Mode shape concat feature (4 × 191 = 764 dims, stored as numpy) | S | P1-1 | – |
| P1-4  | Physics-inspired features: MAC matrix, curvature diff, inter-mode ratio | M | P1-1 | – |
| P1-5  | Reusable feature matrix builder (`src/features/build_features.py`) — outputs X, y arrays + metadata | M | P1-2, P1-3, P1-4 | – |
| P1-6  | Feature inspection report: variance, correlation heatmap, NaN check | S | P1-5 | – |

**Definition of done:** `build_features.py` produces a clean `X_train`, `X_val`, `X_test` with no NaN, consistent shape, and a saved `features_meta.json` describing which features are included.

---

## Phase 2 — Baseline Models ⏳

_Trustworthy baselines before any complex architecture. Baseline first, ablate later._

| ID    | Task                                              | Scope | Deps       | Done |
| ----- | ------------------------------------------------- | ----- | ---------- | ---- |
| P2-1  | Label schema: define `y_count` (damage count classifier) and `y_pos` (damage position targets) | S | P1-5 | – |
| P2-2  | XGBoost baseline — damage count classifier (`src/models/xgb_count.py`) | M | P2-1 | – |
| P2-3  | XGBoost baseline — damage position (multi-output or per-position) (`src/models/xgb_position.py`) | M | P2-1 | – |
| P2-4  | Small MLP baseline — PyTorch, 2–3 hidden layers (`src/models/mlp_baseline.py`) | M | P2-1 | – |
| P2-5  | Training script (`scripts/train_baseline.py`) — reproducible seed, saves model checkpoints to `models/` | S | P2-2, P2-3, P2-4 | – |

**Definition of done:** All three baselines train without error, produce predictions on val set, and saved checkpoints exist under `models/`.

---

## Phase 3 — Evaluation & Error Analysis ⏳

_Rigorous, per-split evaluation. Do not skip val/test breakdown._

| ID    | Task                                              | Scope | Deps       | Done |
| ----- | ------------------------------------------------- | ----- | ---------- | ---- |
| P3-1  | Metrics module (`src/eval/metrics.py`): accuracy, F1 (macro + per-class), MAE on damage position, exact-match rate | M | P2-5 | – |
| P3-2  | Evaluation runner (`scripts/evaluate.py`) — prints + saves CSV to `results/` for train/val/test | S | P3-1 | – |
| P3-3  | Confusion matrix + per-`num_damages` class breakdown | S | P3-1 | – |
| P3-4  | Error case inspector: which `scenario_name` fails, predicted vs true, residual analysis | M | P3-2 | – |
| P3-5  | Leakage sanity re-check: verify no config_id in train appears in val/test predictions | S | P3-2 | – |

**Definition of done:** `results/` contains per-split metric CSVs, confusion matrix plots saved as PNG, and at least one error analysis notebook or script identifying the hardest scenarios.

---

## Phase 4 — Advanced Models & Ablation 🔒

_Only after Phase 3 baselines are stable and evaluated._

| ID    | Task                                              | Scope | Deps       | Done |
| ----- | ------------------------------------------------- | ----- | ---------- | ---- |
| P4-1  | 1D-CNN on mode shape sequences (`src/models/cnn1d.py`) | L | P1-5, P3-3 | – |
| P4-2  | Physics-informed curvature difference features (mode shape second derivative) | M | P1-4 | – |
| P4-3  | Wavelet decomposition features on mode shape vectors | M | P1-4 | – |
| P4-4  | Ablation study: freq-only vs shape-only vs combined feature sets | M | P4-1 | – |
| P4-5  | Ablation result table: feature set × model matrix (script + CSV output) | M | P4-4 | – |

**Definition of done:** Ablation table saved as `results/ablation_table.csv` with all feature set × model combinations and their val/test metrics.

---

## Phase 5 — Paper-Ready Analysis 🔒

_Only after ablation is complete and results are stable._

| ID    | Task                                              | Scope | Deps       | Done |
| ----- | ------------------------------------------------- | ----- | ---------- | ---- |
| P5-1  | Final result table: all models × all metrics, train/val/test, export as LaTeX + CSV | M | P4-5 | – |
| P5-2  | Figures: damage localisation heatmap, mode shape overlays, prediction error plots | M | P3-4 | – |
| P5-3  | Reproducibility checklist: `requirements.txt`, global seed lock, `README.md` run instructions | S | — | – |
| P5-4  | Comparison notes vs reference paper (if evaluation and targets are comparable — flag differences explicitly) | S | P5-1 | – |

**Definition of done:** All figures saved to `figures/`, LaTeX table generated, README describes full pipeline from raw data to final results in one command sequence.

---

## Suggested Execution Order

```
P0 (done) → P1 → P2 → P3 → P4 → P5
```

Within each phase, follow task ID order (P1-1 before P1-2, etc.) unless stated otherwise.

**Immediate next step:** P1-1 — parse JSON mode vectors into numpy arrays.

---

## Key File Map

```
data/
  raw/           ← original Excel
  processed/     ← train.csv, val.csv, test.csv

src/
  data/
    loader.py        ← P0-1, P0-2
    split.py         ← P0-3
    audit.py         ← P0-4
  features/
    vector_utils.py  ← P1-1
    build_features.py ← P1-5
  models/
    xgb_count.py     ← P2-2
    xgb_position.py  ← P2-3
    mlp_baseline.py  ← P2-4
    cnn1d.py         ← P4-1
  eval/
    metrics.py       ← P3-1

scripts/
  audit_dataset.py   ← P0-4
  train_baseline.py  ← P2-5
  evaluate.py        ← P3-2

models/              ← saved checkpoints
results/             ← metric CSVs, confusion matrices
figures/             ← paper figures
```

---

## Critical Rules (carry forward to every phase)

1. **Never split by raw mode row** — split only by `config_id`
2. **Check leakage first** if any result looks too good
3. **Scripts stay thin** — core logic lives in `src/`
4. **Baseline before complexity** — do not jump to CNN until XGBoost is evaluated
5. **Save everything** — predictions, checkpoints, metrics, figures
6. **Flag assumptions explicitly** — do not silently assume comparability with reference paper
