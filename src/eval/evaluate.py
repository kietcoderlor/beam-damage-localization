from __future__ import annotations

import pandas as pd

from src.eval.metrics import DamageMetrics


def print_damage_metrics(title: str, m: DamageMetrics) -> None:
    print(f"\n=== {title} ===")
    print(f"num_damages accuracy: {m.num_damages_accuracy:.4f}")
    print(f"num_damages f1_macro: {m.num_damages_f1_macro:.4f}")
    print(f"damage_pos MAE (overall, masked):  {m.pos_mae_overall:.4f}")
    print(f"damage_pos RMSE (overall, masked): {m.pos_rmse_overall:.4f}")

    per_slot = pd.DataFrame(
        {
            "mae": m.pos_mae_per_slot,
            "rmse": m.pos_rmse_per_slot,
        }
    )
    print("\nPer-slot position errors (masked):")
    print(per_slot.to_string(float_format=lambda x: f"{x:.4f}"))

