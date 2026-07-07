from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}


def competition_metric(
    answer_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_cols: list[str] | None = None,
    capacity_kwh: dict[str, float] | None = None,
) -> dict:
    target_cols = target_cols or TARGET_COLS
    capacity_kwh = capacity_kwh or CAPACITY_KWH

    group_rows = []
    group_nmae = []
    group_ficr = []

    for col in target_cols:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        capacity = float(capacity_kwh[col])

        valid = np.isfinite(actual) & np.isfinite(forecast) & (actual >= capacity * 0.10)
        if valid.sum() == 0:
            raise ValueError(f"No valid evaluation rows for {col}.")

        actual_valid = actual[valid]
        forecast_valid = forecast[valid]
        error_rate = np.abs(forecast_valid - actual_valid) / capacity

        nmae = float(np.mean(error_rate))
        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )
        earned_settlement = float(np.sum(actual_valid * unit_price))
        max_settlement = float(np.sum(actual_valid * 4.0))
        ficr = earned_settlement / max_settlement

        group_nmae.append(nmae)
        group_ficr.append(ficr)
        group_rows.append(
            {
                "target": col,
                "valid_rows": int(valid.sum()),
                "nmae": nmae,
                "one_minus_nmae": 1.0 - nmae,
                "ficr": ficr,
                "within_6pct": float(np.mean(error_rate <= 0.06)),
                "within_8pct": float(np.mean(error_rate <= 0.08)),
            }
        )

    one_minus_nmae = 1.0 - float(np.mean(group_nmae))
    ficr = float(np.mean(group_ficr))
    score = 0.5 * one_minus_nmae + 0.5 * ficr

    return {
        "score": score,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "groups": group_rows,
    }
