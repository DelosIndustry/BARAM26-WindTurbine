from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metric import CAPACITY_KWH, TARGET_COLS, competition_metric


TIME_FEATURES = {
    "month",
    "hour",
    "dayofweek",
    "dayofyear",
    "is_weekend",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "dayofyear_sin",
    "dayofyear_cos",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Very quick Ridge validation sanity check.")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--valid-start", type=str, default="2024-01-01 01:00:00")
    parser.add_argument("--alpha", type=float, default=100.0)
    return parser.parse_args()


def load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_pickle(processed_dir / "features_train.pkl")
    labels = pd.read_pickle(processed_dir / "labels_processed.pkl")
    features.index = pd.to_datetime(features.index)
    labels.index = pd.to_datetime(labels.index)
    common = features.index.intersection(labels.index)
    features = features.loc[common].sort_index().replace([np.inf, -np.inf], np.nan)
    labels = labels.loc[common].sort_index()
    return features, labels


def select_features(features: pd.DataFrame, target: str) -> list[str]:
    group_id = target.rsplit("_", 1)[-1]
    cols = []
    for col in features.columns:
        if col in TIME_FEATURES or col.endswith("_lead_hours") or col.endswith("_available_hour"):
            cols.append(col)
        elif f"_grp{group_id}_near_" in col and (
            "speed" in col
            or "u_unit" in col
            or "v_unit" in col
            or "gust" in col
            or "air_density" in col
            or "surface_0_sp" in col
            or "meanSea_0_prmsl" in col
        ):
            cols.append(col)
    return cols


def fill_answer(labels: pd.DataFrame, valid_idx: pd.Series | np.ndarray) -> pd.DataFrame:
    answer = labels.loc[valid_idx, TARGET_COLS].copy()
    for target in TARGET_COLS:
        missing = answer[target].isna()
        if missing.any():
            answer.loc[missing, target] = labels.loc[answer.index[missing], f"{target}_filled"]
    return answer


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features, labels = load_processed(args.processed_dir)
    valid_start = pd.Timestamp(args.valid_start)
    train_idx = features.index < valid_start
    valid_idx = features.index >= valid_start

    predictions = pd.DataFrame(index=features.loc[valid_idx].index)
    model_rows = []

    for target in TARGET_COLS:
        y_col = f"{target}_filled"
        capacity = CAPACITY_KWH[target]
        cols = select_features(features, target)

        target_train_idx = train_idx & labels[y_col].notna()
        target_valid_idx = valid_idx & labels[y_col].notna()

        x_train = features.loc[target_train_idx, cols]
        y_train = labels.loc[target_train_idx, y_col] / capacity
        x_valid = features.loc[target_valid_idx, cols]
        y_valid = labels.loc[target_valid_idx, y_col]

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=args.alpha),
        )

        print(f"Training {target}: train={len(x_train)}, valid={len(x_valid)}, features={len(cols)}", flush=True)
        model.fit(x_train, y_train)

        pred_norm = model.predict(features.loc[valid_idx, cols])
        pred = np.clip(pred_norm, 0.0, 1.0) * capacity
        predictions[target] = pred

        valid_pred = np.clip(model.predict(x_valid), 0.0, 1.0) * capacity
        model_rows.append(
            {
                "target": target,
                "train_rows": int(len(x_train)),
                "valid_rows": int(len(x_valid)),
                "features": int(len(cols)),
                "alpha": float(args.alpha),
                "valid_mae": float(np.mean(np.abs(valid_pred - y_valid))),
            }
        )

    answer = fill_answer(labels, valid_idx)
    metrics = competition_metric(answer, predictions[TARGET_COLS])
    metrics["models"] = model_rows
    metrics["valid_start"] = str(valid_start)
    metrics["model"] = "Ridge"

    predictions.reset_index().rename(columns={"index": "forecast_kst_dtm"}).to_csv(
        args.out_dir / "validation_ridge_quick_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(model_rows).to_csv(
        args.out_dir / "validation_ridge_quick_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(args.out_dir / "validation_ridge_quick_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
