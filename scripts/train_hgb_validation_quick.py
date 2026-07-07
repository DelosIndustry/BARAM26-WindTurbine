from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from metric import CAPACITY_KWH, TARGET_COLS, competition_metric


TIME_FEATURES = {
    "year",
    "month",
    "day",
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
    parser = argparse.ArgumentParser(description="Quick 2024 validation with sklearn HistGradientBoosting.")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--valid-start", type=str, default="2024-01-01 01:00:00")
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--use-sample-weight", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_pickle(processed_dir / "features_train.pkl")
    labels = pd.read_pickle(processed_dir / "labels_processed.pkl")
    features.index = pd.to_datetime(features.index)
    labels.index = pd.to_datetime(labels.index)
    common = features.index.intersection(labels.index)
    features = features.loc[common].sort_index()
    labels = labels.loc[common].sort_index()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, labels


def select_features(features: pd.DataFrame, target: str) -> list[str]:
    group_id = target.rsplit("_", 1)[-1]
    keywords = (
        "speed",
        "u_unit",
        "v_unit",
        "gust",
        "sp",
        "prmsl",
        "_t_",
        "_2_t",
        "_2_2t",
        "_r_",
        "_q_",
        "air_density",
    )
    cols = []
    for col in features.columns:
        if col in TIME_FEATURES or col.endswith("_lead_hours") or col.endswith("_available_hour"):
            cols.append(col)
        elif f"_grp{group_id}_near_" in col and any(key in col for key in keywords):
            cols.append(col)
    return cols


def sample_weight(y: pd.Series, capacity: float) -> np.ndarray:
    y_arr = y.to_numpy(dtype=float)
    norm = np.clip(y_arr / capacity, 0.0, 1.0)
    return np.where(y_arr >= capacity * 0.10, 1.0 + norm, 0.30).astype("float32")


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
        y_train = labels.loc[target_train_idx, y_col]
        x_valid = features.loc[target_valid_idx, cols]
        y_valid = labels.loc[target_valid_idx, y_col]

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=args.max_iter,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=0.05,
            early_stopping=True,
            validation_fraction=None,
            random_state=args.random_state,
        )
        fit_kwargs = {}
        if args.use_sample_weight:
            fit_kwargs["sample_weight"] = sample_weight(y_train, capacity)

        print(f"Training {target}: train={len(x_train)}, valid={len(x_valid)}, features={len(cols)}", flush=True)
        model.fit(x_train, y_train, **fit_kwargs)

        pred = model.predict(features.loc[valid_idx, cols])
        predictions[target] = np.clip(pred, 0.0, capacity)

        valid_mae = float(np.mean(np.abs(model.predict(x_valid) - y_valid)))
        model_rows.append(
            {
                "target": target,
                "train_rows": int(len(x_train)),
                "valid_rows": int(len(x_valid)),
                "features": int(len(cols)),
                "max_iter": int(args.max_iter),
                "valid_mae": valid_mae,
            }
        )

    answer = fill_answer(labels, valid_idx)
    metrics = competition_metric(answer, predictions[TARGET_COLS])
    metrics["models"] = model_rows
    metrics["valid_start"] = str(valid_start)
    metrics["use_sample_weight"] = bool(args.use_sample_weight)
    metrics["model"] = "HistGradientBoostingRegressor"

    suffix = "weighted" if args.use_sample_weight else "plain"
    predictions.reset_index().rename(columns={"index": "forecast_kst_dtm"}).to_csv(
        args.out_dir / f"validation_hgb_quick_{suffix}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(model_rows).to_csv(
        args.out_dir / f"validation_hgb_quick_{suffix}_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(args.out_dir / f"validation_hgb_quick_{suffix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
