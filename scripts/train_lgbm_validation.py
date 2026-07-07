from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from metric import CAPACITY_KWH, TARGET_COLS, competition_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM models with 2024 local validation.")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--valid-start", type=str, default="2024-01-01 01:00:00")
    parser.add_argument("--use-sample-weight", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_pickle(processed_dir / "features_train.pkl")
    labels = pd.read_pickle(processed_dir / "labels_processed.pkl")
    features.index = pd.to_datetime(features.index)
    labels.index = pd.to_datetime(labels.index)
    common_index = features.index.intersection(labels.index)
    features = features.loc[common_index].sort_index()
    labels = labels.loc[common_index].sort_index()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, labels


def make_model(random_state: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.20,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def sample_weight(y: pd.Series, capacity: float) -> np.ndarray:
    y_arr = y.to_numpy(dtype=float)
    norm = np.clip(y_arr / capacity, 0.0, 1.0)
    weights = np.where(y_arr >= capacity * 0.10, 1.0 + norm, 0.30)
    return weights.astype("float32")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features, labels = load_processed(args.processed_dir)
    valid_start = pd.Timestamp(args.valid_start)

    train_idx = features.index < valid_start
    valid_idx = features.index >= valid_start

    predictions = pd.DataFrame(index=features.loc[valid_idx].index)
    model_rows = []
    importance_frames = []

    for target in TARGET_COLS:
        y_col = f"{target}_filled"
        capacity = CAPACITY_KWH[target]

        target_train_idx = train_idx & labels[y_col].notna()
        target_valid_idx = valid_idx & labels[y_col].notna()

        x_train = features.loc[target_train_idx]
        y_train = labels.loc[target_train_idx, y_col]
        x_valid = features.loc[target_valid_idx]
        y_valid = labels.loc[target_valid_idx, y_col]

        model = make_model(args.random_state)
        fit_kwargs = {}
        if args.use_sample_weight:
            fit_kwargs["sample_weight"] = sample_weight(y_train, capacity)

        print(f"Training {target}: train={len(x_train)}, valid={len(x_valid)}, features={x_train.shape[1]}")
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="l1",
            callbacks=[early_stopping(100), log_evaluation(100)],
            **fit_kwargs,
        )

        pred = model.predict(features.loc[valid_idx], num_iteration=model.best_iteration_)
        pred = np.clip(pred, 0.0, capacity)
        predictions[target] = pred

        model_rows.append(
            {
                "target": target,
                "train_rows": int(len(x_train)),
                "valid_rows": int(len(x_valid)),
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
                "best_l1": float(model.best_score_["valid_0"]["l1"]),
            }
        )

        importance = pd.DataFrame(
            {
                "target": target,
                "feature": features.columns,
                "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                "importance_split": model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values(["target", "importance_gain"], ascending=[True, False])
        importance_frames.append(importance)

    answer = labels.loc[valid_idx, TARGET_COLS].copy()
    for target in TARGET_COLS:
        missing_answer = answer[target].isna()
        answer.loc[missing_answer, target] = labels.loc[missing_answer[missing_answer].index, f"{target}_filled"]

    metrics = competition_metric(answer, predictions[TARGET_COLS])
    metrics["models"] = model_rows
    metrics["valid_start"] = str(valid_start)
    metrics["use_sample_weight"] = bool(args.use_sample_weight)

    pred_out = predictions.reset_index().rename(columns={"index": "forecast_kst_dtm"})
    pred_out.to_csv(args.out_dir / "validation_lgbm_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(importance_frames, ignore_index=True).to_csv(
        args.out_dir / "validation_lgbm_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(model_rows).to_csv(args.out_dir / "validation_lgbm_model_summary.csv", index=False, encoding="utf-8-sig")
    with open(args.out_dir / "validation_lgbm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
