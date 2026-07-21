from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from metric import CAPACITY_KWH, TARGET_COLS


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


DEFAULT_ITERATIONS = {
    "kpx_group_1": 581,
    "kpx_group_2": 632,
    "kpx_group_3": 210,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full-data LightGBM models and create a Dacon submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--output-name", type=str, default="lgbm_weighted_group3_monthly_submission.csv")
    parser.add_argument("--model-summary", type=Path, default=Path("reports/validation_lgbm_fast_weighted_model_summary.csv"))
    parser.add_argument("--monthly-calibration", type=Path, default=Path("reports/group3_monthly_calibration/group3_monthly_calibration_params.csv"))
    parser.add_argument("--iteration-multiplier", type=float, default=1.0)
    parser.add_argument("--device-type", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--use-sample-weight", action="store_true", default=True)
    parser.add_argument("--no-sample-weight", action="store_false", dest="use_sample_weight")
    parser.add_argument("--apply-group3-monthly-calibration", action="store_true", default=True)
    parser.add_argument("--no-group3-monthly-calibration", action="store_false", dest="apply_group3_monthly_calibration")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_iterations(path: Path, multiplier: float) -> dict[str, int]:
    iterations = DEFAULT_ITERATIONS.copy()
    if path.exists():
        summary = pd.read_csv(path, encoding="utf-8-sig")
        for row in summary.itertuples(index=False):
            target = getattr(row, "target")
            best_iteration = int(getattr(row, "best_iteration"))
            iterations[target] = best_iteration
    return {target: max(50, int(round(value * multiplier))) for target, value in iterations.items()}


def load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_train = pd.read_pickle(processed_dir / "features_train.pkl")
    features_test = pd.read_pickle(processed_dir / "features_test.pkl")
    labels = pd.read_pickle(processed_dir / "labels_processed.pkl")

    features_train.index = pd.to_datetime(features_train.index)
    features_test.index = pd.to_datetime(features_test.index)
    labels.index = pd.to_datetime(labels.index)

    common = features_train.index.intersection(labels.index)
    features_train = features_train.loc[common].sort_index().replace([np.inf, -np.inf], np.nan)
    labels = labels.loc[common].sort_index()
    features_test = features_test.sort_index().replace([np.inf, -np.inf], np.nan)
    return features_train, features_test, labels


def select_target_features(features: pd.DataFrame, target: str) -> list[str]:
    group_id = target.rsplit("_", 1)[-1]
    cols = []
    for col in features.columns:
        if f"_grp{group_id}_near_" in col:
            cols.append(col)
        elif col in TIME_FEATURES:
            cols.append(col)
        elif col.endswith("_lead_hours") or col.endswith("_available_hour"):
            cols.append(col)
    return cols


def make_model(random_state: int, n_estimators: int, device_type: str, gpu_device_id: int) -> LGBMRegressor:
    device_params = {}
    if device_type != "cpu":
        device_params = {
            "device_type": device_type,
            "gpu_device_id": gpu_device_id,
        }

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=0.20,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
        **device_params,
    )


def sample_weight(y: pd.Series, capacity: float) -> np.ndarray:
    y_arr = y.to_numpy(dtype=float)
    norm = np.clip(y_arr / capacity, 0.0, 1.0)
    return np.where(y_arr >= capacity * 0.10, 1.0 + norm, 0.30).astype("float32")


def apply_group3_monthly_calibration(pred: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Monthly calibration file not found: {path}")

    params = pd.read_csv(path, encoding="utf-8-sig")
    out = pred.copy()
    capacity = CAPACITY_KWH["kpx_group_3"]
    for row in params.itertuples(index=False):
        month = int(getattr(row, "month"))
        scale = float(getattr(row, "scale"))
        bias = float(getattr(row, "bias"))
        month_mask = out.index.month == month
        out.loc[month_mask, "kpx_group_3"] = np.clip(
            out.loc[month_mask, "kpx_group_3"].to_numpy(dtype=float) * scale + bias,
            0.0,
            capacity,
        )
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    iterations = load_iterations(args.model_summary, args.iteration_multiplier)
    features_train, features_test, labels = load_processed(args.processed_dir)
    predictions = pd.DataFrame(index=features_test.index)
    model_rows = []

    for target in TARGET_COLS:
        y_col = f"{target}_filled"
        capacity = CAPACITY_KWH[target]
        cols = select_target_features(features_train, target)
        train_mask = labels[y_col].notna()

        x_train = features_train.loc[train_mask, cols]
        y_train = labels.loc[train_mask, y_col]
        x_test = features_test[cols]

        n_estimators = iterations[target]
        model = make_model(args.random_state, n_estimators, args.device_type, args.gpu_device_id)
        fit_kwargs = {}
        if args.use_sample_weight:
            fit_kwargs["sample_weight"] = sample_weight(y_train, capacity)

        print(
            f"Training {target}: rows={len(x_train)}, features={len(cols)}, n_estimators={n_estimators}, "
            f"sample_weight={args.use_sample_weight}, device_type={args.device_type}, gpu_device_id={args.gpu_device_id}",
            flush=True,
        )
        model.fit(x_train, y_train, **fit_kwargs)

        pred = model.predict(x_test)
        predictions[target] = np.clip(pred, 0.0, capacity)
        model_rows.append(
            {
                "target": target,
                "train_rows": int(len(x_train)),
                "features": int(len(cols)),
                "n_estimators": int(n_estimators),
                "sample_weight": bool(args.use_sample_weight),
                "device_type": args.device_type,
                "gpu_device_id": int(args.gpu_device_id),
            }
        )

    if args.apply_group3_monthly_calibration:
        print(f"Applying group 3 monthly calibration: {args.monthly_calibration}", flush=True)
        predictions = apply_group3_monthly_calibration(predictions, args.monthly_calibration)

    sample = pd.read_csv(args.data_dir / "sample_submission.csv", encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    submission = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    aligned_pred = predictions.loc[pd.to_datetime(submission["forecast_kst_dtm"]).to_numpy()]
    for target in TARGET_COLS:
        submission[target] = aligned_pred[target].to_numpy()
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path = args.out_dir / args.output_name
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "output_path": str(out_path),
        "rows": int(len(submission)),
        "models": model_rows,
        "group3_monthly_calibration": bool(args.apply_group3_monthly_calibration),
        "monthly_calibration_path": str(args.monthly_calibration) if args.apply_group3_monthly_calibration else None,
        "device_type": args.device_type,
        "gpu_device_id": int(args.gpu_device_id),
        "prediction_summary": {
            target: {
                "min": float(submission[target].min()),
                "mean": float(submission[target].mean()),
                "max": float(submission[target].max()),
            }
            for target in TARGET_COLS
        },
    }
    with open(args.out_dir / out_path.with_suffix(".json").name, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission: {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
