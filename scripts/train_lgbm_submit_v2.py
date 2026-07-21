from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric import CAPACITY_KWH, TARGET_COLS
from train_lgbm_submit import (
    load_processed,
    make_model,
    sample_weight,
    select_target_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the validation-v2 LightGBM seed ensemble and create a Dacon submission."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--recommendations", type=Path, default=Path(
        "reports/validation_v2/validation_v2_recommended_iterations.csv"
    ))
    parser.add_argument("--out-dir", type=Path, default=Path("submissions"))
    parser.add_argument(
        "--output-name",
        type=str,
        default="lgbm_v2_group_iters_seed_ensemble_submission.csv",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--device-type", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--use-sample-weight", action="store_true", default=True)
    parser.add_argument("--no-sample-weight", action="store_false", dest="use_sample_weight")
    parser.add_argument(
        "--iteration-cap",
        type=int,
        default=None,
        help="Cap all recommended iterations; intended only for smoke tests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_recommended_iterations(path: Path, iteration_cap: int | None) -> tuple[dict[str, int], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Recommendation file not found: {path}")

    recommendations = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {"target", "iteration"}
    missing_columns = required_columns.difference(recommendations.columns)
    if missing_columns:
        raise ValueError(f"Recommendation file is missing columns: {sorted(missing_columns)}")

    duplicate_targets = recommendations.loc[recommendations["target"].duplicated(), "target"].tolist()
    if duplicate_targets:
        raise ValueError(f"Duplicate recommendation targets: {duplicate_targets}")

    iterations = {}
    for target in TARGET_COLS:
        target_rows = recommendations.loc[recommendations["target"] == target]
        if len(target_rows) != 1:
            raise ValueError(f"Expected one recommendation for {target}, found {len(target_rows)}")
        iteration = int(target_rows["iteration"].iloc[0])
        if iteration < 1:
            raise ValueError(f"Recommended iteration for {target} must be positive: {iteration}")
        if iteration_cap is not None:
            if iteration_cap < 1:
                raise ValueError("--iteration-cap must be positive.")
            iteration = min(iteration, iteration_cap)
        iterations[target] = iteration

    return iterations, recommendations


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected_columns = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise ValueError(f"Submission row count mismatch: {len(submission)} != {len(sample)}")
    if not submission["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError("forecast_id order differs from sample_submission.csv")

    for target in TARGET_COLS:
        values = submission[target].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite predictions found for {target}")
        capacity = float(CAPACITY_KWH[target])
        if np.any(values < 0.0) or np.any(values > capacity):
            raise ValueError(f"Predictions outside [0, {capacity}] for {target}")


def print_dry_run(
    features_train: pd.DataFrame,
    labels: pd.DataFrame,
    iterations: dict[str, int],
    args: argparse.Namespace,
) -> None:
    rows = []
    for target in TARGET_COLS:
        y_col = f"{target}_filled"
        train_mask = labels[y_col].notna()
        columns = select_target_features(features_train, target)
        for seed in args.seeds:
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "train_rows": int(train_mask.sum()),
                    "features": len(columns),
                    "n_estimators": iterations[target],
                    "sample_weight": args.use_sample_weight,
                    "device_type": args.device_type,
                }
            )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Output: {args.out_dir / args.output_name}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")

    iterations, recommendations = load_recommended_iterations(args.recommendations, args.iteration_cap)
    features_train, features_test, labels = load_processed(args.processed_dir)

    if args.dry_run:
        print_dry_run(features_train, labels, iterations, args)
        return

    predictions = pd.DataFrame(index=features_test.index)
    model_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []

    for target in TARGET_COLS:
        y_col = f"{target}_filled"
        capacity = float(CAPACITY_KWH[target])
        columns = select_target_features(features_train, target)
        train_mask = labels[y_col].notna()
        x_train = features_train.loc[train_mask, columns]
        y_train = labels.loc[train_mask, y_col]
        x_test = features_test[columns]
        n_estimators = iterations[target]
        member_predictions = []

        for seed in args.seeds:
            print(
                f"Training {target}: seed={seed}, rows={len(x_train)}, features={len(columns)}, "
                f"n_estimators={n_estimators}, sample_weight={args.use_sample_weight}, "
                f"device_type={args.device_type}, gpu_device_id={args.gpu_device_id}",
                flush=True,
            )
            model = make_model(seed, n_estimators, args.device_type, args.gpu_device_id)
            fit_kwargs = {}
            if args.use_sample_weight:
                fit_kwargs["sample_weight"] = sample_weight(y_train, capacity)
            model.fit(x_train, y_train, **fit_kwargs)

            member_prediction = np.clip(model.predict(x_test), 0.0, capacity)
            member_predictions.append(member_prediction)
            model_rows.append(
                {
                    "target": target,
                    "seed": int(seed),
                    "train_rows": int(len(x_train)),
                    "features": int(len(columns)),
                    "n_estimators": int(n_estimators),
                    "sample_weight": bool(args.use_sample_weight),
                    "device_type": args.device_type,
                    "gpu_device_id": int(args.gpu_device_id),
                    "prediction_min": float(np.min(member_prediction)),
                    "prediction_mean": float(np.mean(member_prediction)),
                    "prediction_max": float(np.max(member_prediction)),
                }
            )

        member_matrix = np.stack(member_predictions, axis=0)
        ensemble_prediction = np.clip(np.mean(member_matrix, axis=0), 0.0, capacity)
        member_std = np.std(member_matrix, axis=0)
        predictions[target] = ensemble_prediction
        ensemble_rows.append(
            {
                "target": target,
                "members": int(len(args.seeds)),
                "n_estimators": int(n_estimators),
                "prediction_min": float(np.min(ensemble_prediction)),
                "prediction_mean": float(np.mean(ensemble_prediction)),
                "prediction_max": float(np.max(ensemble_prediction)),
                "mean_member_std": float(np.mean(member_std)),
                "p95_member_std": float(np.quantile(member_std, 0.95)),
                "max_member_std": float(np.max(member_std)),
            }
        )

    sample = pd.read_csv(
        args.data_dir / "sample_submission.csv",
        encoding="utf-8-sig",
        parse_dates=["forecast_kst_dtm"],
    )
    submission = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    forecast_times = pd.to_datetime(submission["forecast_kst_dtm"])
    aligned_predictions = predictions.loc[forecast_times.to_numpy()]
    for target in TARGET_COLS:
        submission[target] = aligned_predictions[target].to_numpy()
    submission["forecast_kst_dtm"] = forecast_times.dt.strftime("%Y-%m-%d %H:%M:%S")

    sample_for_validation = sample.copy()
    sample_for_validation["forecast_kst_dtm"] = pd.to_datetime(
        sample_for_validation["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    validate_submission(submission, sample_for_validation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / args.output_name
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")

    recommendation_columns = [
        column
        for column in [
            "target",
            "iteration",
            "folds",
            "mean_score",
            "min_score",
            "std_score",
            "mean_one_minus_nmae",
            "mean_ficr",
            "validation_warning",
        ]
        if column in recommendations.columns
    ]
    summary = {
        "output_path": str(output_path),
        "rows": int(len(submission)),
        "recommendation_path": str(args.recommendations),
        "recommended_iterations_used": iterations,
        "seeds": [int(seed) for seed in args.seeds],
        "use_sample_weight": bool(args.use_sample_weight),
        "group3_calibration": False,
        "device_type": args.device_type,
        "gpu_device_id": int(args.gpu_device_id),
        "validation_recommendations": recommendations[recommendation_columns].to_dict(orient="records"),
        "models": model_rows,
        "ensemble_summary": ensemble_rows,
    }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_path": str(output_path.resolve()),
        "recommended_iterations_used": iterations,
        "seeds": args.seeds,
        "ensemble_summary": ensemble_rows,
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved submission: {output_path.resolve()}", flush=True)
    print(f"Saved metadata  : {json_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
