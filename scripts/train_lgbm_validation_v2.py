from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric import CAPACITY_KWH, TARGET_COLS
from train_lgbm_validation_fast import (
    load_processed,
    make_model,
    sample_weight,
    select_target_features,
)


REFERENCE_ITERATIONS = {
    "kpx_group_1": [465, 581, 697],
    "kpx_group_2": [506, 632, 758],
    "kpx_group_3": [168, 210, 252],
}

METRIC_COLUMNS = ["score", "one_minus_nmae", "ficr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward LightGBM validation with iteration sweeps and seed ensembles."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/validation_v2"))
    parser.add_argument("--fold-years", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--max-estimators", type=int, default=1200)
    parser.add_argument("--iteration-step", type=int, default=50)
    parser.add_argument("--use-sample-weight", action="store_true", default=True)
    parser.add_argument("--no-sample-weight", action="store_false", dest="use_sample_weight")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def annual_fold_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Hourly labels use 01:00 through the following year's 00:00 as one year.
    start = pd.Timestamp(year=year, month=1, day=1, hour=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1, hour=1)
    return start, end


def build_checkpoints(max_estimators: int, step: int, target: str) -> list[int]:
    if max_estimators < 1:
        raise ValueError("--max-estimators must be positive.")
    if step < 1:
        raise ValueError("--iteration-step must be positive.")

    checkpoints = set(range(step, max_estimators + 1, step))
    checkpoints.add(max_estimators)
    checkpoints.update(i for i in REFERENCE_ITERATIONS[target] if i <= max_estimators)
    return sorted(checkpoints)


def evaluate_group(actual: np.ndarray, forecast: np.ndarray, capacity: float) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(forecast) & (actual >= capacity * 0.10)
    if not valid.any():
        raise ValueError("No official evaluation rows at or above 10% capacity.")

    actual_valid = actual[valid]
    forecast_valid = forecast[valid]
    error_rate = np.abs(forecast_valid - actual_valid) / capacity
    nmae = float(np.mean(error_rate))

    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    ficr = float(np.sum(actual_valid * unit_price) / np.sum(actual_valid * 4.0))
    one_minus_nmae = 1.0 - nmae

    return {
        "score": 0.5 * one_minus_nmae + 0.5 * ficr,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "nmae": nmae,
        "within_6pct": float(np.mean(error_rate <= 0.06)),
        "within_8pct": float(np.mean(error_rate <= 0.08)),
        "evaluated_rows": int(valid.sum()),
    }


def metric_row(
    fold: str,
    target: str,
    scope: str,
    iteration: int,
    actual: np.ndarray,
    forecast: np.ndarray,
    capacity: float,
    train_rows: int,
    valid_rows: int,
) -> dict[str, object]:
    return {
        "fold": fold,
        "target": target,
        "scope": scope,
        "iteration": int(iteration),
        "train_rows": int(train_rows),
        "valid_rows": int(valid_rows),
        **evaluate_group(actual, forecast, capacity),
    }


def summarize_iterations(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensemble = metrics.loc[metrics["scope"] == "ensemble"].copy()
    best_by_fold = (
        ensemble.sort_values(
            ["fold", "target", "score", "ficr", "iteration"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["fold", "target"], as_index=False)
        .first()
    )

    robust = (
        ensemble.groupby(["target", "iteration"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            std_score=("score", "std"),
            mean_one_minus_nmae=("one_minus_nmae", "mean"),
            mean_ficr=("ficr", "mean"),
        )
        .fillna({"std_score": 0.0})
    )

    recommended = (
        robust.sort_values(
            ["target", "mean_score", "min_score", "iteration"],
            ascending=[True, False, False, True],
        )
        .groupby("target", as_index=False)
        .first()
    )
    recommended["validation_warning"] = np.where(
        recommended["folds"] < 2,
        "single annual fold; treat as provisional",
        "",
    )
    return best_by_fold, robust, recommended


def plot_metric_curves(
    metrics: pd.DataFrame,
    recommended: pd.DataFrame,
    out_path: Path,
) -> None:
    ensemble = metrics.loc[metrics["scope"] == "ensemble"].copy()
    available_targets = [target for target in TARGET_COLS if target in set(ensemble["target"])]
    fig, axes = plt.subplots(
        len(available_targets),
        len(METRIC_COLUMNS),
        figsize=(16, 4.2 * len(available_targets)),
        squeeze=False,
        sharex="col",
    )
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    for row_idx, target in enumerate(available_targets):
        target_data = ensemble.loc[ensemble["target"] == target]
        rec_iteration = int(recommended.loc[recommended["target"] == target, "iteration"].iloc[0])

        for col_idx, metric_name in enumerate(METRIC_COLUMNS):
            ax = axes[row_idx, col_idx]
            for color, (fold, fold_data) in zip(colors, target_data.groupby("fold", sort=True)):
                fold_data = fold_data.sort_values("iteration")
                ax.plot(
                    fold_data["iteration"],
                    fold_data[metric_name],
                    color=color,
                    linewidth=1.8,
                    label=fold,
                )
            ax.axvline(rec_iteration, color="#333333", linestyle="--", linewidth=1.2)
            ax.grid(alpha=0.25)
            ax.set_title(f"{target} | {metric_name}")
            ax.set_xlabel("Trees")
            ax.set_ylabel(metric_name)
            if col_idx == 0:
                ax.legend(frameon=False)

    fig.suptitle("BARAM validation v2: exact metric by iteration (seed ensemble)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_seed_comparison(metrics: pd.DataFrame, out_path: Path) -> None:
    best = (
        metrics.sort_values(
            ["fold", "target", "scope", "score", "iteration"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["fold", "target", "scope"], as_index=False)
        .first()
    )
    panels = list(best.groupby(["fold", "target"], sort=True))
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10, max(3.0, 2.4 * len(panels))),
        squeeze=False,
    )

    for ax, ((fold, target), panel) in zip(axes[:, 0], panels):
        panel = panel.sort_values("score")
        colors = ["#009E73" if scope == "ensemble" else "#0072B2" for scope in panel["scope"]]
        ax.barh(panel["scope"], panel["score"], color=colors)
        ax.set_title(f"{fold} | {target}: best score by seed scope")
        ax.set_xlabel("Best score across iteration checkpoints")
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def dataframe_html(df: pd.DataFrame, digits: int = 6) -> str:
    return df.round(digits).to_html(index=False, border=0, classes="data-table")


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    best_by_fold: pd.DataFrame,
    recommended: pd.DataFrame,
    skipped: list[dict[str, object]],
) -> None:
    skipped_df = pd.DataFrame(skipped)
    skipped_html = "<p>None</p>" if skipped_df.empty else dataframe_html(skipped_df)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BARAM validation v2</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #202124; background: #ffffff; }}
h1, h2 {{ margin: 24px 0 10px; }}
p {{ line-height: 1.55; }}
.note {{ border-left: 4px solid #D55E00; padding: 10px 14px; background: #fff7f0; }}
.data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
.data-table th, .data-table td {{ border-bottom: 1px solid #dadce0; padding: 8px; text-align: right; }}
.data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
img {{ width: 100%; max-width: 1500px; height: auto; margin: 10px 0 28px; }}
</style>
</head>
<body>
<h1>BARAM validation v2</h1>
<p>Seeds: {args.seeds} | Max trees: {args.max_estimators} | Step: {args.iteration_step} | Weighted: {args.use_sample_weight}</p>
<p class="note">Recommendations maximize the mean exact competition score across available annual folds. Group 3 normally has only the 2024 annual fold because 2022 labels are unavailable.</p>
<h2>Recommended iterations</h2>
{dataframe_html(recommended)}
<h2>Best iteration by fold</h2>
{dataframe_html(best_by_fold)}
<h2>Skipped fold-target pairs</h2>
{skipped_html}
<h2>Metric curves</h2>
<img src="validation_v2_metric_curves.png" alt="metric curves">
<h2>Seed comparison</h2>
<img src="validation_v2_seed_comparison.png" alt="seed comparison">
</body>
</html>
"""
    (out_dir / "validation_v2_report.html").write_text(html, encoding="utf-8")


def print_dry_run_plan(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    rows = []
    for year in args.fold_years:
        valid_start, valid_end = annual_fold_bounds(year)
        train_time = features.index < valid_start
        valid_time = (features.index >= valid_start) & (features.index < valid_end)
        for target in TARGET_COLS:
            y_col = f"{target}_filled"
            train_rows = int((train_time & labels[y_col].notna()).sum())
            valid_rows = int((valid_time & labels[target].notna()).sum())
            rows.append(
                {
                    "fold": f"valid_{year}",
                    "target": target,
                    "train_rows": train_rows,
                    "official_valid_rows": valid_rows,
                    "features": len(select_target_features(features, target)),
                    "checkpoints": len(build_checkpoints(args.max_estimators, args.iteration_step, target)),
                    "will_run": train_rows > 0 and valid_rows > 0,
                }
            )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Seeds: {args.seeds}", flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features, labels = load_processed(args.processed_dir)

    if args.dry_run:
        print_dry_run_plan(features, labels, args)
        return

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for year in args.fold_years:
        fold = f"valid_{year}"
        valid_start, valid_end = annual_fold_bounds(year)
        train_time = features.index < valid_start
        valid_time = (features.index >= valid_start) & (features.index < valid_end)

        for target in TARGET_COLS:
            y_col = f"{target}_filled"
            capacity = float(CAPACITY_KWH[target])
            columns = select_target_features(features, target)
            train_mask = train_time & labels[y_col].notna()
            official_valid_mask = valid_time & labels[target].notna()
            train_rows = int(train_mask.sum())
            valid_rows = int(official_valid_mask.sum())

            if train_rows == 0 or valid_rows == 0:
                skipped.append(
                    {
                        "fold": fold,
                        "target": target,
                        "train_rows": train_rows,
                        "official_valid_rows": valid_rows,
                        "reason": "no train labels" if train_rows == 0 else "no official validation labels",
                    }
                )
                print(f"Skipping {fold} {target}: train={train_rows}, valid={valid_rows}", flush=True)
                continue

            x_train = features.loc[train_mask, columns]
            y_train = labels.loc[train_mask, y_col]
            x_valid = features.loc[official_valid_mask, columns]
            actual = labels.loc[official_valid_mask, target].to_numpy(dtype=float)
            checkpoints = build_checkpoints(args.max_estimators, args.iteration_step, target)
            predictions_by_iteration: dict[int, list[np.ndarray]] = {i: [] for i in checkpoints}

            for seed in args.seeds:
                print(
                    f"Training {fold} {target}: seed={seed}, train={train_rows}, "
                    f"valid={valid_rows}, features={len(columns)}, trees={args.max_estimators}",
                    flush=True,
                )
                model = make_model(seed, args.max_estimators)
                fit_kwargs = {}
                if args.use_sample_weight:
                    fit_kwargs["sample_weight"] = sample_weight(y_train, capacity)
                model.fit(x_train, y_train, **fit_kwargs)

                for iteration in checkpoints:
                    prediction = np.clip(
                        model.predict(x_valid, num_iteration=iteration),
                        0.0,
                        capacity,
                    )
                    predictions_by_iteration[iteration].append(prediction)
                    rows.append(
                        metric_row(
                            fold,
                            target,
                            f"seed_{seed}",
                            iteration,
                            actual,
                            prediction,
                            capacity,
                            train_rows,
                            valid_rows,
                        )
                    )

            for iteration, seed_predictions in predictions_by_iteration.items():
                ensemble_prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0)
                rows.append(
                    metric_row(
                        fold,
                        target,
                        "ensemble",
                        iteration,
                        actual,
                        ensemble_prediction,
                        capacity,
                        train_rows,
                        valid_rows,
                    )
                )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No fold-target pair could be evaluated.")

    best_by_fold, robust, recommended = summarize_iterations(metrics)
    metrics.to_csv(args.out_dir / "validation_v2_iteration_metrics.csv", index=False, encoding="utf-8-sig")
    best_by_fold.to_csv(args.out_dir / "validation_v2_best_by_fold.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(args.out_dir / "validation_v2_robust_iteration_summary.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(args.out_dir / "validation_v2_recommended_iterations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped).to_csv(args.out_dir / "validation_v2_skipped.csv", index=False, encoding="utf-8-sig")

    plot_metric_curves(metrics, recommended, args.out_dir / "validation_v2_metric_curves.png")
    plot_seed_comparison(metrics, args.out_dir / "validation_v2_seed_comparison.png")
    write_report(args.out_dir, args, best_by_fold, recommended, skipped)

    summary = {
        "fold_years": args.fold_years,
        "seeds": args.seeds,
        "max_estimators": args.max_estimators,
        "iteration_step": args.iteration_step,
        "use_sample_weight": args.use_sample_weight,
        "recommended_iterations": recommended.to_dict(orient="records"),
        "skipped": skipped,
    }
    with open(args.out_dir / "validation_v2_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nRecommended iterations", flush=True)
    print(recommended.to_string(index=False), flush=True)
    print(f"\nSaved report: {(args.out_dir / 'validation_v2_report.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
