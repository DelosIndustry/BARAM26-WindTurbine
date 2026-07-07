from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metric import CAPACITY_KWH, TARGET_COLS, competition_metric


TARGET = "kpx_group_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune group 3 scale/bias post-processing on validation predictions.")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--pred-path", type=Path, default=Path("reports/validation_lgbm_fast_weighted_predictions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/group3_calibration"))
    parser.add_argument("--valid-start", type=str, default="2024-01-01 01:00:00")
    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.20)
    parser.add_argument("--scale-step", type=float, default=0.005)
    parser.add_argument("--bias-min", type=float, default=-2000.0)
    parser.add_argument("--bias-max", type=float, default=2000.0)
    parser.add_argument("--bias-step", type=float, default=50.0)
    return parser.parse_args()


def load_answer(processed_dir: Path, valid_start: str, prediction_index: pd.Index) -> pd.DataFrame:
    labels = pd.read_pickle(processed_dir / "labels_processed.pkl")
    labels.index = pd.to_datetime(labels.index)
    valid_start_ts = pd.Timestamp(valid_start)
    labels = labels.loc[labels.index >= valid_start_ts].copy()
    labels = labels.loc[prediction_index]

    answer = labels[TARGET_COLS].copy()
    for target in TARGET_COLS:
        missing = answer[target].isna()
        if missing.any():
            answer.loc[missing, target] = labels.loc[answer.index[missing], f"{target}_filled"]
    return answer


def group_metric(actual: np.ndarray, pred: np.ndarray, capacity: float) -> dict:
    valid = np.isfinite(actual) & np.isfinite(pred) & (actual >= capacity * 0.10)
    actual = actual[valid]
    pred = pred[valid]
    error_rate = np.abs(pred - actual) / capacity
    signed_error_rate = (pred - actual) / capacity
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    return {
        "valid_rows": int(valid.sum()),
        "nmae": float(np.mean(error_rate)),
        "ficr": float(np.sum(actual * unit_price) / np.sum(actual * 4.0)),
        "within_6pct": float(np.mean(error_rate <= 0.06)),
        "within_8pct": float(np.mean(error_rate <= 0.08)),
        "signed_bias": float(np.mean(signed_error_rate)),
        "mae_kwh": float(np.mean(np.abs(pred - actual))),
    }


def score_from_group_metrics(group_metrics: dict[str, dict]) -> dict:
    mean_nmae = float(np.mean([group_metrics[target]["nmae"] for target in TARGET_COLS]))
    mean_ficr = float(np.mean([group_metrics[target]["ficr"] for target in TARGET_COLS]))
    one_minus_nmae = 1.0 - mean_nmae
    score = 0.5 * one_minus_nmae + 0.5 * mean_ficr
    return {
        "score": score,
        "one_minus_nmae": one_minus_nmae,
        "ficr": mean_ficr,
    }


def make_calibrated(pred: pd.DataFrame, scale: float, bias: float) -> pd.DataFrame:
    out = pred.copy()
    out[TARGET] = np.clip(out[TARGET].to_numpy(dtype=float) * scale + bias, 0.0, CAPACITY_KWH[TARGET])
    return out


def tune(answer: pd.DataFrame, pred: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict, dict]:
    base_metrics = competition_metric(answer, pred[TARGET_COLS])
    base_group_metrics = {
        target: group_metric(
            answer[target].to_numpy(dtype=float),
            pred[target].to_numpy(dtype=float),
            CAPACITY_KWH[target],
        )
        for target in TARGET_COLS
    }

    scales = np.round(np.arange(args.scale_min, args.scale_max + args.scale_step / 2, args.scale_step), 6)
    biases = np.round(np.arange(args.bias_min, args.bias_max + args.bias_step / 2, args.bias_step), 6)

    actual_g3 = answer[TARGET].to_numpy(dtype=float)
    pred_g3 = pred[TARGET].to_numpy(dtype=float)

    rows = []
    for scale in scales:
        scaled = pred_g3 * scale
        for bias in biases:
            calibrated_g3 = np.clip(scaled + bias, 0.0, CAPACITY_KWH[TARGET])
            metrics_g3 = group_metric(actual_g3, calibrated_g3, CAPACITY_KWH[TARGET])
            group_metrics = dict(base_group_metrics)
            group_metrics[TARGET] = metrics_g3
            total = score_from_group_metrics(group_metrics)
            rows.append(
                {
                    "scale": float(scale),
                    "bias": float(bias),
                    "score": total["score"],
                    "one_minus_nmae": total["one_minus_nmae"],
                    "ficr": total["ficr"],
                    "g3_nmae": metrics_g3["nmae"],
                    "g3_ficr": metrics_g3["ficr"],
                    "g3_within_6pct": metrics_g3["within_6pct"],
                    "g3_within_8pct": metrics_g3["within_8pct"],
                    "g3_signed_bias": metrics_g3["signed_bias"],
                    "g3_mae_kwh": metrics_g3["mae_kwh"],
                }
            )

    sweep = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    best = sweep.iloc[0].to_dict()
    best_pred = make_calibrated(pred, best["scale"], best["bias"])
    best_metrics = competition_metric(answer, best_pred[TARGET_COLS])
    return sweep, base_metrics, {"best": best, "metrics": best_metrics, "pred": best_pred}


def valid_mask(answer: pd.DataFrame, target: str) -> pd.Series:
    return answer[target] >= CAPACITY_KWH[target] * 0.10


def monthly_metrics(answer: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, month_answer in answer.groupby(answer.index.month):
        idx = month_answer.index
        for label, pred in [("before", before), ("after", after)]:
            metrics = group_metric(
                month_answer[TARGET].to_numpy(dtype=float),
                pred.loc[idx, TARGET].to_numpy(dtype=float),
                CAPACITY_KWH[TARGET],
            )
            rows.append({"month": int(month), "version": label, **metrics})
    return pd.DataFrame(rows)


def plot_scatter(answer: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> None:
    mask = valid_mask(answer, TARGET)
    actual = answer.loc[mask, TARGET].to_numpy(dtype=float)
    cap = CAPACITY_KWH[TARGET]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, pred_df, title in [
        (axes[0], before, "Before calibration"),
        (axes[1], after, "After calibration"),
    ]:
        pred = pred_df.loc[mask, TARGET].to_numpy(dtype=float)
        ax.scatter(actual, pred, s=8, alpha=0.35, edgecolors="none")
        x = np.linspace(0, cap, 200)
        ax.plot(x, x, color="black", linewidth=1.2, label="perfect")
        ax.plot(x, x + cap * 0.06, color="#2ca02c", linewidth=0.9, linestyle="--", label="6% band")
        ax.plot(x, x - cap * 0.06, color="#2ca02c", linewidth=0.9, linestyle="--")
        ax.plot(x, x + cap * 0.08, color="#ff7f0e", linewidth=0.9, linestyle=":", label="8% band")
        ax.plot(x, x - cap * 0.08, color="#ff7f0e", linewidth=0.9, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Actual kWh")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted kWh")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Group 3 Validation Scatter")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_distribution(answer: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> None:
    mask = valid_mask(answer, TARGET)
    cap = CAPACITY_KWH[TARGET]
    actual = answer.loc[mask, TARGET].to_numpy(dtype=float)
    before_error = (before.loc[mask, TARGET].to_numpy(dtype=float) - actual) / cap
    after_error = (after.loc[mask, TARGET].to_numpy(dtype=float) - actual) / cap

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(-0.35, 0.35, 80)
    axes[0].hist(before_error, bins=bins, alpha=0.60, label="before")
    axes[0].hist(after_error, bins=bins, alpha=0.60, label="after")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].axvline(0.06, color="#2ca02c", linestyle="--", linewidth=1)
    axes[0].axvline(-0.06, color="#2ca02c", linestyle="--", linewidth=1)
    axes[0].axvline(0.08, color="#ff7f0e", linestyle=":", linewidth=1)
    axes[0].axvline(-0.08, color="#ff7f0e", linestyle=":", linewidth=1)
    axes[0].set_title("Signed error / capacity")
    axes[0].set_xlabel("(prediction - actual) / capacity")
    axes[0].legend()

    abs_bins = np.linspace(0, 0.35, 70)
    axes[1].hist(np.abs(before_error), bins=abs_bins, alpha=0.60, label="before")
    axes[1].hist(np.abs(after_error), bins=abs_bins, alpha=0.60, label="after")
    axes[1].axvline(0.06, color="#2ca02c", linestyle="--", linewidth=1, label="6%")
    axes[1].axvline(0.08, color="#ff7f0e", linestyle=":", linewidth=1, label="8%")
    axes[1].set_title("Absolute error / capacity")
    axes[1].set_xlabel("absolute error / capacity")
    axes[1].legend()

    fig.suptitle("Group 3 Error Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(sweep: pd.DataFrame, best: dict, out_path: Path) -> None:
    pivot = sweep.pivot(index="scale", columns="bias", values="score").sort_index()
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        origin="lower",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        cmap="viridis",
    )
    ax.scatter([best["bias"]], [best["scale"]], color="red", s=70, marker="x", linewidths=2, label="best")
    ax.set_xlabel("Bias added to group 3 prediction")
    ax.set_ylabel("Scale multiplied to group 3 prediction")
    ax.set_title("Group 3 Calibration Sweep: Total Score")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Total validation score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_monthly(monthly: pd.DataFrame, out_path: Path) -> None:
    months = sorted(monthly["month"].unique())
    before = monthly[monthly["version"] == "before"].set_index("month").loc[months]
    after = monthly[monthly["version"] == "after"].set_index("month").loc[months]
    x = np.arange(len(months))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(x - width / 2, before["nmae"], width, label="before")
    axes[0].bar(x + width / 2, after["nmae"], width, label="after")
    axes[0].set_ylabel("NMAE")
    axes[0].set_title("Group 3 Monthly NMAE")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, before["ficr"], width, label="before")
    axes[1].bar(x + width / 2, after["ficr"], width, label="after")
    axes[1].set_ylabel("FICR")
    axes[1].set_title("Group 3 Monthly FICR")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(months)
    axes[1].set_xlabel("Month")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_daily_timeseries(answer: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> None:
    daily = pd.DataFrame(
        {
            "actual": answer[TARGET],
            "before": before[TARGET],
            "after": after[TARGET],
        }
    ).resample("D").mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily["actual"], label="actual", color="black", linewidth=1.2)
    ax.plot(daily.index, daily["before"], label="before", alpha=0.75)
    ax.plot(daily.index, daily["after"], label="after", alpha=0.75)
    ax.set_title("Group 3 Daily Mean Power")
    ax.set_ylabel("kWh")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_html_report(out_dir: Path, before_metrics: dict, after_metrics: dict, best: dict) -> None:
    def fmt(x: float) -> str:
        return f"{x:.6f}"

    before_g3 = next(row for row in before_metrics["groups"] if row["target"] == TARGET)
    after_g3 = next(row for row in after_metrics["groups"] if row["target"] == TARGET)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Group 3 Calibration Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f2f2f2; }}
    img {{ max-width: 100%; margin: 12px 0 28px; border: 1px solid #ddd; }}
    .note {{ color: #555; }}
  </style>
</head>
<body>
  <h1>Group 3 Calibration Report</h1>
  <p class="note">Validation period starts at 2024-01-01 01:00:00. Calibration form: pred = clip(pred * scale + bias, 0, 21000).</p>
  <h2>Best Calibration</h2>
  <table>
    <tr><th>scale</th><th>bias</th><th>score</th><th>g3_nmae</th><th>g3_ficr</th><th>g3_within_6pct</th><th>g3_within_8pct</th></tr>
    <tr><td>{fmt(best["scale"])}</td><td>{fmt(best["bias"])}</td><td>{fmt(best["score"])}</td><td>{fmt(best["g3_nmae"])}</td><td>{fmt(best["g3_ficr"])}</td><td>{fmt(best["g3_within_6pct"])}</td><td>{fmt(best["g3_within_8pct"])}</td></tr>
  </table>
  <h2>Before vs After</h2>
  <table>
    <tr><th>metric</th><th>before</th><th>after</th><th>delta</th></tr>
    <tr><td>Total score</td><td>{fmt(before_metrics["score"])}</td><td>{fmt(after_metrics["score"])}</td><td>{fmt(after_metrics["score"] - before_metrics["score"])}</td></tr>
    <tr><td>1-NMAE</td><td>{fmt(before_metrics["one_minus_nmae"])}</td><td>{fmt(after_metrics["one_minus_nmae"])}</td><td>{fmt(after_metrics["one_minus_nmae"] - before_metrics["one_minus_nmae"])}</td></tr>
    <tr><td>FICR</td><td>{fmt(before_metrics["ficr"])}</td><td>{fmt(after_metrics["ficr"])}</td><td>{fmt(after_metrics["ficr"] - before_metrics["ficr"])}</td></tr>
    <tr><td>Group 3 NMAE</td><td>{fmt(before_g3["nmae"])}</td><td>{fmt(after_g3["nmae"])}</td><td>{fmt(after_g3["nmae"] - before_g3["nmae"])}</td></tr>
    <tr><td>Group 3 FICR</td><td>{fmt(before_g3["ficr"])}</td><td>{fmt(after_g3["ficr"])}</td><td>{fmt(after_g3["ficr"] - before_g3["ficr"])}</td></tr>
  </table>
  <h2>Visuals</h2>
  <img src="group3_calibration_heatmap.png" alt="calibration heatmap">
  <img src="group3_scatter_before_after.png" alt="scatter before after">
  <img src="group3_error_distribution.png" alt="error distribution">
  <img src="group3_monthly_metrics.png" alt="monthly metrics">
  <img src="group3_daily_timeseries.png" alt="daily timeseries">
</body>
</html>
"""
    (out_dir / "group3_calibration_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    pred = pred.set_index("forecast_kst_dtm").sort_index()
    answer = load_answer(args.processed_dir, args.valid_start, pred.index)

    sweep, base_metrics, best_bundle = tune(answer, pred, args)
    best = best_bundle["best"]
    after_metrics = best_bundle["metrics"]
    calibrated = best_bundle["pred"]

    sweep.to_csv(args.out_dir / "group3_calibration_sweep.csv", index=False, encoding="utf-8-sig")
    sweep.head(30).to_csv(args.out_dir / "group3_calibration_top30.csv", index=False, encoding="utf-8-sig")
    calibrated.reset_index().to_csv(args.out_dir / "validation_lgbm_fast_weighted_group3_calibrated_predictions.csv", index=False, encoding="utf-8-sig")

    monthly = monthly_metrics(answer, pred, calibrated)
    monthly.to_csv(args.out_dir / "group3_monthly_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "target": TARGET,
        "calibration": {
            "scale": best["scale"],
            "bias": best["bias"],
        },
        "before": base_metrics,
        "after": after_metrics,
        "best_sweep_row": best,
    }
    with open(args.out_dir / "group3_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_heatmap(sweep, best, args.out_dir / "group3_calibration_heatmap.png")
    plot_scatter(answer, pred, calibrated, args.out_dir / "group3_scatter_before_after.png")
    plot_error_distribution(answer, pred, calibrated, args.out_dir / "group3_error_distribution.png")
    plot_monthly(monthly, args.out_dir / "group3_monthly_metrics.png")
    plot_daily_timeseries(answer, pred, calibrated, args.out_dir / "group3_daily_timeseries.png")
    write_html_report(args.out_dir, base_metrics, after_metrics, best)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report: {(args.out_dir / 'group3_calibration_report.html').resolve()}")


if __name__ == "__main__":
    main()
