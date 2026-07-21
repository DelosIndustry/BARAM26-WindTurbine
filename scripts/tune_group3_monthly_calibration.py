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
    parser = argparse.ArgumentParser(description="Tune month-specific group 3 scale/bias calibration.")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--pred-path", type=Path, default=Path("reports/validation_lgbm_fast_weighted_predictions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/group3_monthly_calibration"))
    parser.add_argument("--valid-start", type=str, default="2024-01-01 01:00:00")
    parser.add_argument("--scale-min", type=float, default=0.80)
    parser.add_argument("--scale-max", type=float, default=1.50)
    parser.add_argument("--scale-step", type=float, default=0.01)
    parser.add_argument("--bias-min", type=float, default=-2500.0)
    parser.add_argument("--bias-max", type=float, default=2500.0)
    parser.add_argument("--bias-step", type=float, default=100.0)
    return parser.parse_args()


def load_answer(processed_dir: Path, valid_start: str, prediction_index: pd.Index) -> pd.DataFrame:
    labels = pd.read_pickle(processed_dir / "labels_processed.pkl")
    labels.index = pd.to_datetime(labels.index)
    labels = labels.loc[labels.index >= pd.Timestamp(valid_start)].copy()
    labels = labels.loc[prediction_index]

    answer = labels[TARGET_COLS].copy()
    for target in TARGET_COLS:
        missing = answer[target].isna()
        if missing.any():
            answer.loc[missing, target] = labels.loc[answer.index[missing], f"{target}_filled"]
    return answer


def group_metric(actual: np.ndarray, pred: np.ndarray, capacity: float) -> dict:
    valid = np.isfinite(actual) & np.isfinite(pred) & (actual >= capacity * 0.10)
    actual_valid = actual[valid]
    pred_valid = pred[valid]
    error_rate = np.abs(pred_valid - actual_valid) / capacity
    unit_price = np.select([error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0)
    return {
        "valid_rows": int(valid.sum()),
        "nmae": float(np.mean(error_rate)),
        "ficr": float(np.sum(actual_valid * unit_price) / np.sum(actual_valid * 4.0)),
        "within_6pct": float(np.mean(error_rate <= 0.06)),
        "within_8pct": float(np.mean(error_rate <= 0.08)),
        "signed_bias": float(np.mean((pred_valid - actual_valid) / capacity)),
        "mae_kwh": float(np.mean(np.abs(pred_valid - actual_valid))),
    }


def fixed_group_metrics(answer: pd.DataFrame, pred: pd.DataFrame) -> dict[str, dict]:
    return {
        target: group_metric(
            answer[target].to_numpy(dtype=float),
            pred[target].to_numpy(dtype=float),
            CAPACITY_KWH[target],
        )
        for target in TARGET_COLS
    }


def total_score_from_parts(
    fixed_metrics: dict[str, dict],
    g3_nmae: float,
    g3_ficr: float,
) -> dict:
    nmaes = [fixed_metrics["kpx_group_1"]["nmae"], fixed_metrics["kpx_group_2"]["nmae"], g3_nmae]
    ficrs = [fixed_metrics["kpx_group_1"]["ficr"], fixed_metrics["kpx_group_2"]["ficr"], g3_ficr]
    one_minus_nmae = 1.0 - float(np.mean(nmaes))
    ficr = float(np.mean(ficrs))
    return {
        "score": 0.5 * one_minus_nmae + 0.5 * ficr,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def tune_monthly(answer: pd.DataFrame, pred: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    capacity = CAPACITY_KWH[TARGET]
    fixed = fixed_group_metrics(answer, pred)

    actual = answer[TARGET].to_numpy(dtype=float)
    pred_arr = pred[TARGET].to_numpy(dtype=float)
    months = answer.index.month.to_numpy()
    valid = np.isfinite(actual) & np.isfinite(pred_arr) & (actual >= capacity * 0.10)

    base_error = np.abs(pred_arr - actual) / capacity
    base_unit_price = np.select([base_error <= 0.06, base_error <= 0.08], [4.0, 3.0], default=0.0)

    total_valid_count = int(valid.sum())
    total_max_settlement = float(np.sum(actual[valid] * 4.0))
    base_error_sum = float(np.sum(base_error[valid]))
    base_earned = float(np.sum(actual[valid] * base_unit_price[valid]))

    scales = np.round(np.arange(args.scale_min, args.scale_max + args.scale_step / 2, args.scale_step), 6)
    biases = np.round(np.arange(args.bias_min, args.bias_max + args.bias_step / 2, args.bias_step), 6)

    chosen_rows = []
    sweep_rows = []

    for month in range(1, 13):
        month_valid = valid & (months == month)
        if not month_valid.any():
            continue

        actual_m = actual[month_valid]
        pred_m = pred_arr[month_valid]
        base_error_sum_m = float(np.sum(base_error[month_valid]))
        base_earned_m = float(np.sum(actual[month_valid] * base_unit_price[month_valid]))

        other_error_sum = base_error_sum - base_error_sum_m
        other_earned = base_earned - base_earned_m

        best = None
        month_rows = []
        for scale in scales:
            scaled = pred_m * scale
            for bias in biases:
                calibrated = np.clip(scaled + bias, 0.0, capacity)
                error = np.abs(calibrated - actual_m) / capacity
                unit_price = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
                g3_nmae = (other_error_sum + float(np.sum(error))) / total_valid_count
                g3_ficr = (other_earned + float(np.sum(actual_m * unit_price))) / total_max_settlement
                total = total_score_from_parts(fixed, g3_nmae, g3_ficr)
                row = {
                    "month": month,
                    "scale": float(scale),
                    "bias": float(bias),
                    "score_if_only_this_month_changed": total["score"],
                    "one_minus_nmae": total["one_minus_nmae"],
                    "ficr": total["ficr"],
                    "g3_nmae": g3_nmae,
                    "g3_ficr": g3_ficr,
                    "month_nmae": float(np.mean(error)),
                    "month_ficr": float(np.sum(actual_m * unit_price) / np.sum(actual_m * 4.0)),
                    "month_within_6pct": float(np.mean(error <= 0.06)),
                    "month_within_8pct": float(np.mean(error <= 0.08)),
                    "month_signed_bias": float(np.mean((calibrated - actual_m) / capacity)),
                }
                month_rows.append(row)
                if best is None or row["score_if_only_this_month_changed"] > best["score_if_only_this_month_changed"]:
                    best = row

        month_df = pd.DataFrame(month_rows)
        sweep_rows.append(month_df)
        chosen_rows.append(best)

    return pd.DataFrame(chosen_rows), pd.concat(sweep_rows, ignore_index=True)


def apply_monthly(pred: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    capacity = CAPACITY_KWH[TARGET]
    for row in params.itertuples(index=False):
        month_mask = out.index.month == row.month
        out.loc[month_mask, TARGET] = np.clip(
            out.loc[month_mask, TARGET].to_numpy(dtype=float) * row.scale + row.bias,
            0.0,
            capacity,
        )
    return out


def monthly_metrics(answer: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, month_answer in answer.groupby(answer.index.month):
        idx = month_answer.index
        for version, pred in [("before", before), ("after", after)]:
            rows.append(
                {
                    "month": int(month),
                    "version": version,
                    **group_metric(
                        month_answer[TARGET].to_numpy(dtype=float),
                        pred.loc[idx, TARGET].to_numpy(dtype=float),
                        CAPACITY_KWH[TARGET],
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_params(params: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(params["month"], params["scale"], color="#4c78a8")
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].set_ylabel("Scale")
    axes[0].set_title("Group 3 Monthly Calibration Scale")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(params["month"], params["bias"], color="#f58518")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Bias kWh")
    axes[1].set_xlabel("Month")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("Group 3 Monthly Calibration Bias")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_monthly_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    months = sorted(metrics["month"].unique())
    before = metrics[metrics["version"] == "before"].set_index("month").loc[months]
    after = metrics[metrics["version"] == "after"].set_index("month").loc[months]
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


def plot_scatter(answer: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> None:
    mask = answer[TARGET] >= CAPACITY_KWH[TARGET] * 0.10
    actual = answer.loc[mask, TARGET].to_numpy(dtype=float)
    cap = CAPACITY_KWH[TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, pred_df, title in [(axes[0], before, "Before"), (axes[1], after, "After monthly calibration")]:
        pred = pred_df.loc[mask, TARGET].to_numpy(dtype=float)
        ax.scatter(actual, pred, s=8, alpha=0.35, edgecolors="none")
        x = np.linspace(0, cap, 200)
        ax.plot(x, x, color="black", linewidth=1.2)
        ax.plot(x, x + cap * 0.06, color="#2ca02c", linestyle="--", linewidth=0.9)
        ax.plot(x, x - cap * 0.06, color="#2ca02c", linestyle="--", linewidth=0.9)
        ax.plot(x, x + cap * 0.08, color="#ff7f0e", linestyle=":", linewidth=0.9)
        ax.plot(x, x - cap * 0.08, color="#ff7f0e", linestyle=":", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Actual kWh")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Predicted kWh")
    fig.suptitle("Group 3 Before vs Monthly Calibration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_daily(answer: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> None:
    daily = pd.DataFrame({"actual": answer[TARGET], "before": before[TARGET], "after": after[TARGET]}).resample("D").mean()
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


def write_html(out_dir: Path, before: dict, after: dict, params: pd.DataFrame) -> None:
    def fmt(x: float) -> str:
        return f"{x:.6f}"

    before_g3 = next(row for row in before["groups"] if row["target"] == TARGET)
    after_g3 = next(row for row in after["groups"] if row["target"] == TARGET)
    param_rows = "\n".join(
        f"<tr><td>{int(r.month)}</td><td>{r.scale:.3f}</td><td>{r.bias:.1f}</td></tr>"
        for r in params.itertuples(index=False)
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Group 3 Monthly Calibration</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #222; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f2f2f2; }}
    img {{ max-width: 100%; margin: 12px 0 28px; border: 1px solid #ddd; }}
    .warn {{ color: #8a5a00; }}
  </style>
</head>
<body>
  <h1>Group 3 Monthly Calibration</h1>
  <p class="warn">Monthly calibration can overfit validation months. Treat this as a diagnostic and use conservative checks before final submission.</p>
  <h2>Before vs After</h2>
  <table>
    <tr><th>metric</th><th>before</th><th>after</th><th>delta</th></tr>
    <tr><td>Total score</td><td>{fmt(before["score"])}</td><td>{fmt(after["score"])}</td><td>{fmt(after["score"] - before["score"])}</td></tr>
    <tr><td>1-NMAE</td><td>{fmt(before["one_minus_nmae"])}</td><td>{fmt(after["one_minus_nmae"])}</td><td>{fmt(after["one_minus_nmae"] - before["one_minus_nmae"])}</td></tr>
    <tr><td>FICR</td><td>{fmt(before["ficr"])}</td><td>{fmt(after["ficr"])}</td><td>{fmt(after["ficr"] - before["ficr"])}</td></tr>
    <tr><td>Group 3 NMAE</td><td>{fmt(before_g3["nmae"])}</td><td>{fmt(after_g3["nmae"])}</td><td>{fmt(after_g3["nmae"] - before_g3["nmae"])}</td></tr>
    <tr><td>Group 3 FICR</td><td>{fmt(before_g3["ficr"])}</td><td>{fmt(after_g3["ficr"])}</td><td>{fmt(after_g3["ficr"] - before_g3["ficr"])}</td></tr>
  </table>
  <h2>Monthly Parameters</h2>
  <table><tr><th>month</th><th>scale</th><th>bias</th></tr>{param_rows}</table>
  <h2>Visuals</h2>
  <img src="group3_monthly_params.png" alt="monthly parameters">
  <img src="group3_monthly_metrics.png" alt="monthly metrics">
  <img src="group3_monthly_scatter.png" alt="scatter">
  <img src="group3_monthly_daily_timeseries.png" alt="daily timeseries">
</body>
</html>
"""
    (out_dir / "group3_monthly_calibration_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    pred = pred.set_index("forecast_kst_dtm").sort_index()
    answer = load_answer(args.processed_dir, args.valid_start, pred.index)

    before_metrics = competition_metric(answer, pred[TARGET_COLS])
    params, sweep = tune_monthly(answer, pred, args)
    calibrated = apply_monthly(pred, params)
    after_metrics = competition_metric(answer, calibrated[TARGET_COLS])
    monthly = monthly_metrics(answer, pred, calibrated)

    params.to_csv(args.out_dir / "group3_monthly_calibration_params.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(args.out_dir / "group3_monthly_calibration_sweep.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(args.out_dir / "group3_monthly_metrics.csv", index=False, encoding="utf-8-sig")
    calibrated.reset_index().to_csv(
        args.out_dir / "validation_lgbm_fast_weighted_group3_monthly_calibrated_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "target": TARGET,
        "before": before_metrics,
        "after": after_metrics,
        "params": params[["month", "scale", "bias"]].to_dict(orient="records"),
    }
    with open(args.out_dir / "group3_monthly_calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_params(params, args.out_dir / "group3_monthly_params.png")
    plot_monthly_metrics(monthly, args.out_dir / "group3_monthly_metrics.png")
    plot_scatter(answer, pred, calibrated, args.out_dir / "group3_monthly_scatter.png")
    plot_daily(answer, pred, calibrated, args.out_dir / "group3_monthly_daily_timeseries.png")
    write_html(args.out_dir, before_metrics, after_metrics, params)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report: {(args.out_dir / 'group3_monthly_calibration_report.html').resolve()}")


if __name__ == "__main__":
    main()
