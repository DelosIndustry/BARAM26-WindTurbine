from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply simple post-processing calibration to a submission file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=str, default="kpx_group_3")
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--bias", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.target not in submission.columns:
        raise ValueError(f"Target column not found: {args.target}")

    capacity = CAPACITY_KWH[args.target]
    before = submission[args.target].copy()
    submission[args.target] = np.clip(before.to_numpy(dtype=float) * args.scale + args.bias, 0.0, capacity)
    submission.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "target": args.target,
        "scale": args.scale,
        "bias": args.bias,
        "capacity": capacity,
        "before": {
            "min": float(before.min()),
            "mean": float(before.mean()),
            "max": float(before.max()),
        },
        "after": {
            "min": float(submission[args.target].min()),
            "mean": float(submission[args.target].mean()),
            "max": float(submission[args.target].max()),
        },
    }
    with open(args.output.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
