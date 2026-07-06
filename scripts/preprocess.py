from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}

LDAPS_GROUP_GRIDS = {
    "kpx_group_1": [5, 6, 10],
    "kpx_group_2": [6, 11],
    "kpx_group_3": [6, 12],
}

GFS_GROUP_GRIDS = {
    "kpx_group_1": [5],
    "kpx_group_2": [5],
    "kpx_group_3": [5],
}

VESTAS_RATED_KWH_10M = 600.0
UNISON_RATED_KWH_10M = 700.0


LDAPS_UV_PAIRS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind_10m"),
    ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "wind_50m_max"),
    ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "wind_50m_min"),
    ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "wind_5m_bl"),
]

GFS_UV_PAIRS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind_10m"),
    ("heightAboveGround_80_u", "heightAboveGround_80_v", "wind_80m"),
    ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "wind_100m"),
    ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "wind_pbl"),
    ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "wind_850hpa"),
    ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "wind_700hpa"),
    ("isobaricInhPa_500_u", "isobaricInhPa_500_v", "wind_500hpa"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BARAM 2026 data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed"))
    parser.add_argument("--no-wide-grids", action="store_true", help="Save only group-near aggregates, not all grid-wide columns.")
    return parser.parse_args()


def add_time_features(index: pd.Index) -> pd.DataFrame:
    dt = pd.to_datetime(index)
    out = pd.DataFrame(index=dt)
    out.index.name = "forecast_kst_dtm"
    out["year"] = dt.year.astype("int16")
    out["month"] = dt.month.astype("int8")
    out["day"] = dt.day.astype("int8")
    out["hour"] = dt.hour.astype("int8")
    out["dayofweek"] = dt.dayofweek.astype("int8")
    out["dayofyear"] = dt.dayofyear.astype("int16")
    out["is_weekend"] = (dt.dayofweek >= 5).astype("int8")

    for col, period in [("month", 12), ("hour", 24), ("dayofweek", 7), ("dayofyear", 366)]:
        values = out[col].astype(float)
        out[f"{col}_sin"] = np.sin(2 * np.pi * values / period).astype("float32")
        out[f"{col}_cos"] = np.cos(2 * np.pi * values / period).astype("float32")

    return out


def add_uv_features(df: pd.DataFrame, uv_pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    for u_col, v_col, prefix in uv_pairs:
        if u_col not in df.columns or v_col not in df.columns:
            continue

        u = df[u_col].to_numpy(dtype=float)
        v = df[v_col].to_numpy(dtype=float)
        ws = np.hypot(u, v)

        df[f"{prefix}_speed"] = ws
        df[f"{prefix}_speed2"] = ws**2
        df[f"{prefix}_speed3"] = ws**3
        df[f"{prefix}_u_unit"] = np.divide(u, ws, out=np.zeros_like(u), where=ws > 0)
        df[f"{prefix}_v_unit"] = np.divide(v, ws, out=np.zeros_like(v), where=ws > 0)

    if "surface_0_sp" in df.columns:
        temp_candidates = [
            "heightAboveGround_2_t",
            "heightAboveGround_2_2t",
            "isobaricInhPa_850_t",
        ]
        for temp_col in temp_candidates:
            if temp_col in df.columns:
                temp = df[temp_col].to_numpy(dtype=float)
                pressure = df["surface_0_sp"].to_numpy(dtype=float)
                df["air_density_proxy"] = np.divide(pressure, temp, out=np.full_like(pressure, np.nan), where=temp != 0)
                break

    return df


def weather_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
    }
    return [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_weather_features(
    path: Path,
    source: str,
    uv_pairs: list[tuple[str, str, str]],
    group_grids: dict[str, list[int]],
    include_wide_grids: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"])
    df["grid_id"] = df["grid_id"].astype(int)
    df = add_uv_features(df, uv_pairs)

    feature_cols = weather_feature_columns(df)

    lead = (
        df[["forecast_kst_dtm", "data_available_kst_dtm"]]
        .drop_duplicates("forecast_kst_dtm")
        .set_index("forecast_kst_dtm")
    )
    lead[f"{source}_lead_hours"] = (
        (lead.index.to_series() - lead["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    ).astype("float32")
    lead[f"{source}_available_hour"] = lead["data_available_kst_dtm"].dt.hour.astype("int8")
    lead = lead.drop(columns=["data_available_kst_dtm"])

    frames = [lead]

    if include_wide_grids:
        wide = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=feature_cols)
        wide.columns = [f"{source}_g{int(grid):02d}_{var}" for var, grid in wide.columns]
        wide = wide.sort_index(axis=1)
        frames.append(wide)

    for target_col, grids in group_grids.items():
        group_id = target_col.rsplit("_", 1)[-1]
        near = df[df["grid_id"].isin(grids)]
        agg = near.groupby("forecast_kst_dtm")[feature_cols].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{source}_grp{group_id}_near_{var}_{stat}" for var, stat in agg.columns]
        agg = agg.fillna(0.0)
        frames.append(agg)

    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "forecast_kst_dtm"
    return reduce_numeric_memory(out)


def reduce_numeric_memory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype("float32")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


def build_feature_table(data_dir: Path, split: str, include_wide_grids: bool) -> pd.DataFrame:
    ldaps = build_weather_features(
        data_dir / split / f"ldaps_{split}.csv",
        "ldaps",
        LDAPS_UV_PAIRS,
        LDAPS_GROUP_GRIDS,
        include_wide_grids,
    )
    gfs = build_weather_features(
        data_dir / split / f"gfs_{split}.csv",
        "gfs",
        GFS_UV_PAIRS,
        GFS_GROUP_GRIDS,
        include_wide_grids,
    )
    features = ldaps.join(gfs, how="inner")
    features = features.join(add_time_features(features.index), how="left")
    features = reduce_numeric_memory(features)
    features.index.name = "forecast_kst_dtm"
    return features


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index.name = "forecast_kst_dtm"

    for col in TARGET_COLS:
        capacity = CAPACITY_KWH[col]
        labels[f"{col}_norm"] = labels[col] / capacity
        labels[f"{col}_eval_valid"] = labels[col] >= capacity * 0.10

    return labels


def hourly_scada_power(df: pd.DataFrame, cols: list[str], upper: float) -> pd.Series:
    power = df[cols].apply(pd.to_numeric, errors="coerce").clip(lower=0.0, upper=upper)
    hourly = power.sum(axis=1, min_count=1).resample("1h", label="right", closed="right").sum(min_count=1)
    return hourly


def build_scada_hourly(data_dir: Path) -> pd.DataFrame:
    vestas = pd.read_csv(data_dir / "train" / "scada_vestas_train.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    unison = pd.read_csv(data_dir / "train" / "scada_unison_train.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    vestas = vestas.set_index("kst_dtm").sort_index()
    unison = unison.set_index("kst_dtm").sort_index()

    v_g1_cols = [f"vestas_wtg{i:02d}_power_kw10m" for i in range(1, 7)]
    v_g2_cols = [f"vestas_wtg{i:02d}_power_kw10m" for i in range(7, 13)]
    u_g3_cols = [f"unison_wtg{i:02d}_power_kw10m" for i in range(1, 6)]

    scada = pd.DataFrame(
        {
            "scada_kpx_group_1": hourly_scada_power(vestas, v_g1_cols, VESTAS_RATED_KWH_10M),
            "scada_kpx_group_2": hourly_scada_power(vestas, v_g2_cols, VESTAS_RATED_KWH_10M),
            "scada_kpx_group_3": hourly_scada_power(unison, u_g3_cols, UNISON_RATED_KWH_10M),
        }
    )
    scada.index.name = "forecast_kst_dtm"
    return reduce_numeric_memory(scada)


def add_scada_filled_targets(labels: pd.DataFrame, scada: pd.DataFrame) -> pd.DataFrame:
    out = labels.join(scada, how="left")

    for col in TARGET_COLS:
        scada_col = f"scada_{col}"
        filled_col = f"{col}_filled"
        flag_col = f"{col}_filled_from_scada"

        out[filled_col] = out[col]
        fill_mask = out[filled_col].isna() & out[scada_col].notna()
        out.loc[fill_mask, filled_col] = out.loc[fill_mask, scada_col]
        out[flag_col] = fill_mask.astype("int8")
        out[f"{filled_col}_norm"] = out[filled_col] / CAPACITY_KWH[col]
        out[f"{filled_col}_eval_valid"] = out[filled_col] >= CAPACITY_KWH[col] * 0.10

    return reduce_numeric_memory(out)


def save_pickle(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def write_summary(out_dir: Path, artifacts: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, df in artifacts.items():
        rows.append(
            {
                "artifact": name,
                "rows": len(df),
                "columns": len(df.columns),
                "start": df.index.min(),
                "end": df.index.max(),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "preprocess_summary.csv", index=False, encoding="utf-8-sig")

    feature_cols = pd.Series(artifacts["features_train"].columns, name="feature")
    feature_cols.to_csv(out_dir / "feature_columns.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    include_wide_grids = not args.no_wide_grids

    print("Building weather feature tables...")
    features_train = build_feature_table(data_dir, "train", include_wide_grids)
    features_test = build_feature_table(data_dir, "test", include_wide_grids)

    print("Building labels and SCADA tables...")
    labels = load_labels(data_dir / "train" / "train_labels.csv")
    scada_hourly = build_scada_hourly(data_dir)
    labels_processed = add_scada_filled_targets(labels, scada_hourly)

    print("Saving processed artifacts...")
    save_pickle(features_train, out_dir / "features_train.pkl")
    save_pickle(features_test, out_dir / "features_test.pkl")
    save_pickle(labels_processed, out_dir / "labels_processed.pkl")
    save_pickle(scada_hourly, out_dir / "scada_hourly.pkl")

    write_summary(
        out_dir,
        {
            "features_train": features_train,
            "features_test": features_test,
            "labels_processed": labels_processed,
            "scada_hourly": scada_hourly,
        },
    )

    print("Done.")
    print(f"features_train: {features_train.shape}")
    print(f"features_test : {features_test.shape}")
    print(f"labels        : {labels_processed.shape}")
    print(f"scada_hourly  : {scada_hourly.shape}")
    print(f"Output dir    : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
