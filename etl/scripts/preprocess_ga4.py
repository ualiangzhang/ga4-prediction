#!/usr/bin/env python3
# File: etl/scripts/preprocess_ga4.py

"""
Preprocess raw GA4 exports into training-ready datasets.

This script reads three raw CSVs in etl/data/raw/:
  - ga4_user_features.csv   (classification features)
  - daily_pageviews.csv     (daily page-view counts)
  - hourly_pageviews.csv    (hourly page-view counts)

Outputs cleaned CSVs to etl/data/processed/:
  - ga4_training_data.csv
  - daily_pageviews.csv
  - hourly_pageviews.csv

Dependencies:
    pip install pandas numpy
"""

from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd

# ─── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent       # etl/
RAW_DIR  = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─── Classification Preprocessing ─────────────────────────────────────────────
def preprocess_classification(
    raw_csv: Path,
    out_csv: Path
) -> pd.DataFrame:
    """
    Load raw user features, clean and export classification dataset.

    Args:
        raw_csv (Path): Path to raw 'ga4_user_features.csv'.
        out_csv (Path): Path to write 'ga4_training_data.csv'.

    Returns:
        pd.DataFrame: Processed classification DataFrame.
    """
    df = pd.read_csv(raw_csv)

    # Numeric features: fill missing with 0 and convert to int
    numeric_cols = [
        "num_pageviews", "num_addtocart", "num_sessions",
        "total_engagement_ms", "active_days", "recency_days"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    # Categorical features: fill missing and strip whitespace
    cat_cols = ["device_type", "traffic_source", "country"]
    for col in cat_cols:
        df[col] = df.get(col, "unknown").fillna("unknown").astype(str).str.strip()

    # Label: ensure binary integer
    df["converted"] = df.get("converted", False).astype(bool).astype(int)

    # Save final classification set
    df.to_csv(out_csv, index=False)
    print(f"✅ Written classification data: {out_csv.name} (rows={len(df)})")
    return df


# ─── Daily Time-Series Preprocessing ──────────────────────────────────────────
def preprocess_daily_series(
    raw_csv: Path,
    out_csv: Path
) -> pd.DataFrame:
    """
    Load raw daily page views, reindex to continuous dates, and export.

    Args:
        raw_csv (Path): Path to raw 'daily_pageviews.csv'.
        out_csv (Path): Path to write cleaned 'daily_pageviews.csv'.

    Returns:
        pd.DataFrame: Processed daily time series.
    """
    df = pd.read_csv(raw_csv, parse_dates=["ds"])
    df = df[["ds", "y"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)

    # Continuous daily index
    df = df.set_index("ds").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx, fill_value=0)
    df.index.name = "ds"
    df = df.reset_index()

    df.to_csv(out_csv, index=False)
    print(f"✅ Written daily series: {out_csv.name} (rows={len(df)})")
    return df


# ─── Hourly Time-Series Preprocessing ─────────────────────────────────────────
def preprocess_hourly_series(
    raw_csv: Path,
    out_csv: Path
) -> pd.DataFrame:
    """
    Load raw hourly page views, reindex to continuous hours, and export.

    Args:
        raw_csv (Path): Path to raw 'hourly_pageviews.csv'.
        out_csv (Path): Path to write cleaned 'hourly_pageviews.csv'.

    Returns:
        pd.DataFrame: Processed hourly time series.
    """
    df = pd.read_csv(raw_csv, parse_dates=["ds"])
    df = df[["ds", "y"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)

    # Continuous hourly index
    df = df.set_index("ds").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="H")
    df = df.reindex(full_idx, fill_value=0)
    df.index.name = "ds"
    df = df.reset_index()

    df.to_csv(out_csv, index=False)
    print(f"✅ Written hourly series: {out_csv.name} (rows={len(df)})")
    return df


# ─── Main Routine ─────────────────────────────────────────────────────────────
def main() -> NoReturn:
    """
    Execute preprocessing for classification, daily and hourly series.
    """
    # Define paths
    class_raw   = RAW_DIR / "ga4_user_features.csv"
    class_out   = PROC_DIR / "ga4_training_data.csv"

    daily_raw   = RAW_DIR / "daily_pageviews.csv"
    daily_out   = PROC_DIR / "daily_pageviews.csv"

    hourly_raw  = RAW_DIR / "hourly_pageviews.csv"
    hourly_out  = PROC_DIR / "hourly_pageviews.csv"

    # Run each preprocessing step
    preprocess_classification(class_raw, class_out)
    preprocess_daily_series(daily_raw, daily_out)
    preprocess_hourly_series(hourly_raw, hourly_out)

    print(f"\nAll processed files saved to: {PROC_DIR.resolve()}")


if __name__ == "__main__":
    main()
