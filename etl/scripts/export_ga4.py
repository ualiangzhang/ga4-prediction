#!/usr/bin/env python3
"""
Enhanced ETL: export classification features + daily & hourly page-view series.

Raw CSV outputs go to: etl/data/raw/
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import NoReturn

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import Client

# ────────────────────── Paths ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # etl/
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SQL_DIR = ROOT / "queries"

# Map of dataset name → (SQL filename, output CSV path)
FILES = {
    "user_features": ("user_features.sql", RAW_DIR / "ga4_user_features.csv"),
    "daily_series": ("daily_pageviews.sql", RAW_DIR / "daily_pageviews.csv"),
    "hourly_series": ("hourly_pageviews.sql", RAW_DIR / "hourly_pageviews.csv"),
}


# ────────────────────── Helpers ────────────────────────────────────────────────
def run_query(client: Client, sql_path: Path, out_csv: Path) -> pd.DataFrame:
    """
    Execute a SQL file against BigQuery and save the DataFrame to CSV.

    Args:
        client  (Client): Authenticated BigQuery client.
        sql_path(Path)  : Path to the .sql query file.
        out_csv (Path)  : Destination CSV path.

    Returns:
        pd.DataFrame: The query results.
    """
    with sql_path.open("r", encoding="utf-8") as f:
        query: str = f.read()

    job = client.query(query)
    df: pd.DataFrame = job.to_dataframe()  # type: ignore

    df.to_csv(out_csv, index=False)
    print(f"✅  {out_csv.name:<22} rows={len(df):,}")
    return df


# ────────────────────── Main ──────────────────────────────────────────────────
def main() -> NoReturn:
    # 1. Check auth
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("🔴 Set GOOGLE_APPLICATION_CREDENTIALS to your service-account JSON key")

    # 2. Determine GCP project
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or input("Enter GCP project ID: ")
    client: Client = bigquery.Client(project=project_id)

    # 3. Run exports
    for name, (sql_fname, csv_path) in FILES.items():
        sql_path = SQL_DIR / sql_fname
        run_query(client, sql_path, csv_path)

    print(f"\n🎉 All raw exports saved to: {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
