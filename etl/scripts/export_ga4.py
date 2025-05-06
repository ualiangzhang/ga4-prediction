# File: etl/scripts/export_ga4.py
"""
Exports user-level GA4 features from BigQuery to a local CSV file.

Pre-requisites:
  1. export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"
  2. export GOOGLE_CLOUD_PROJECT="your-real-project-id"

Run:
  python export_ga4.py
"""

import os
from typing import NoReturn

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import Client


def export_user_features(
    client: Client,
    sql_query: str,
    output_path: str
) -> pd.DataFrame:
    """
    Execute the given SQL query against BigQuery and save the results to CSV.

    Args:
        client (Client): BigQuery client initialized with a valid project.
        sql_query (str): Query string to run in BigQuery.
        output_path (str): Local file path for saving the CSV output.

    Returns:
        pd.DataFrame: The query results as a pandas DataFrame.
    """
    job = client.query(sql_query)
    df: pd.DataFrame = job.to_dataframe()  # type: ignore

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Data exported to {output_path} (rows: {df.shape[0]}, cols: {df.shape[1]})")
    return df


def main() -> NoReturn:
    """
    Main entry point for the ETL script.
    Ensures required environment variables are set, then runs the export.
    """
    # 1. Credentials for BigQuery
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError(
            "🔴 Please set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON key."
        )

    # 2. GCP Project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise EnvironmentError(
            "🔴 Please set GOOGLE_CLOUD_PROJECT to your GCP project ID."
        )

    # 3. Initialize BigQuery client with the correct project
    client: Client = bigquery.Client(project=project_id)

    # 4. Load the SQL query
    sql_path = os.path.join(
        os.path.dirname(__file__),
        "..", "queries", "user_features.sql"
    )
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_query: str = f.read()

    # 5. Define output path
    output_csv = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "processed", "ga4_training_data.csv"
    )

    # 6. Run export
    export_user_features(client, sql_query, output_csv)


if __name__ == "__main__":
    main()
