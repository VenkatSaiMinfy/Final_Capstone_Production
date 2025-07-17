# src/airflow/scripts/trigger_upload_monitor.py

import os
import pandas as pd
from datetime import datetime
from airflow.exceptions import AirflowSkipException
from src.airflow.utils.airflow_loader import load_data
from dotenv import load_dotenv

# Load environment variables from .env file (if not already loaded)
load_dotenv()

# File to track last run timestamp
LAST_RUN_PATH = os.getenv("LAST_RUN_FILE", "/tmp/last_upload_check.txt")

def has_new_upload():
    """
    Checks if the table has any rows newer than the last run.
    Requires an 'uploaded_at' timestamp column.
    """
    # Use DB_TABLE from environment or fallback to 'lead_data'
    table_name = os.getenv("DB_NEW_TABLE_PREPROECESSED")

    print(f"[INFO] Checking table: {table_name}")
    df = load_data(table_name)

    if "uploaded_at" not in df.columns:
        raise AirflowSkipException(f"[SKIP] No 'uploaded_at' column found in table: {table_name}")

    df["uploaded_at"] = pd.to_datetime(df["uploaded_at"])
    most_recent = df["uploaded_at"].max()

    print(f"[INFO] Latest upload timestamp in DB: {most_recent}")

    # Read the last checked timestamp
    try:
        with open(LAST_RUN_PATH, "r") as f:
            last_run = pd.to_datetime(f.read().strip())
            print(f"[INFO] Last checked timestamp: {last_run}")
    except FileNotFoundError:
        last_run = datetime.min
        print("[INFO] First run — no previous timestamp found.")

    # Compare timestamps
    if most_recent > last_run:
        print("[INFO] New data detected — proceeding to drift check.")
        # Save new latest timestamp
        with open(LAST_RUN_PATH, "w") as f:
            f.write(most_recent.isoformat())
        return True

    print("[SKIP] No new uploads detected since last check.")
    raise AirflowSkipException("No new data uploaded since last check.")
