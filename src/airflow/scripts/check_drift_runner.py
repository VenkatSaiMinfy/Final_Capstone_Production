# src/airflow/scripts/check_drift_runner.py

import os
import json
import sys
import mlflow
from airflow.exceptions import AirflowException
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from dotenv import load_dotenv
load_dotenv()
from src.airflow.utils.airflow_loader import load_data
from src.drift.check_drift import check_drift

def run_drift_check():
    """
    Loads reference & new preprocessed data, runs Evidently drift check,
    logs HTML/JSON reports and metrics to MLflow, and returns whether drift
    was detected.
    """
    # 1) Configure MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise AirflowException("MLFLOW_TRACKING_URI is not set")
    mlflow.set_tracking_uri(mlflow_uri)

    # 2) Table names
    ref_table = os.getenv("DB_TABLE_PREPROCESSED", "preprocessed_train_data")
    new_table = os.getenv("DB_NEW_TABLE_PREPROCESSED", "user_uploaded_preprocessed")

    # 3) Load preprocessed tables
    try:
        ref_df = load_data(ref_table)
        new_df = load_data(new_table)
    except Exception as e:
        raise AirflowException(f"Failed to load tables: {e}")

    # 4) Run drift check
    result = check_drift(
        train_df      = ref_df,
        test_df       = new_df,
        dataset_name  = "lead_data_vs_uploaded",
        save_report   = True,
        log_to_mlflow = True
    )

    # 5) Grab the raw JSON string
    raw_json = result.json()
    print("🔍 [DEBUG] result.json() output:")
    print(raw_json)  # make sure it's valid JSON

    # 6) Parse JSON to dict
    try:
        report_json = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise AirflowException(f"Invalid JSON from result.json(): {e}")

    # 7) Inspect metrics list
    metrics_list = report_json.get("metrics", [])
    print("🔍 [DEBUG] Parsed metrics list:")
    print(json.dumps(metrics_list, indent=2))

    # 8) Find any dataset_drift flag
    drift_flag = False
    for m in metrics_list:
        res = m.get("result", {})
        if "dataset_drift" in res:
            drift_flag = bool(res["dataset_drift"])
            print(f"🔍 [DEBUG] Found dataset_drift={drift_flag} in metric {m.get('metric_name')}")
            break

    return drift_flag

if __name__ == "__main__":
    flag = run_drift_check()
    print(f"Dataset drift detected: {flag}")
