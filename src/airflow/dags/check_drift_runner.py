

# ==== CONFIGURATION ====
DB_USER="admin"
DB_PASSWORD="Venky643!!"
DB_HOST="capstoneworkgroup.731239205085.ap-south-1.redshift-serverless.amazonaws.com"
DB_PORT="5439"
DB_NAME="dev"
#DB_TABLE="lead_data"
#DB_TEST_TABLE_UPLOADED="test_lead_data"
#DB_TABLE_PREPROCESSED="preprocessed_train_data"
#DB_NEW_TABLE_PREPROECESSED="user_uploaded_preprocessed"
MLFLOW_TRACKING_URI="http://13.232.137.190:5000"
S3_BUCKET="capstonedataminfy"
REF_TABLE    = "preprocessed_train_data"
NEW_TABLE    = "user_uploaded_preprocessed"
DATASET_NAME = "lead_data_vs_uploaded"
import os
import json
import sys
import mlflow
from airflow.exceptions import AirflowException
from datetime import datetime
import re

from dotenv import load_dotenv
load_dotenv()

from airflow_loader import load_data

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def _sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\. :/]", "_", name)


def check_drift(train_df, test_df, dataset_name="train_vs_test", log_to_mlflow=True):
    common_cols = sorted(set(train_df.columns) & set(test_df.columns))
    if not common_cols:
        print(f"⚠️ No common columns found; skipping drift check for {dataset_name}")
        return

    ref = train_df[common_cols].copy()
    cur = test_df[common_cols].copy()

    # ✅ Coerce current data types to match reference data types
    for col in common_cols:
        ref_dtype = ref[col].dtype
        try:
            cur[col] = cur[col].astype(ref_dtype)
        except Exception as e:
            print(f"⚠️ Could not cast column '{col}' to {ref_dtype}: {e}")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    if log_to_mlflow:
        # Save HTML and JSON temporarily for logging
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_temp_path = f"/tmp/drift_{dataset_name}_{ts}.html"
        json_temp_path = f"/tmp/drift_{dataset_name}_{ts}.json"

        report.save_html(html_temp_path)
        report.save_json(json_temp_path)

        mlflow.log_artifact(html_temp_path, artifact_path=f"drift/{dataset_name}")
        mlflow.log_artifact(json_temp_path, artifact_path=f"drift/{dataset_name}")
        print(f"📦 MLflow: Logged artifacts under drift/{dataset_name}")

        # Load metrics from JSON
        with open(json_temp_path, "r") as f:
            report_json = json.load(f)

        for m in report_json.get("metrics", []):
            val = m.get("value", None)
            metric_id = m.get("metric_id") or m.get("metric") or ""

            if val is None and isinstance(m.get("result"), dict):
                for k in (
                    "drift_score", "mean", "mean_reference", "mean_current",
                    "number_of_rows", "number_of_columns",
                    "number_of_drifted_columns", "share_of_drifted_columns"
                ):
                    if k in m["result"]:
                        val = m["result"][k]
                        metric_id = f"{metric_id}_{k}"
                        break

            if isinstance(val, (int, float)):
                safe_name = f"{dataset_name}__{_sanitize_metric_name(metric_id)}"
                mlflow.log_metric(safe_name, float(val))
                print(f"📊 MLflow metric: {safe_name} = {val}")

        print(f"✅ Logged numeric metrics for `{dataset_name}`")

    return report


def run_drift_check():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise AirflowException("MLFLOW_TRACKING_URI is not set")
    mlflow.set_tracking_uri(mlflow_uri)

    ref_table = os.getenv("DB_TABLE_PREPROCESSED", "preprocessed_train_data")
    new_table = os.getenv("DB_NEW_TABLE_PREPROCESSED", "user_uploaded_preprocessed")

    try:
        ref_df = load_data(ref_table)
        new_df = load_data(new_table)
    except Exception as e:
        raise AirflowException(f"Failed to load tables: {e}")

    result = check_drift(
        train_df=ref_df,
        test_df=new_df,
        dataset_name="lead_data_vs_uploaded",
        log_to_mlflow=True
    )

    raw_json = result.json()
    try:
        report_json = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise AirflowException(f"Invalid JSON from result.json(): {e}")

    drift_flag = False
    for m in report_json.get("metrics", []):
        res = m.get("result", {})
        if "dataset_drift" in res:
            drift_flag = bool(res["dataset_drift"])
            print(f"🔍 [DEBUG] Found dataset_drift={drift_flag} in metric {m.get('metric_name')}")
            break

    return drift_flag


if __name__ == "__main__":
    flag = run_drift_check()
    print(f"Dataset drift detected: {flag}")
