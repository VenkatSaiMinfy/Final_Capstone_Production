# src/drift/check_drift.py

import os
import json
import re
from pathlib import Path
from datetime import datetime

import mlflow
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Ensure output directory exists
REPORT_DIR = Path("reports/drift")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _sanitize_metric_name(name: str) -> str:
    """
    Replace characters not allowed in MLflow metric names with underscores.
    Allowed: alphanumerics, _, -, ., space, :, /
    """
    return re.sub(r"[^A-Za-z0-9_\-\. :/]", "_", name)

def check_drift(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                dataset_name: str = "train_vs_test",
                save_report: bool = True,
                log_to_mlflow: bool = True):
    """
    Runs Evidently DataDrift + DataSummary, saves HTML/JSON,
    and logs artifacts & sanitized metrics to MLflow.
    """

    common_cols = sorted(set(train_df.columns) & set(test_df.columns))
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping.")
        return

    ref = train_df[common_cols].copy()
    cur = test_df[common_cols].copy()

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORT_DIR / f"drift_{dataset_name}_{ts}.html"
    json_path = REPORT_DIR / f"drift_{dataset_name}_{ts}.json"

    if save_report:
        result.save_html(str(html_path))
        result.save_json(str(json_path))
        print(f"✅ Drift HTML saved to {html_path}")
        print(f"📦 Drift JSON saved to {json_path}")

    if log_to_mlflow:
        mlflow.log_artifact(str(html_path), artifact_path=f"drift/{dataset_name}")
        mlflow.log_artifact(str(json_path), artifact_path=f"drift/{dataset_name}")
        print(f"✅ Logged artifacts under drift/{dataset_name}")

        # Parse metrics and log them
        report_json = json.loads(result.json())
        for m in report_json.get("metrics", []):
            # Try top-level value
            val = m.get("value", None)
            metric_id = m.get("metric_id") or m.get("metric") or m.get("metric_name", "")
            # If nested in result, pull common numeric keys
            if val is None and isinstance(m.get("result"), dict):
                for k in ("drift_score","mean","mean_reference","mean_current",
                          "number_of_rows","number_of_columns",
                          "number_of_drifted_columns","share_of_drifted_columns"):
                    if k in m["result"]:
                        val = m["result"][k]
                        metric_id = f"{metric_id}_{k}"
                        break

            if isinstance(val, (int, float)):
                safe_name = f"{dataset_name}__{_sanitize_metric_name(metric_id)}"
                mlflow.log_metric(safe_name, float(val))
                print(f"🔢 {safe_name} = {val}")

        print(f"✅ Logged all numeric drift & summary metrics for `{dataset_name}`")

    return result
