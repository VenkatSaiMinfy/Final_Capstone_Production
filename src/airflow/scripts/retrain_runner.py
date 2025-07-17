# src/airflow/scripts/retrain_runner.py

import os
from src.ml.pipeline.pipeline_runner import run_pipeline

def run_retrain():
    """
    Calls your pipeline_runner to retrain, save, and register to MLflow.
    """
    # Ensure MLflow tracking URI is set
    os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI", ""))

    # Retrain on the entire reference table
    run_pipeline(
        table_name=os.getenv("REFERENCE_TABLE", "lead_data"),
        target_col="Converted",
        save=True,
        register=True
    )
    return True
