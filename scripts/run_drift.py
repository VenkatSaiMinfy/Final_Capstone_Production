# scripts/run_drift.py

import os
import sys
from datetime import datetime

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import pandas as pd
from sklearn.model_selection import train_test_split

# make sure you can import your src/ modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.data_loader.data_loader import load_data_from_postgres
from src.ml.pipeline.preprocessing import clean_columns
from src.ml.pipeline.feature_engineering import feature_engineering
from src.drift.check_drift import check_drift

if __name__ == "__main__":
    # 1️⃣ Load + preprocess
    df = load_data_from_postgres("lead_data")
    df = clean_columns(df)
    df = feature_engineering(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2️⃣ Ensure experiment exists (or restore if deleted)
    exp_name = "Lead Scoring Drift Checks"
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        mlflow.create_experiment(exp_name)
        print(f"ℹ️ Created experiment '{exp_name}'")
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        print(f"ℹ️ Restored deleted experiment '{exp_name}'")

    # 3️⃣ Set active experiment
    mlflow.set_experiment(exp_name)

    # 4️⃣ Run drift check inside a run
    with mlflow.start_run(run_name="train_test_drift_check"):
        print("📊 Checking drift between train and test datasets...")
        check_drift(
            train_df=train_df,
            test_df=test_df,
            dataset_name="train_vs_test",
            save_report=True,
            log_to_mlflow=True
        )
