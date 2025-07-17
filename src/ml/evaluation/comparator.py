import pandas as pd
import os
from mlflow.tracking import MlflowClient

def get_run_metrics(run_id, client):
    data = client.get_run(run_id).data
    return {**data.metrics, **data.params}

def compare_models(experiment_name="Lead Scoring Models"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1000)

    records = []
    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "model_name": run.data.tags.get("mlflow.runName", "unknown")
        }
        metrics_params = get_run_metrics(run.info.run_id, client)
        run_info.update(metrics_params)
        records.append(run_info)

    df = pd.DataFrame(records)
    
    # Convert numeric fields for sorting
    for col in ['accuracy', 'f1_score', 'roc_auc', 'cv_mean_test_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_sorted = df.sort_values(by="f1_score", ascending=False)
    return df_sorted.reset_index(drop=True)
