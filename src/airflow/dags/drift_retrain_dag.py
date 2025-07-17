# src/airflow/dags/drift_retrain_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to Python path dynamically
# Dynamically find the path to the 'lead_scoring_project' directory
CURRENT_FILE_PATH = os.path.abspath(__file__)
while True:
    CURRENT_FILE_PATH = os.path.dirname(CURRENT_FILE_PATH)
    if os.path.exists(os.path.join(CURRENT_FILE_PATH, 'src')):
        break

if CURRENT_FILE_PATH not in sys.path:
    sys.path.insert(0, CURRENT_FILE_PATH)


from src.airflow.scripts.trigger_upload_monitor import has_new_upload
from src.airflow.scripts.check_drift_runner import run_drift_check
from src.airflow.scripts.retrain_runner import run_retrain


default_args = {
    "owner": "VenkatSai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="drift_and_retrain",
    default_args=default_args,
    description="Detect data drift and retrain when needed",
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["lead_scoring", "drift"],
) as dag:


    check_drift = PythonOperator(
        task_id="perform_drift_check",
        python_callable=run_drift_check,
    )

    branch_retrain = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=lambda drift: "retrain_model" if drift else "end",
        op_args=["{{ ti.xcom_pull(task_ids='perform_drift_check') }}"],
    )

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=run_retrain,
    )

    end = EmptyOperator(task_id="end")

    check_drift >> branch_retrain
    branch_retrain >> [retrain, end]
    retrain >> end
