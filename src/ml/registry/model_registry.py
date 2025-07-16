# src/ml/registry/model_registry.py

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from datetime import datetime


def _ensure_model_registered(client, registry_name: str):
    """Ensure the registered model exists in the MLflow registry."""
    try:
        client.get_registered_model(registry_name)
    except MlflowException as e:
        if "not found" in str(e).lower():
            print(f"ℹ️ Registered model '{registry_name}' not found. Creating it...")
            client.create_registered_model(registry_name)
        else:
            raise


def _promote_to_production(client, registry_name: str, version: str):
    """Promote the given model version to Production stage."""
    client.transition_model_version_stage(
        name=registry_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🚀 Model '{registry_name}' version {version} promoted to 'Production'")


def register_and_promote(
    registry_name: str,
    model_object=None,
    run_id: str = None,
    model_uri: str = None,
    is_pipeline: bool = False
):
    """
    Register and promote either a training model (using run_id + uri) or a fitted pipeline (sklearn object).
    
    Parameters:
        registry_name (str): Name of the model in MLflow registry.
        model_object (sklearn-compatible object): Preprocessor or model object to log.
        run_id (str): MLflow run ID (required for trained models).
        model_uri (str): Artifact URI for the model (e.g., 'runs:/.../model').
        is_pipeline (bool): Set True if logging preprocessor directly, False if logging trained model from run.
    """
    client = MlflowClient()
    _ensure_model_registered(client, registry_name)

    if is_pipeline:
        mlflow.set_experiment("Preprocessing Pipeline Registry")
        with mlflow.start_run(run_name=f"{registry_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            try:
                mlflow.sklearn.log_model(
                    sk_model=model_object,
                    artifact_path="preprocessor",
                    registered_model_name=registry_name
                )
                print(f"✅ Preprocessor registered as '{registry_name}'")
            except Exception as e:
                print(f"❌ Failed to log/register preprocessor: {e}")
                return

        # Promote latest "None" version to Production
        versions = client.get_latest_versions(registry_name, stages=["None"])
        if not versions:
            print("⚠️ No unpromoted versions found.")
            return

        latest_version = versions[0].version
        _promote_to_production(client, registry_name, latest_version)

    else:
        if not run_id or not model_uri:
            raise ValueError("run_id and model_uri must be provided for trained model registration.")

        try:
            mv = client.create_model_version(
                name=registry_name,
                source=model_uri,
                run_id=run_id
            )
        except MlflowException as e:
            if "not found" in str(e).lower():
                print(f"ℹ️ Model '{registry_name}' not found. Creating it...")
                client.create_registered_model(registry_name)
                mv = client.create_model_version(
                    name=registry_name,
                    source=model_uri,
                    run_id=run_id
                )
            else:
                raise

        _promote_to_production(client, registry_name, mv.version)
