# src/ml/training/train.py (final and corrected)

import mlflow
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.ml.training.train_utils import get_models_with_params, train_and_log_model
from src.ml.pipeline.pipeline_runner import run_pipeline
from src.ml.registry.model_registry import register_and_promote

def train_all_models():
    # Load and preprocess data
    X_processed, y = run_pipeline(save=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    models_with_params = get_models_with_params()

    # Set and create experiment if not exists
    mlflow.set_tracking_uri("http://localhost:5000")  # ← Set this only if using remote tracking
    mlflow.set_experiment("Lead Scoring Model")

    best_f1 = -1.0
    best_info = (None, None)

    # Start a single parent run for all models
    with mlflow.start_run(run_name="All_Model_Training_Run") as parent_run:
        print(f"[INFO] Parent run ID: {parent_run.info.run_id}")

        for name, (model, param_grid) in models_with_params.items():
            # Pass nested=True in train_utils
            model_name, run_id, f1 = train_and_log_model(
                name, model, param_grid,
                X_train, X_test, y_train, y_test
            )

            if f1 > best_f1:
                best_f1 = f1
                best_info = (model_name, run_id)

        # Register and promote the best model
        best_name, best_run_id = best_info
        if best_name and best_run_id:
            print(f"\n🏆 Best model is '{best_name}' with F1 = {best_f1:.4f}")
            model_uri = f"runs:/{best_run_id}/{best_name}"
            register_and_promote(
                registry_name="LeadScoringBestModel",
                run_id=best_run_id,
                model_uri=model_uri,
                is_pipeline=False
            )
        else:
            print("❌ No valid model trained to register.")

if __name__ == "__main__":
    train_all_models()