# src/ml/training/train.py

import os
import sys
from datetime import datetime
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.drift.check_drift import check_drift


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
load_dotenv()

from src.ml.pipeline.pipeline_runner import run_pipeline
from src.ml.training.train_utils     import get_models_with_params, train_and_log_model
from src.ml.registry.model_registry   import register_and_promote
from src.ml.eda.profiler import generate_eda_report

def train_all_models():
    generate_eda_report()
    
    # 1) Preprocess + feature‑select + get pipeline
    X_sel, y, final_pipeline = run_pipeline(
        save=True,
        register=True,
        return_pipeline=True
    )

    # 2) Extract column names & indices
    preproc = final_pipeline.named_steps["preprocessing"]
    all_names = preproc.get_feature_names_out()
    sel_idxs  = final_pipeline.named_steps["feature_selection"].selected_features
    feat_names = all_names[sel_idxs]

    # 3) Build DataFrame
    df = pd.DataFrame(X_sel, columns=feat_names)

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2,
        random_state=42, stratify=y
    )

    # 5) Configure MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", None))
    mlflow.set_experiment("Lead Scoring Model")

    best_f1 = -1.0
    best_info = (None, None)

    # 6) Parent run
    with mlflow.start_run(run_name="All_Model_Training_Run") as parent:
        print(f"[INFO] Parent run ID: {parent.info.run_id}")
        check_drift(
            train_df=X_train,
            test_df=X_test,
            dataset_name="train_vs_test_drift",
            save_report=True,
            log_to_mlflow=True
        )

        for name, (model, params) in get_models_with_params().items():
            mname, run_id, f1 = train_and_log_model(
                name=name,
                model=model,
                param_grid=params,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feat_names
            )

            if f1 > best_f1:
                best_f1 = f1
                best_info = (mname, run_id)

        # 7) Register best model
        best_name, best_run = best_info
        if best_name:
            print(f"\n🏆 Best model: {best_name} (F1={best_f1:.4f})")
            uri = f"runs:/{best_run}/{best_name}"
            register_and_promote(
                registry_name="LeadScoringBestModel",
                run_id=best_run,
                model_uri=uri,
                is_pipeline=False
            )
        else:
            print("❌ No successful model runs to register.")

    # End parent run implicitly on exit of with-block


if __name__ == "__main__":
    train_all_models()
