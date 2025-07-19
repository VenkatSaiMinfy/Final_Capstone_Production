import os
import sys
import joblib
import pandas as pd
from datetime import datetime
import mlflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.ml.data_loader.data_loader import load_data_from_postgres
from src.ml.pipeline.preprocessing import clean_columns, get_full_pipeline
from src.ml.pipeline.feature_selection import apply_feature_selection
from src.ml.pipeline.feature_selector import FeatureSelector
from src.ml.registry.model_registry import register_and_promote
from sklearn.pipeline import Pipeline
from src.ml.data_loader.data_loader import save_dataframe_to_postgres


def print_time(step, t0):
    print(f"[⏱️] {step} finished in {(datetime.now()-t0).total_seconds():.2f}s")
    return datetime.now()

def run_pipeline(
    table_name: str = "lead_data",
    target_col: str = "converted",
    save: bool = True,
    register: bool = False,
    return_pipeline: bool = False
):
    t0 = datetime.now()

    # 1. Load and Clean Data
    df = load_data_from_postgres(table_name)
    print(f"[INFO] Loaded data from '{table_name}', shape: {df.shape}")
    df = clean_columns(df)
    t0 = print_time("Data load & clean", t0)

    # 2. Separate Features and Target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3. Define Feature Columns Early
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    t0 = print_time("Feature type definition", t0)

    # 4. Build and Fit Initial Pipeline (Feature Eng + Preprocessing)
    full_pipeline = get_full_pipeline(numeric_features, categorical_features)
    X_transformed = full_pipeline.fit_transform(X, y)
    t0 = print_time("Pipeline fit & transform", t0)

    # 5. Apply Feature Selection (fit and transform on array, not DataFrame)
    X_selected, selected_indices = apply_feature_selection(X_transformed, y)
    t0 = print_time("Feature selection", t0)

    # 6. Build final pipeline (reuse already fitted objects)
    final_pipeline = Pipeline([
        ("feature_engineering", full_pipeline.named_steps["feature_engineering"]),
        ("preprocessing",     full_pipeline.named_steps["preprocessing"]),
        ("feature_selection", FeatureSelector(selected_features=selected_indices)),
    ])
    # fit not required here if all steps use selectors (for inference); fit if needed
    final_pipeline.fit(X, y)
    t0 = print_time("Final pipeline construction", t0)

    # 7. Save pipeline and transformed features efficiently
    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_pipeline, "models/full_pipeline.pkl", compress=3)
        print("✅ Full preprocessing pipeline (with feature selection) saved at 'models/full_pipeline.pkl'")

        # Save only the preprocessed features (no labels) to Postgres
        df_pre = pd.DataFrame(X_selected, columns=[f"f_{i}" for i in selected_indices])
        # Consider batch saves if df_pre is very large
        save_dataframe_to_postgres(df_pre, key="preprocessed_train_data.csv")
        print("[INFO] Preprocessed features DataFrame ready.")
        t0 = print_time("Saving pipeline/features", t0)

    # 8. Register to MLflow
    if register:
        try:
            register_and_promote(
                registry_name="LeadScoringPreprocessor",
                model_object=final_pipeline,
                is_pipeline=True
            )
            print("✅ Pipeline registered in MLflow as 'LeadScoringPreprocessor'")
        except Exception as e:
            print(f"❌ Failed to register/promote pipeline: {e}")
        t0 = print_time("MLflow registration", t0)

    # 9. Return for Upstream Use
    if return_pipeline:
        return X_selected, y, final_pipeline

    return X_selected, y


if __name__ == "__main__":
    run_pipeline(save=True, register=True)
