import os
import sys
import joblib
import pandas as pd
from datetime import datetime

# Ensure absolute imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.ml.data_loader.data_loader import load_data_from_postgres
from src.ml.pipeline.preprocessing import clean_columns, get_full_pipeline
from src.ml.pipeline.feature_selection import apply_feature_selection
from src.ml.pipeline.feature_selector import FeatureSelector
from src.ml.registry.model_registry import register_and_promote
from sklearn.pipeline import Pipeline
from src.ml.data_loader.data_loader import save_dataframe_to_postgres



def run_pipeline(
    table_name: str = "lead_data",
    target_col: str = "Converted",
    save: bool = True,
    register: bool = False,
    return_pipeline: bool = False
):
    # ─────────────────────────────────────────────
    # 🧹 Step 1: Load and Clean Raw Data
    # ─────────────────────────────────────────────
    df = load_data_from_postgres(table_name)
    print(f"[INFO] Loaded data from '{table_name}', shape: {df.shape}")
    df = clean_columns(df)

    # ─────────────────────────────────────────────
    # 🎯 Step 2: Separate Features and Target
    # ─────────────────────────────────────────────
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # ─────────────────────────────────────────────
    # ⚙️ Step 3: Define Feature Columns
    # ─────────────────────────────────────────────
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ─────────────────────────────────────────────
    # 🔄 Step 4: Build Initial Pipeline (Feature Eng + Preprocessing)
    # ─────────────────────────────────────────────
    full_pipeline = get_full_pipeline(numeric_features, categorical_features)
    full_pipeline.fit(X, y)

    # ─────────────────────────────────────────────
    # 📉 Step 5: Apply Feature Selection
    # ─────────────────────────────────────────────
    X_transformed = full_pipeline.transform(X)
    # After transforming X via full_pipeline.transform(...)
    X_selected, selected_indices = apply_feature_selection(X_transformed, y)

    final_pipeline = Pipeline([
        ("feature_engineering", full_pipeline.named_steps["feature_engineering"]),
        ("preprocessing",     full_pipeline.named_steps["preprocessing"]),
        ("feature_selection", FeatureSelector(selected_features=selected_indices)),
    ])
    final_pipeline.fit(X, y)


    # ─────────────────────────────────────────────
    # 💾 Step 7: Save Final Pipeline (No Model Included)
    # ─────────────────────────────────────────────
    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_pipeline, "models/full_pipeline.pkl")
        print("✅ Full preprocessing pipeline (with feature selection) saved at 'models/full_pipeline.pkl'")

        # Save only the preprocessed features (no labels) into Postgres
        df_pre = pd.DataFrame(
            X_selected,
            columns=[f"f_{i}" for i in selected_indices]
        )
        save_dataframe_to_postgres(df_pre, table_name="preprocessed_train_data")

    # ─────────────────────────────────────────────
    # 🚀 Step 8: Register to MLflow
    # ─────────────────────────────────────────────
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

    # ─────────────────────────────────────────────
    # 🔁 Step 9: Return for Upstream Use
    # ─────────────────────────────────────────────
    if return_pipeline:
        return X_selected, y, final_pipeline

    return X_selected, y


if __name__ == "__main__":
    run_pipeline(save=True, register=True)
