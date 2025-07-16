# src/ml/pipeline/pipeline_runner.py

import pandas as pd
import os
import sys
import joblib
import mlflow

from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.ml.data_loader.data_loader import load_data_from_postgres
from src.ml.pipeline.preprocessing import clean_columns, get_preprocessing_pipeline
from src.ml.pipeline.feature_engineering import feature_engineering
from src.ml.pipeline.feature_selection import apply_feature_selection
from src.ml.registry.model_registry import register_and_promote

def run_pipeline(save=True, register=False):
    df = load_data_from_postgres("lead_data")
    df = clean_columns(df)
    df = feature_engineering(df)

    target = 'Converted'
    y = df[target]
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    preprocessor = get_preprocessing_pipeline(num_cols, cat_cols)
    X_processed = preprocessor.fit_transform(X)

    X_selected, _ = apply_feature_selection(X_processed, y)

    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        print("✅ Preprocessing pipeline saved to 'models/preprocessor.pkl'")

    if register:
        try:
            from src.ml.registry.model_registry import register_and_promote
            register_and_promote(
                registry_name="LeadScoringPreprocessor",
                model_object=preprocessor,
                is_pipeline=True
            )
        except Exception as e:
            print(f"❌ Failed to register/promote preprocessor: {e}")

    return X_selected, y  # ✅ Always return this


if __name__ == "__main__":
    run_pipeline(save=True, register=True)
