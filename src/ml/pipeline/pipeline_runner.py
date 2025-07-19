# src/ml/pipeline/pipeline_runner.py

import os
import sys
import joblib
import pandas as pd
from datetime import datetime
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.ml.data_loader.data_loader import load_data_from_postgres, save_dataframe_to_postgres
from src.ml.pipeline.preprocessing    import clean_columns, get_full_pipeline
from src.ml.pipeline.feature_selection import apply_feature_selection
from src.ml.pipeline.feature_selector  import FeatureSelector
from src.ml.registry.model_registry   import register_and_promote
from sklearn.pipeline import Pipeline


def print_time(step: str, t0: datetime) -> datetime:
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[⏱️] {step} finished in {elapsed:.2f}s")
    return datetime.now()


def run_pipeline(
    table_name: str = "lead_data",
    target_col: str = "converted",
    save: bool = True,
    register: bool = False,
    return_pipeline: bool = False
):
    t0 = datetime.now()

    # 1) Load & clean
    df = load_data_from_postgres(table_name)
    print(f"[INFO] Loaded data from '{table_name}', shape: {df.shape}")
    df.columns = df.columns.str.lower()
    df = clean_columns(df)

    t0 = print_time("Data load & clean", t0)

    # 2) Split X/y
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 3) Identify feature types
    numeric_features     = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    t0 = print_time("Feature type definition", t0)

    # 4) Build & fit full pipeline
    full_pipeline = get_full_pipeline(numeric_features, categorical_features)
    X_transformed = full_pipeline.fit_transform(X, y)
    t0 = print_time("Pipeline fit & transform", t0)

    # 5) Extract real feature names
    preprocessor = full_pipeline.named_steps["preprocessing"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except TypeError:
        feature_names = full_pipeline.get_feature_names_out()
    # note: this aligns 1–1 with columns of X_transformed

    # 6) Apply RFE feature selection
    X_selected, selected_indices = apply_feature_selection(X_transformed, y)
    t0 = print_time("Feature selection", t0)

    # 7) Build final inference pipeline
    final_pipeline = Pipeline([
        ("feature_engineering", full_pipeline.named_steps["feature_engineering"]),
        ("preprocessing",      full_pipeline.named_steps["preprocessing"]),
        ("feature_selection",  FeatureSelector(selected_features=selected_indices)),
    ])
    # re-fit on original X/y so pipeline internals are consistent
    final_pipeline.fit(X, y)
    t0 = print_time("Final pipeline construction", t0)

    # 8) Save artifacts
    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_pipeline, "models/full_pipeline.pkl", compress=3)
        print("✅ Full preprocessing pipeline saved to 'models/full_pipeline.pkl'")

        # build DataFrame of selected features with real names
        selected_names = [feature_names[i] for i in selected_indices]
        df_pre = pd.DataFrame(X_selected, columns=selected_names)

        # save to Postgres
        save_dataframe_to_postgres(df_pre, key="preprocessed_train_data")
        print(f"✅ Preprocessed data saved to 'preprocessed_train_data' with columns: {selected_names}")
        t0 = print_time("Saving pipeline & features", t0)

    # 9) Optional MLflow pipeline registration
    if register:
        try:
            register_and_promote(
                registry_name="LeadScoringPreprocessor",
                model_object=final_pipeline,
                is_pipeline=True
            )
            print("✅ Pipeline registered in MLflow as 'LeadScoringPreprocessor'")
        except Exception as e:
            print(f"❌ Registration failed: {e}")
        t0 = print_time("MLflow registration", t0)

    # 10) Return for training
    if return_pipeline:
        return X_selected, y, final_pipeline

    return X_selected, y


if __name__ == "__main__":
    run_pipeline(save=True, register=True)
