# src/ml/predict.py

import os
import sys
import pandas as pd
import joblib

here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(here, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load the full pipeline (preprocessor + model)
full_pipeline_path = os.path.join(project_root, "models", "full_pipeline.pkl")
full_pipeline = joblib.load(full_pipeline_path)

def predict_lead(input_dict: dict) -> float:
    df = pd.DataFrame([input_dict])
    proba = full_pipeline.predict_proba(df)[:, 1][0]
    return float(proba)

def predict_batch(df: pd.DataFrame) -> list:
    probs = full_pipeline.predict_proba(df)[:, 1]
    return [float(p) for p in probs]
