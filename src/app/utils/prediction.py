import os
import sys
import pandas as pd
import numpy as np
import mlflow.sklearn
from typing import Union, List

# ─────────────────────────────────────────────
# Set project root for module imports if needed
# ─────────────────────────────────────────────
here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(here, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ─────────────────────────────────────────────
# Constants: Model Registry Names and Stage
# ─────────────────────────────────────────────
PREPROCESSOR_NAME = "LeadScoringPreprocessor"
MODEL_NAME = "LeadScoringBestModel"
STAGE = "Production"

# ─────────────────────────────────────────────
# Load Preprocessor Pipeline & Model
# ─────────────────────────────────────────────
try:
    # This pipeline includes feature_eng → preprocessing → feature_selection
    preprocessor_uri = f"models:/{PREPROCESSOR_NAME}/{STAGE}"
    preprocessor = mlflow.sklearn.load_model(preprocessor_uri)

    # This is just the trained classifier for predict_proba
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"✅ Loaded preprocessor and model from MLflow registry (stage={STAGE})")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or preprocessor: {e}")


# ─────────────────────────────────────────────
# Single-lead prediction
# ─────────────────────────────────────────────
def predict_lead(input_dict: dict) -> Union[float, dict]:
    try:
        df = pd.DataFrame([input_dict])
        X_proc = preprocessor.transform(df)
        raw = model.predict_proba(X_proc)

        arr = np.asarray(raw)
        # Handle 1D or 2D outputs
        if arr.ndim == 1:
            return float(arr[0])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return float(arr[0, 1])
        if arr.ndim == 2 and arr.shape[1] == 1:
            return float(arr[0, 0])

        return {"error": f"Unexpected output shape: {arr.shape}"}

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────
def predict_batch(df: pd.DataFrame) -> Union[List[float], dict]:
    """
    Predict conversion probabilities for a batch of leads.
    Includes debug logs to inspect intermediate shapes and types.
    """
    try:
        # 1) Full pipeline transform
        X_proc = preprocessor.transform(df)
        print("🔧 [DEBUG] After transform: type=", type(X_proc), 
              "shape=", getattr(X_proc, "shape", None))

        # 2) Raw model output
        raw = model.predict_proba(X_proc)
        print("📈 [DEBUG] Raw predict_proba output: type=", type(raw), 
              "repr=", raw)

        # 3) Coerce to NumPy array
        arr = np.asarray(raw)
        print("🔍 [DEBUG] As array: ndim=", arr.ndim, "shape=", arr.shape)

        # 4) Index appropriately
        if arr.ndim == 1:
            return [float(x) for x in arr]
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return [float(x) for x in arr[:, 1]]
        if arr.ndim == 2 and arr.shape[1] == 1:
            return [float(x) for x in arr[:, 0]]

        return {"error": f"Unexpected output shape: {arr.shape}"}

    except Exception as e:
        print("❌ [ERROR] in predict_batch:", e)
        return {"error": str(e)}



# ─────────────────────────────────────────────
# (Optional) Debug test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Create a dummy row with the exact columns your preprocess pipeline expects
    sample = {col: 0 for col in preprocessor.feature_names_in_}  # fill zeros
    print("Single:", predict_lead(sample))

    df = pd.DataFrame([sample, sample])
    print("Batch:", predict_batch(df))
