# debug_predict.py
import os, sys
import pandas as pd
import numpy as np
import mlflow.sklearn

# 1) Load the same pipeline & model you’re using in Flask
PREPROCESSOR_NAME = "LeadScoringPreprocessor"
MODEL_NAME = "LeadScoringBestModel"
STAGE = "Production"

preprocessor = mlflow.sklearn.load_model(f"models:/{PREPROCESSOR_NAME}/{STAGE}")
model        = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{STAGE}")

# 2) Read the same CSV you’re uploading
csv_path = "uploads/test_Lead_Scoring.csv"   # ← point to your test file
df = pd.read_csv(csv_path)
print("🗄️  Loaded CSV:", df.shape)

# 3) Run the transform & predict_proba steps
X_proc = preprocessor.transform(df)
print("🔧 After transform:", type(X_proc), "shape:", X_proc.shape)

raw = model.predict_proba(X_proc)
print("📈 Raw predict_proba:", type(raw), repr(raw)[:300])

arr = np.asarray(raw)
print("🔍 As NumPy array:", arr.ndim, "dims:", arr.shape)

# 4) Try your indexing logic
if arr.ndim == 1:
    print("→ returning 1D array as-is")
elif arr.ndim == 2 and arr.shape[1] >= 2:
    print("→ returning arr[:,1]:", arr[:,1][:5])
elif arr.ndim == 2 and arr.shape[1] == 1:
    print("→ returning arr[:,0]:", arr[:,0][:5])
else:
    print("❗ Unexpected shape, needs custom handling")
