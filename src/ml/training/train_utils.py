import os
import sys
import joblib
import shap
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import warnings

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.ml.evaluation.metrics import compute_metrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
MODEL_DIR = os.path.join("src", "ml", "model_objects")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 📈 SHAP Summary Plot Logger
# ─────────────────────────────────────────────────────────────
def log_shap_plot(model_name, model, X_train, X_test):
    try:
        print(f"🔍 Generating SHAP for {model_name}")
        if model_name in ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, X_train)

        shap_values = explainer(X_test[:100])  # Limit to 100 rows
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.figure()
            shap.summary_plot(shap_values, X_test[:100], show=False)
            plt.tight_layout()
            plt.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path=f"{model_name}_shap")
            print(f"📈 SHAP plot logged for {model_name}")
    except Exception as e:
        print(f"⚠️ SHAP failed for {model_name}: {e}")

# ─────────────────────────────────────────────────────────────
# 🎯 Train + Evaluate + Log to MLflow (as nested run)
# # src/ml/training/train_utils.py

import os
import joblib
import numpy as np
import mlflow
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.ml.evaluation.metrics import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_DIR = os.path.join("src", "ml", "model_objects")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_log_model(name, model, param_grid, X_train, X_test, y_train, y_test):
    """
    Trains, evaluates, logs to MLflow under a nested run, and returns
    (model_name, run_id, f1_score).
    """
    print(f"\n📌 Training model: {name}")

    # 1) Choose CV search
    search = (
        RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=min(5, len(param_grid)),
            cv=3,
            scoring="f1",
            n_jobs=-1,
            random_state=42,
            return_train_score=True,
        )
        if name in ["XGBoost", "LightGBM"]
        else GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            return_train_score=True,
        )
    )

    # 2) Fit and select best
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # 3) Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = (
        best_model.predict_proba(X_test)[:, 1]
        if hasattr(best_model, "predict_proba")
        else None
    )
    metrics = compute_metrics(y_test, y_pred, y_prob)
    f1 = metrics["f1_score"]

    print(f"✅ Best Params: {search.best_params_}")
    print(f"✅ Accuracy: {metrics['accuracy']:.4f} | F1: {f1:.4f}")

    # 4) Save locally (optional)
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name}_model.pkl"))

    # 5) Add CV scores
    metrics.update(
        {
            "cv_mean_train_score": float(np.mean(search.cv_results_["mean_train_score"])),
            "cv_mean_test_score": float(np.mean(search.cv_results_["mean_test_score"])),
        }
    )

    # 6) Log under nested MLflow run
    with mlflow.start_run(run_name=name, nested=True) as run:
        run_id = run.info.run_id

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        # Infer signature
        try:
            from mlflow.models.signature import infer_signature

            input_example = X_train[:5]
            signature = infer_signature(input_example, best_model.predict(input_example))
        except Exception as e:
            print(f"⚠️ Could not infer signature: {e}")
            input_example, signature = None, None

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=name,
            signature=signature,
            input_example=input_example,
        )

        # SHAP summary (if desired)
        log_shap_plot(name, best_model, X_train, X_test)

    # 7) Return the info for later registration
    return name, run_id, f1

# ─────────────────────────────────────────────────────────────
# 🧠 Define Models + Param Grids
# ─────────────────────────────────────────────────────────────
def get_models_with_params():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    return {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            'C': [0.1, 1.0, 10]
        }),
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [100],
            'max_depth': [None, 10, 20]
        }),
        "GradientBoosting": (GradientBoostingClassifier(), {
            'n_estimators': [100],
            'learning_rate': [0.01, 0.1]
        }),
        "XGBoost": (XGBClassifier(eval_metric='logloss', use_label_encoder=False), {
            'n_estimators': [100],
            'learning_rate': [0.05, 0.1]
        }),
        "LightGBM": (LGBMClassifier(verbose=-1), {
            'n_estimators': [100],
            'learning_rate': [0.05, 0.1]
        }),
        "SVM": (SVC(probability=True), {
            'C': [0.1, 1.0],
            'kernel': ['linear', 'rbf']
        })
    }