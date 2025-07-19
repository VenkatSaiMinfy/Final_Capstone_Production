# src/ml/training/train_utils.py

import os
import joblib
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import warnings
import shap

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.ml.evaluation.metrics import compute_metrics

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_DIR = os.path.join("src", "ml", "model_objects")
os.makedirs(MODEL_DIR, exist_ok=True)


def log_shap_plot(model_name, model, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
    try:
        print(f"🔍 Generating SHAP for {model_name}")
        if model_name in ["RandomForest","GradientBoosting","XGBoost","LightGBM"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, X_train_df)

        subset = X_test_df.iloc[:100]
        shap_values = explainer(subset)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.figure()
            shap.summary_plot(shap_values, subset, show=False)
            plt.tight_layout()
            plt.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path=f"{model_name}_shap")
        print(f"📈 SHAP plot logged for {model_name}")
    except Exception as e:
        print(f"⚠️ SHAP failed for {model_name}: {e}")


def train_and_log_model(
    name: str,
    model,
    param_grid: dict,
    X_train, X_test, y_train, y_test,
    feature_names: list = None
):
    print(f"\n📌 Training model: {name}")

    # ensure DataFrame for SHAP
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when using numpy arrays")
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df  = pd.DataFrame(X_test,  columns=feature_names)
    else:
        X_train_df = X_train.copy()
        X_test_df  = X_test.copy()

    # pick CV search
    search = (RandomizedSearchCV(
                  estimator=model,
                  param_distributions=param_grid,
                  n_iter=min(5, len(param_grid)),
                  cv=3, scoring="f1", n_jobs=-1,
                  random_state=42, return_train_score=True
              ) if name in ["XGBoost","LightGBM"]
              else GridSearchCV(
                  estimator=model,
                  param_grid=param_grid,
                  cv=3, scoring="f1", n_jobs=-1,
                  return_train_score=True
              ))

    search.fit(X_train_df, y_train)
    best_model = search.best_estimator_

    # evaluate
    y_pred = best_model.predict(X_test_df)
    y_prob = best_model.predict_proba(X_test_df)[:,1] if hasattr(best_model, "predict_proba") else None
    metrics = compute_metrics(y_test, y_pred, y_prob)
    f1 = metrics["f1_score"]

    print(f"✅ Best Params: {search.best_params_}")
    print(f"✅ Accuracy: {metrics['accuracy']:.4f} | F1: {f1:.4f}")

    # save locally
    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name}_model.pkl"))

    # add CV scores
    metrics.update({
        "cv_mean_train_score": float(np.mean(search.cv_results_["mean_train_score"])),
        "cv_mean_test_score":  float(np.mean(search.cv_results_["mean_test_score"]))
    })

    # log to MLflow under nested run
    with mlflow.start_run(run_name=name, nested=True) as run:
        run_id = run.info.run_id
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        # infer signature
        try:
            from mlflow.models.signature import infer_signature
            example = X_train_df.iloc[:5]
            sig = infer_signature(example, best_model.predict(example))
        except Exception:
            example, sig = None, None

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path=name,
            signature=sig,
            input_example=example
        )

        # shap plot
        log_shap_plot(name, best_model, X_train_df, X_test_df)

    return name, run_id, f1


def get_models_with_params():
    from sklearn.linear_model    import LogisticRegression
    from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm              import SVC
    from xgboost                  import XGBClassifier
    from lightgbm                import LGBMClassifier

    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1.0, 10]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100], "max_depth": [None, 10, 20]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(),
            {"n_estimators": [100], "learning_rate": [0.01, 0.1]}
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", use_label_encoder=False),
            {"n_estimators": [100], "learning_rate": [0.05, 0.1]}
        ),
        "LightGBM": (
            LGBMClassifier(verbose=-1),
            {"n_estimators": [100], "learning_rate": [0.05, 0.1]}
        ),
        "SVM": (
            SVC(probability=True),
            {"C": [0.1, 1.0], "kernel": ["linear","rbf"]}
        )
    }
