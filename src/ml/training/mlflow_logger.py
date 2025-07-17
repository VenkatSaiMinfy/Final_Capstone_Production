import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def mlflow_logger(model_name, model, metrics, params, X_sample=None):
    # End any active MLflow run if not properly closed
    if mlflow.active_run():
        print(f"⚠️ Active run detected. Ending it before starting a new one.")
        mlflow.end_run()

    try:
        print(f"\n🕒 Starting MLflow run for {model_name}")
        mlflow.start_run(run_name=model_name)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Infer model signature and input example
        signature = None
        input_example = None
        if X_sample is not None:
            try:
                y_pred = model.predict(X_sample)
                signature = infer_signature(X_sample, y_pred)
                input_example = X_sample[:5]
            except Exception as e:
                print(f"⚠️ Could not infer signature: {e}")

        # Log the model with or without signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            input_example=input_example,
            signature=signature
        )
        print(f"[MLflow ✅] Logged model: {model_name}")

    except Exception as e:
        print(f"❌ Error logging model to MLflow: {e}")

    finally:
        # Always end the run to prevent locking
        if mlflow.active_run():
            mlflow.end_run()
