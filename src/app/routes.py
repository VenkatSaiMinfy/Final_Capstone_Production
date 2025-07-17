import os
from flask import Blueprint, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename

from .utils.prediction import predict_lead
from .utils.upload import handle_csv_upload

bp = Blueprint("routes", __name__)

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/", methods=["GET", "POST"])
def index():
    """
    Renders a simple form for single‐lead prediction (JSON) or CSV upload.
    """
    return render_template("index.html")

@bp.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload with lead features, returns conversion probability.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error":"Invalid JSON"}), 400
    try:
        proba = predict_lead(data)
        return jsonify({"conversion_probability": proba})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(upload_path)

        try:
            # Save to database
            handle_csv_upload(upload_path, table_name="uploaded_leads")

            # Optional: Predict on uploaded CSV immediately
            import pandas as pd
            from app.utils.prediction import predict_batch

            df = pd.read_csv(upload_path)
            predictions = predict_batch(df)  # You must write this function in prediction.py

            return jsonify({"predictions": predictions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unsupported file type"}), 400
