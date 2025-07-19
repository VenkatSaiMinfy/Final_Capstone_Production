import os
import pandas as pd
from flask import Blueprint, request, render_template, jsonify
from werkzeug.utils import secure_filename

from .utils.prediction import predict_lead, predict_batch
from .utils.upload import handle_csv_upload

bp = Blueprint("routes", __name__)
ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/", methods=["GET", "POST"])
def index():
    """
    Renders the home page with single-lead JSON and batch CSV upload options.
    """
    return render_template("index.html")

@bp.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload with lead features and returns conversion probability.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    try:
        proba = predict_lead(data)
        return jsonify({"conversion_probability": proba})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/upload", methods=["POST"])
def upload():
    """
    Accepts a CSV file, stores it, runs batch prediction, and returns results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(upload_path)

        try:
            # Read and validate CSV
            df = pd.read_csv(upload_path)

            if df.empty:
                return jsonify({"error": "Uploaded file is empty."}), 400

            # Save to DB
            handle_csv_upload(upload_path, table_name="uploaded_leads")

            # Get predictions
            predictions = predict_batch(df)
            df["prediction"] = predictions

            # Clean up NaNs for JSON serialization
            response_data = df.where(pd.notnull(df), None).to_dict(orient="records")

            return jsonify({"predictions": response_data})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unsupported file type. Only .csv allowed"}), 400
