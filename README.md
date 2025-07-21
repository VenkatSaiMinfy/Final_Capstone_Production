# Sample Env
```text
# ==========================
# 🛠️ DATABASE CONFIGURATION
# ==========================
DB_USER=your_db_user
DB_PASSWORD=your_secure_password
DB_HOST=your-redshift-cluster.endpoint.region.redshift.amazonaws.com
DB_PORT=5439
DB_NAME=your_database_name

# ====================
# 📊 TABLE NAMES SETUP
# ====================
DB_TABLE=lead_data
DB_TEST_TABLE_UPLOADED=test_lead_data
DB_TABLE_PREPROCESSED=preprocessed_train_data
DB_NEW_TABLE_PREPROECESSED=user_uploaded_preprocessed

# ================================
# 🚀 MLFLOW TRACKING CONFIGURATION
# ================================
MLFLOW_TRACKING_URI=http://your-mlflow-server-address:5000
# Example: http://mlflow.example.com:5000

# =========================
# ☁️ AWS S3 CONFIGURATION
# =========================
S3_BUCKET=your_s3_bucket_name
```


