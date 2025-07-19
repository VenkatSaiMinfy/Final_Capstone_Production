# src/ml/data_loader/data_loader.py

import os
import io
import boto3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from src.db.db_utils import get_db_engine

load_dotenv()

bucket = os.getenv("S3_BUCKET")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")  # default region
iam_role = os.getenv("REDSHIFT_IAM_ROLE")  # Must be provided in .env

def load_data_from_postgres(table_name: str) -> pd.DataFrame:
    """
    Load data from Redshift using shared engine.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        print(f"[INFO] Loaded data from '{table_name}', shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load data from Redshift: {e}")


def load_csv_to_postgres(csv_path: str, table_name: str, if_exists: str = "replace"):
    """
    Uploads CSV to S3 and loads it into Redshift using COPY.

    Args:
        csv_path (str): Local path to CSV.
        table_name (str): Redshift table name.
        if_exists (str): 'replace' or 'append'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    s3_key = f"tmp/{table_name}.csv"
    s3_path = f"s3://{bucket}/{s3_key}"

    # Upload to S3
    s3 = boto3.client("s3")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=buffer.getvalue())
    print(f"[INFO] Uploaded CSV to {s3_path}")

    # Load into Redshift
    engine = get_redshift_engine()
    with engine.connect() as conn:
        if if_exists == "replace":
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))

        # Create table automatically using Pandas dtype mapping
        df.head(0).to_sql(table_name, engine, index=False, if_exists="append")

        # COPY from S3 to Redshift
        copy_sql = text(f"""
            COPY {table_name}
            FROM '{s3_path}'
            IAM_ROLE '{iam_role}'
            FORMAT AS CSV
            IGNOREHEADER 1
            REGION '{aws_region}';
        """)
        conn.execute(copy_sql)
        print(f"✅ Data loaded into table '{table_name}' in Redshift using COPY")


def save_dataframe_to_postgres(
    df: pd.DataFrame,
    key: str,
    file_format: str = "csv",
    **kwargs
):
    """
    Save pandas DataFrame to S3 as CSV or Parquet using boto3.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty and cannot be saved.")

    s3 = boto3.client("s3")

    if file_format == "csv":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, **kwargs)
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    elif file_format == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, **kwargs)
        buffer.seek(0)
        s3.put_object(Bucket=bucket, Key=key, Body=buffer)
    else:
        raise ValueError("file_format must be 'csv' or 'parquet'.")

    print(f"✅ DataFrame saved to s3://{bucket}/{key} ({file_format.upper()})")
