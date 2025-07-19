# src/ml/data_loader/data_loader.py

import pandas as pd
from src.db.db_utils import get_db_engine  # ⬅️ Clean import from shared db_utils
import os
from dotenv import load_dotenv
load_dotenv()
import boto3
import io
bucket=(os.getenv("S3_BUCKET"))
def load_data_from_postgres(table_name: str) -> pd.DataFrame:
    """
    Load data from PostgreSQL using shared engine.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        print(f"[INFO] Loaded data from '{table_name}', shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load data: {e}")


def load_csv_to_postgres(csv_path: str, table_name: str, if_exists: str = "replace"):
    """
    Loads a CSV file into a PostgreSQL table.

    Args:
        csv_path (str): Path to the CSV file.
        table_name (str): Name of the target PostgreSQL table.
        if_exists (str): What to do if table exists: 'replace', 'append', or 'fail'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    engine = get_db_engine()

    df.to_sql(table_name, engine, index=False, if_exists=if_exists)
    print(f"✅ Data loaded into table '{table_name}' in PostgreSQL")


def save_dataframe_to_postgres(
    df: pd.DataFrame,
    key: str,
    file_format: str = "csv",
    **kwargs
):
    """
    Save pandas DataFrame to S3 as CSV or Parquet, with bucket loaded from env.
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

# Usage:
# save_dataframe_to_s3(df=my_dataframe, key="project1/results/mydata.csv")
