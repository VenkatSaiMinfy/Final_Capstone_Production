# src/airflow/utils/airflow_loader.py

import os
import pandas as pd
from sqlalchemy import create_engine
from airflow.exceptions import AirflowException

def load_data(table_name: str) -> pd.DataFrame:
    """
    Load a full table from PostgreSQL into a DataFrame.
    Expects these env vars: 
      • DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASSWORD")

    if not all([host, port, db, user, pwd]):
        raise AirflowException(
            "Postgres credentials not fully set. "
            "Please define DB_HOST, DB_PORT, DB_NAME, DB_USER, and DB_PASSWORD."
        )

    conn_str = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    engine = create_engine(conn_str)
    try:
        df = pd.read_sql_table(table_name, engine)
    except Exception as e:
        raise AirflowException(f"Failed to load table '{table_name}': {e}")
    finally:
        engine.dispose()

    return df
