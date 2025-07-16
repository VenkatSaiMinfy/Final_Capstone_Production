# src/ml/data_loader/data_loader.py

import pandas as pd
from src.db.db_utils import get_db_engine  # ⬅️ Clean import from shared db_utils
import os
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
