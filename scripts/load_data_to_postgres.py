# scripts/load_data_to_postgres.py

import sys
import os
import pandas as pd

# Add the root of the project to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.db_utils import get_db_engine
from src.ml.data_loader.data_loader import load_csv_to_postgres

csv_path = "data/lead_scoring.csv"
table_name = os.getenv("DB_TABLE")

if __name__ == "__main__":
    load_csv_to_postgres(csv_path, table_name)
