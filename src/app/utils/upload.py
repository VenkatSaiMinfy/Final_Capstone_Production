import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.ml.data_loader.data_loader import load_csv_to_postgres

def handle_csv_upload(filepath: str, table_name: str):
    """
    Loads the CSV at `filepath` into Postgres table `table_name`.
    """
    # Delegates to your shared utility
    load_csv_to_postgres(filepath, table_name, if_exists="replace")
