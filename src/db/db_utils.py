# src/db/db_utils.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load DB credentials from .env file
load_dotenv()

def get_db_engine():
    """
    Creates SQLAlchemy engine using environment variables.
    """
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_pass, db_host, db_port, db_name]):
        raise EnvironmentError("Database credentials are not fully set in .env")

    connection_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_url)