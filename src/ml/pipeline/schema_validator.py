# src/ml/pipeline/schema_validator.py

import pandas as pd

def validate_input_schema(df: pd.DataFrame, expected_columns: list) -> bool:
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")
    return True
