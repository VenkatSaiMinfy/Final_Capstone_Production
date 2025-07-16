# src/ml/pipeline/feature_engineering.py

import pandas as pd

def add_behavioral_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary flags like is_zero for skewed numeric features.
    """
    for col in ['TotalVisits', 'Total Time Spent on Website']:
        if col in df.columns:
            df[f'{col}_is_zero'] = (df[col] == 0).astype(int)
    return df

def add_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features like engagement score.
    """
    if {'TotalVisits', 'Total Time Spent on Website'}.issubset(df.columns):
        df['EngagementScore'] = df['TotalVisits'] * df['Total Time Spent on Website']
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = add_behavioral_flags(df)
    df = add_combined_features(df)
    return df
