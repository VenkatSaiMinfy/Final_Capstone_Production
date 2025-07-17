# src/ml/pipeline/feature_engineering.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def add_behavioral_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['TotalVisits', 'Total Time Spent on Website']:
            if col in df.columns:
                df[f'{col}_is_zero'] = (df[col] == 0).astype(int)
        return df

    def add_combined_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if {'TotalVisits', 'Total Time Spent on Website'}.issubset(df.columns):
            df['EngagementScore'] = df['TotalVisits'] * df['Total Time Spent on Website']
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = self.add_behavioral_flags(X_copy)
        X_copy = self.add_combined_features(X_copy)
        return X_copy
