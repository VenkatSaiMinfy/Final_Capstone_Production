# src/ml/pipeline/custom_transformers.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CleanColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_cols = [
            'Prospect ID', 'Lead Number', 'Magazine',
            'Receive More Updates About Our Courses',
            'Update me on Supply Chain Content',
            'Get updates on DM Content',
            'I agree to pay the amount through cheque',
            'Newspaper Article', 'X Education Forums',
            'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
            'Last Notable Activity', 'Page Views Per Visit'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[col for col in self.drop_cols if col in X.columns], errors='ignore')

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in ['TotalVisits', 'Total Time Spent on Website']:
            if col in df.columns:
                df[f'{col}_is_zero'] = (df[col] == 0).astype(int)
        if {'TotalVisits', 'Total Time Spent on Website'}.issubset(df.columns):
            df['EngagementScore'] = df['TotalVisits'] * df['Total Time Spent on Website']
        return df
