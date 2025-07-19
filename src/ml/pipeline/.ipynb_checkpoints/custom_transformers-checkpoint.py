# src/ml/pipeline/custom_transformers.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CleanColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_cols = [
            'prospect_id', 'lead_number', 'magazine',
            'receive_more_updates_about_our_courses',
            'update_me_on_supply_chain_content',
            'get_updates_on_dm_content',
            'i_agree_to_pay_the_amount_through_cheque',
            'newspaper_article', 'x_education_forums',
            'asymmetrique_activity_index', 'asymmetrique_profile_index',
            'last_notable_activity', 'page_views_per_visit'
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
