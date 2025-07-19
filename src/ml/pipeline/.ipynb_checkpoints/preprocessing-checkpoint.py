# src/ml/pipeline/preprocessing.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.ml.pipeline.feature_engineering import FeatureEngineeringTransformer

# ─────────────────────────────────────────────────────────────
# 🧹 Column Cleaner
# ─────────────────────────────────────────────────────────────
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        'prospect_id', 'lead_number', 'magazine',
        'receive_more_updates_about_our_courses',
        'update_me_on_supply_chain_content',
        'get_updates_on_dm_content',
        'i_agree_to_pay_the_amount_through_cheque',
        'newspaper_article', 'x_education_forums',
        'asymmetrique_activity_index', 'asymmetrique_profile_index',
        'last_notable_activity', 'page_views_per_visit'

    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df

# ─────────────────────────────────────────────────────────────
# ⚙️ Column-wise Preprocessing Pipelines
# ─────────────────────────────────────────────────────────────
def get_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

# ─────────────────────────────────────────────────────────────
# 🔄 Full Pipeline (Feature Eng + Preprocessing)
# ─────────────────────────────────────────────────────────────
def get_full_pipeline(numeric_features, categorical_features):
    """
    Returns a pipeline with FeatureEngineering and Preprocessing steps.
    Model is not included here.
    """
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)

    full_pipeline = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("preprocessing", preprocessor)
    ])

    return full_pipeline
