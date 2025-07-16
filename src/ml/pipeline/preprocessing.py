# src/ml/pipeline/preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        'Prospect ID', 'Lead Number', 'Magazine',
        'Receive More Updates About Our Courses',
        'Update me on Supply Chain Content',
        'Get updates on DM Content',
        'I agree to pay the amount through cheque',
        'Newspaper Article', 'X Education Forums',
        'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
        'Last Notable Activity', 'Page Views Per Visit'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df
