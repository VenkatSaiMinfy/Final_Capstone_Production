# src/ml/pipeline/feature_selection.py

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def apply_feature_selection(X, y, top_n=20):
    """
    Uses RFE with RandomForest to select top N features.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(model, n_features_to_select=top_n)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_selected, selected_features
