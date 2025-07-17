from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def apply_feature_selection(X, y, top_n=20):
    """
    Uses RFE with RandomForest to select top N features.
    Returns:
        - X_selected: transformed array with selected features
        - selected_indices: list of integer indices of selected features
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(model, n_features_to_select=top_n)
    selector.fit(X, y)

    selected_indices = list(selector.get_support(indices=True))
    X_selected = selector.transform(X)

    return X_selected, selected_indices
