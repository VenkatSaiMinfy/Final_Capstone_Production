from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def apply_feature_selection(X, y, top_n=20):
    model = RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    X_selected = X[:, indices]
    return X_selected, indices.tolist()
