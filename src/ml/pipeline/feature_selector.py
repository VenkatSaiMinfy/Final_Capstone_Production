import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        # Expecting `selected_features` as integer indices
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X)
        return arr[:, self.selected_features]
