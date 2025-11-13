"""Shared preprocessing components for the NYC taxi deployment."""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler


class DummyEncoder(BaseEstimator, TransformerMixin):
    """Custom one-hot encoder that preserves train-time column order."""

    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.columns_ = None

    def fit(self, X, y=None):  # noqa: D401
        X_copy = X.copy()
        for col in self.categorical_cols:
            X_copy[col] = X_copy[col].astype('category')
        X_encoded = pd.get_dummies(X_copy, columns=self.categorical_cols, drop_first=True)
        self.columns_ = X_encoded.columns.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.categorical_cols:
            X_copy[col] = X_copy[col].astype('category')
        X_encoded = pd.get_dummies(X_copy, columns=self.categorical_cols, drop_first=True)
        for col in self.columns_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.columns_]
        return X_encoded.astype(float).values


class MixedScaler(BaseEstimator, TransformerMixin):
    """Apply StandardScaler to select columns and PowerTransformer to the rest."""

    def __init__(self, std_scaler_cols, power_transform_cols):
        self.std_scaler_cols = std_scaler_cols
        self.power_transform_cols = power_transform_cols
        self.std_scaler = StandardScaler()
        self.power_transformer = PowerTransformer()

    def fit(self, X, y=None):  # noqa: D401
        self.std_scaler.fit(X[self.std_scaler_cols])
        self.power_transformer.fit(X[self.power_transform_cols])
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.std_scaler_cols] = self.std_scaler.transform(X[self.std_scaler_cols])
        X_scaled[self.power_transform_cols] = self.power_transformer.transform(X[self.power_transform_cols])
        return X_scaled


sys.modules.setdefault('deployment.pipeline_components', sys.modules[__name__])
