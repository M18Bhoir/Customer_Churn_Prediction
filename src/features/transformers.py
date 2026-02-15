import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values for numeric and categorical columns
    """

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns

        self.categorical_cols = X.select_dtypes(
            include=["object"]
        ).columns

        self.numeric_medians = X[self.numeric_cols].median()
        self.categorical_modes = X[self.categorical_cols].mode().iloc[0]

        return self

    def transform(self, X):
        X = X.copy()

        X[self.numeric_cols] = X[self.numeric_cols].fillna(
            self.numeric_medians
        )

        X[self.categorical_cols] = X[self.categorical_cols].fillna(
            self.categorical_modes
        )

        return X


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Create domain-based engineered features
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Average charge per tenure month
        X["avg_monthly_value"] = (
            X["total_charges"] / (X["tenure_months"] + 1)
        )

        # Support intensity
        X["support_call_ratio"] = (
            X["support_calls"] / (X["tenure_months"] + 1)
        )

        # Payment risk indicator
        X["payment_risk"] = np.where(
            X["late_payments"] > 2, 1, 0
        )

        return X
