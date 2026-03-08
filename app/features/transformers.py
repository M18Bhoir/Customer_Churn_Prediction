import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class ColumnExcluder(BaseEstimator, TransformerMixin):
    """
    Exclude specified columns from the dataset
    """

    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns or ["customer_id", "age", "monthly_charges"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Only exclude columns that exist in the dataframe
        cols_to_exclude = [col for col in self.exclude_columns if col in X.columns]
        
        if cols_to_exclude:
            X = X.drop(columns=cols_to_exclude)
        
        return X


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values for numeric and categorical columns
    """

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        self.categorical_cols = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # Safe handling for empty columns
        if self.numeric_cols:
            self.numeric_medians = X[self.numeric_cols].median()
        else:
            self.numeric_medians = {}

        if self.categorical_cols:
            self.categorical_modes = X[self.categorical_cols].mode().iloc[0] if len(X) > 0 else {}
        else:
            self.categorical_modes = {}

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.numeric_medians[col])
        
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.categorical_modes[col])

        return X


class CategoricalLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Apply LabelEncoder to categorical variables
    """

    def __init__(self):
        self.label_encoders = {}
        self.categorical_cols = []

    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))

        return X


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    Transform numerical features: impute and scale
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.numeric_cols = []

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        if self.numeric_cols:
            X_numeric = X[self.numeric_cols].copy()
            
            # Ensure columns are actually numeric - handle mixed types
            for col in self.numeric_cols:
                try:
                    X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
                except:
                    pass
            
            self.imputer.fit(X_numeric)
            X_imputed = self.imputer.transform(X_numeric)
            self.scaler.fit(X_imputed)

        return self

    def transform(self, X):
        X = X.copy()

        if self.numeric_cols:
            X_numeric = X[self.numeric_cols].copy()
            
            # Ensure columns are actually numeric - handle mixed types
            for col in self.numeric_cols:
                try:
                    X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
                except:
                    pass
            
            X_numeric = self.imputer.transform(X_numeric)
            X_numeric = self.scaler.transform(X_numeric)
            X[self.numeric_cols] = X_numeric

        return X


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Create domain-based engineered features
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # # Average charge per tenure month
        # if "total_charges" in X.columns and "tenure_months" in X.columns:
        #     X["avg_monthly_value"] = (
        #         X["total_charges"] / (X["tenure_months"] + 1)
        #     )

        # # Support intensity
        # if "support_calls" in X.columns and "tenure_months" in X.columns:
        #     X["support_call_ratio"] = (
        #         X["support_calls"] / (X["tenure_months"] + 1)
        #     )

        # # Payment risk indicator
        # if "late_payments" in X.columns:
        #     X["payment_risk"] = np.where(
        #         X["late_payments"] > 2, 1, 0
        #     )

        return X
