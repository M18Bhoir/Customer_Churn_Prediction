from pydantic import BaseModel
from typing import Optional


class LogisticRegressionConfig(BaseModel):
    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[str] = None
    n_jobs: Optional[int] = -1


class RandomForestConfig(BaseModel):
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    n_jobs: int = -1


class XGBoostConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0
    min_child_weight: int = 1
    random_state: int = 42
    n_jobs: int = -1


class ModelConfig(BaseModel):
    type: str
    logistic_regression: Optional[LogisticRegressionConfig] = None
    random_forest: Optional[RandomForestConfig] = None
    xgboost: Optional[XGBoostConfig] = None


class TrainConfig(BaseModel):
    save_model: bool
    model_registry_path: str
    random_state: int
    test_size: float
    target_column: str
    cross_validation_folds: int
    model: ModelConfig
    apply_smote: bool = True
    exclude_columns: list = ["customer_id", "age", "monthly_charges"]
