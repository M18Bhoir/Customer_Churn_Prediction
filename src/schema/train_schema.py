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


class ModelConfig(BaseModel):
    type: str
    logistic_regression: Optional[LogisticRegressionConfig]
    random_forest: Optional[RandomForestConfig]


class TrainConfig(BaseModel):
    save_model: bool
    model_registry_path: str
    random_state: int
    test_size: float
    target_column: str
    cross_validation_folds: int
    model: ModelConfig
