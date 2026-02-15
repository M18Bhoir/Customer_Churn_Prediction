import os
import yaml
import pandas as pd

from sklearn.pipeline import Pipeline

from src.logger.logger import get_logger
from src.schema.data_schema import DataConfig
from src.features.transformers import (
    MissingValueImputer,
    FeatureGenerator
)

logger = get_logger(__name__)


class FeatureBuilder:
    """
    Feature Engineering Pipeline

    Responsibilities:
    - Apply transformations
    - Generate features
    - Save processed dataset
    """

    def __init__(self, config):

        self.config = DataConfig(**config["data"])

        os.makedirs(
            os.path.dirname(self.config.processed_path),
            exist_ok=True
        )

        self.pipeline = Pipeline(
            steps=[
                ("imputer", MissingValueImputer()),
                ("feature_generator", FeatureGenerator())
            ]
        )

        logger.info("FeatureBuilder initialized")

    def build(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Starting feature engineering")

        processed_df = self.pipeline.fit_transform(df)

        processed_df.to_csv(
            self.config.processed_path,
            index=False
        )

        logger.info(
            f"Processed dataset saved at "
            f"{self.config.processed_path}"
        )

        return processed_df
