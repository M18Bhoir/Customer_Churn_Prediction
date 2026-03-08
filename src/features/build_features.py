import os
import yaml
import pandas as pd

from sklearn.pipeline import Pipeline

from src.logger.logger import get_logger
from src.schema.data_schema import DataConfig
from src.features.transformers import (
    ColumnExcluder,
    MissingValueImputer,
    CategoricalLabelEncoder
)

logger = get_logger(__name__)


class FeatureBuilder:
    """
    Feature Engineering Pipeline

    Responsibilities:
    - Exclude specified columns
    - Apply transformations
    - Encode categorical variables
    - Generate features
    - Save processed dataset
    """

    def __init__(self, config):

        self.config = DataConfig(**config["data"])

        os.makedirs(
            os.path.dirname(self.config.processed_path),
            exist_ok=True
        )

        # Get exclude columns from training config if available
        exclude_columns = config.get("training", {}).get("exclude_columns", 
                                                          ["customer_id", "age", "monthly_charges"])

        self.pipeline = Pipeline(
            steps=[
                ("column_excluder", ColumnExcluder(exclude_columns=exclude_columns)),
                ("imputer", MissingValueImputer()),
                ("label_encoder", CategoricalLabelEncoder())
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

