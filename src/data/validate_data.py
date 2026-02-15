import yaml
import pandas as pd
from typing import List

from src.logger.logger import get_logger
from src.schema.data_schema import DataConfig


logger = get_logger(__name__)


class DataValidator:
    """
    Industrial Data Validator

    Responsibilities:
    - Validate schema compliance
    - Check missing values
    - Detect duplicates
    - Validate target column
    - Log data health metrics
    """

    def __init__(self, config):

        self.config = DataConfig(**config["data"])

        logger.info("DataValidator initialized")

    # ---------------------------------------------------------
    # Required Columns Check
    # ---------------------------------------------------------
    def _validate_required_columns(self, df: pd.DataFrame):

        missing_cols = [
            col for col in self.config.required_columns
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )

        logger.info("Required column validation passed")

    # ---------------------------------------------------------
    # Missing Value Validation
    # ---------------------------------------------------------
    def _validate_missing_values(self, df: pd.DataFrame):

        missing_pct = df.isnull().mean()

        high_missing = missing_pct[
            missing_pct > self.config.allowed_missing_pct
        ]

        if len(high_missing) > 0:
            raise ValueError(
                f"Columns exceeding missing threshold: "
                f"{high_missing.to_dict()}"
            )

        logger.info(
            f"Missing value validation passed. "
            f"Max missing={missing_pct.max():.4f}"
        )

    # ---------------------------------------------------------
    # Target Validation
    # ---------------------------------------------------------
    def _validate_target(self, df: pd.DataFrame):

        target = self.config.target_column

        if target not in df.columns:
            raise ValueError("Target column missing")

        unique_values = df[target].nunique()

        if unique_values < 2:
            raise ValueError(
                "Target column has only one class"
            )

        logger.info(
            f"Target validation passed. Classes={unique_values}"
        )

    # ---------------------------------------------------------
    # Duplicate Validation
    # ---------------------------------------------------------
    def _validate_duplicates(self, df: pd.DataFrame):

        dup_count = df.duplicated().sum()

        if dup_count > 0:
            logger.warning(
                f"Dataset contains {dup_count} duplicate rows"
            )
        else:
            logger.info("No duplicate rows detected")

    # ---------------------------------------------------------
    # Main Validation Entry
    # ---------------------------------------------------------
    def validate(self, df: pd.DataFrame):

        logger.info("Starting data validation")

        self._validate_required_columns(df)
        self._validate_missing_values(df)
        self._validate_target(df)
        self._validate_duplicates(df)

        logger.info("Data validation completed successfully")
