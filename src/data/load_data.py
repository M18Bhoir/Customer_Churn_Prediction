import os
import yaml
import pandas as pd
from typing import Optional

from src.logger.logger import get_logger
from src.schema.data_schema import DataConfig


logger = get_logger(__name__)


class DataLoader:
    """
    Industrial Data Loader

    Responsibilities:
    - Load dataset from configured source
    - Validate file existence
    - Perform basic structural validation
    - Return pandas dataframe
    """

    def __init__(self, config):

        self.config = DataConfig(**config["data"])

        logger.info("DataLoader initialized")

    # ---------------------------------------------------------
    # Load Data
    # ---------------------------------------------------------
    def load(self) -> pd.DataFrame:

        file_path = self.config.input_path

        logger.info(f"Loading data from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Input data file not found at {file_path}"
            )

        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)

            elif file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)

            else:
                raise ValueError(
                    "Unsupported file format. Use CSV or Parquet."
                )

        except Exception as e:
            logger.exception("Failed while loading dataset")
            raise e

        logger.info(
            f"Data loaded successfully. Shape={df.shape}"
        )

        if df.empty:
            raise ValueError("Loaded dataframe is empty")

        return df
