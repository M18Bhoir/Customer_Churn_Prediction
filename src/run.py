import sys
import traceback
import pandas as pd
from pathlib import Path
import os

SRC_ROOT = Path(__file__).resolve().parent.parent
os.chdir(SRC_ROOT)


from src.logger.logger import get_logger
from src.config.config_loader import ConfigLoader

from src.data.load_data import DataLoader
from src.data.validate_data import DataValidator
from src.features.build_features import FeatureBuilder

from src.training.train import Trainer


logger = get_logger(__name__)
logger.info("current working directory: " + str(Path.cwd()))
CONFIG_PATH = "src/config/config.yaml"



class PipelineRunner:
    """
    Main pipeline orchestrator.

    Executes:
        1. Load data
        2. Validate data
        3. Feature engineering
        4. Train model
        5. Predict
        6. Evaluate
        7. Save artifacts
    """

    def __init__(self, config_path: str):

        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()

        logger.info("PipelineRunner initialized")

    # ---------------------------------------------------------
    # Pipeline Execution
    # ---------------------------------------------------------
    def run(self):

        try:
            logger.info("========== PIPELINE STARTED ==========")

            # =====================================================
            # STEP 1 — LOAD DATA
            # =====================================================
            logger.info("STEP 1: Loading raw data")

            data_loader = DataLoader(self.config)
            raw_df = data_loader.load()

            logger.info(f"Raw data shape: {raw_df.shape}")

            # =====================================================
            # STEP 2 — VALIDATE DATA
            # =====================================================
            logger.info("STEP 2: Validating data")

            validator = DataValidator(self.config)
            validator.validate(raw_df)

            # =====================================================
            # STEP 3 — BUILD FEATURES
            # =====================================================
            logger.info("STEP 3: Feature engineering")

            feature_builder = FeatureBuilder(self.config)
            processed_df = feature_builder.build(raw_df)

            logger.info(
                f"Processed data shape: {processed_df.shape}"
            )

            # =====================================================
            # STEP 4 — TRAIN MODEL
            # =====================================================
            logger.info("STEP 4: Training model")

            trainer = Trainer(self.config)
            model_path, _ = trainer.train(processed_df)

            logger.info(f"Model saved at: {model_path}")

            logger.info("========== PIPELINE COMPLETED ==========")

        except Exception as e:
            logger.error("Pipeline execution failed")
            logger.error(str(e))
            traceback.print_exc()
            sys.exit(1)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    runner = PipelineRunner(CONFIG_PATH)
    runner.run()
