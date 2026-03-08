import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from app.core.logger import logger
from app.features.transformers import (
    ColumnExcluder,
    MissingValueImputer,
    CategoricalLabelEncoder,
    NumericalTransformer,
)


class PreprocessingService:
    """Service for preprocessing customer data for predictions using unified pipeline"""
    
    # Columns to exclude from predictions
    EXCLUDE_COLUMNS = ["customer_id", "age", "monthly_charges"]
    
    def __init__(self):
        """Initialize the preprocessing pipeline"""
        self.pipeline = Pipeline(
            steps=[
                ("column_excluder", ColumnExcluder(exclude_columns=self.EXCLUDE_COLUMNS)),
                ("imputer", MissingValueImputer()),
                ("categorical_encoder", CategoricalLabelEncoder()),
                ("numerical_transformer", NumericalTransformer())
            ]
        )
        logger.debug("PreprocessingService initialized with unified pipeline")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete preprocessing pipeline
        
        Args:
            data: Input dataframe
            
        Returns:
            Preprocessed dataframe ready for model prediction
        """
        logger.debug(f"Starting preprocessing pipeline. Input shape: {data.shape}")
        
        try:
            processed_data = self.pipeline.fit_transform(data)
            logger.debug(f"Preprocessing completed. Output shape: {processed_data.shape}")
            return processed_data
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            raise


# Global service instance
preprocessing_service = PreprocessingService()

