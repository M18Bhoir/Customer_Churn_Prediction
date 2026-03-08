import pandas as pd
import numpy as np
from typing import Tuple

from app.core.config import settings
from app.core.logger import logger
from app.models.model_loader import model_loader
from app.schemas.request import CustomerData
from app.schemas.response import PredictionResponse
from app.services.preprocessing_service import preprocessing_service


class PredictionService:
    """Service for handling prediction logic with unified preprocessing + model pipeline"""
    
    @staticmethod
    def _determine_risk_level(probability: float) -> str:
        """
        Determine risk level based on churn probability
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            Risk level string
        """
        if probability >= 0.7:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def _preprocess_input(customer: CustomerData) -> Tuple[pd.DataFrame, int]:
        """
        Convert customer data to DataFrame for model input
        
        Args:
            customer: Customer data
            
        Returns:
            Tuple of (DataFrame with customer features, customer_id)
        """
        # Convert to dict and create DataFrame
        customer_dict = customer.model_dump()
        customer_id = customer_dict.pop('customer_id')
        
        df = pd.DataFrame([customer_dict])
        
        logger.debug(f"Preprocessed input for customer {customer_id}: shape={df.shape}")
        
        return df, customer_id
    
    @staticmethod
    def predict(customer: CustomerData) -> PredictionResponse:
        """
        Make churn prediction for a customer using unified pipeline
        
        The preprocessing pipeline handles:
        1. Column exclusion (customer_id, age, monthly_charges)
        2. Missing value imputation
        3. Categorical label encoding
        4. Numerical transformation (scaling)
        5. Model prediction (XGBoost)
        
        Args:
            customer: Customer data
            
        Returns:
            Prediction response
            
        Raises:
            ValueError: If model is not loaded
            Exception: If prediction fails
        """
        if not model_loader.is_loaded():
            logger.error("Prediction attempted but model is not loaded")
            raise ValueError("Model is not loaded. Please check model registry.")
        
        try:
            # Convert input to DataFrame
            df, customer_id = PredictionService._preprocess_input(customer)
            
            logger.debug(f"Input DataFrame shape: {df.shape}, Columns: {df.columns.tolist()}")
            
            # Apply preprocessing using unified pipeline
            if model_loader.is_complete_pipeline:
                logger.debug("Using complete unified pipeline (preprocessing + model)")
                # The pipeline handles all preprocessing internally
                prediction = model_loader.pipeline.predict(df)[0]
                
                if hasattr(model_loader.pipeline, 'predict_proba'):
                    probability = model_loader.pipeline.predict_proba(df)[0][1]
                else:
                    probability = float(prediction)
            else:
                logger.debug("Using preprocessing service + model")
                # Apply preprocessing pipeline
                X = preprocessing_service.preprocess(df)
                
                # Use the model for prediction
                prediction = model_loader.model.predict(X)[0]
                
                if hasattr(model_loader.model, 'predict_proba'):
                    probability = model_loader.model.predict_proba(X)[0][1]
                else:
                    probability = float(prediction)
            
            # Determine risk level
            risk_level = PredictionService._determine_risk_level(probability)
            
            logger.info(f"Prediction for customer {customer_id}: "
                       f"churn={prediction}, probability={probability:.4f}, risk={risk_level}")
            
            return PredictionResponse(
                customer_id=customer_id,
                churn_prediction=int(prediction),
                churn_probability=float(probability),
                risk_level=risk_level,
                model_version=model_loader.model_version
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for customer {customer.customer_id}: {str(e)}", 
                        exc_info=True)
            raise Exception(f"Prediction failed: {str(e)}")


# Global service instance
prediction_service = PredictionService()

