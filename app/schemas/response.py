from pydantic import BaseModel, Field
from typing import Optional


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="API health status")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "app_name": "Customer Churn Prediction API",
                "version": "1.0.0",
                "model_loaded": True
            }
        }


class ModelVersionResponse(BaseModel):
    """Model version response schema"""
    status: str = Field(..., description="Model loading status")
    version: Optional[str] = Field(None, description="Model version")
    model_type: Optional[str] = Field(None, description="Type of ML model")
    loaded_at: Optional[str] = Field(None, description="Timestamp when model was loaded")
    model_file: Optional[str] = Field(None, description="Path to model file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "loaded",
                "version": "v1.0.0",
                "model_type": "RandomForestClassifier",
                "loaded_at": "2024-01-15T10:30:00",
                "model_file": "model_registry/churn_model_v1.0.0.joblib"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    customer_id: int = Field(..., description="Customer identifier")
    churn_prediction: int = Field(..., description="Churn prediction (0 or 1)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: str = Field(..., description="Risk level category")
    model_version: Optional[str] = Field(None, description="Version of model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": 12345,
                "churn_prediction": 1,
                "churn_probability": 0.78,
                "risk_level": "High",
                "model_version": "v1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")