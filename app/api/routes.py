from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.logger import logger
from app.models.model_loader import model_loader
from app.services.prediction_service import prediction_service
from app.schemas.request import CustomerData
from app.schemas.response import (
    HealthCheckResponse,
    ModelVersionResponse,
    PredictionResponse,
    ErrorResponse
)

# Create API router
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API is running and healthy",
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint
    
    Returns health status of the API and model loading status
    """
    logger.info("Health check endpoint called")
    
    return HealthCheckResponse(
        status="healthy",
        app_name=settings.application.APP_NAME,
        version=settings.application.APP_VERSION,
        model_loaded=model_loader.is_loaded()
    )


@router.get(
    "/model/version",
    response_model=ModelVersionResponse,
    summary="Get Model Version",
    description="Get information about the currently loaded model",
    tags=["Model"]
)
async def get_model_version():
    """
    Get model version and information
    
    Returns details about the currently loaded model including version,
    type, and when it was loaded
    """
    logger.info("Model version endpoint called")
    
    model_info = model_loader.get_model_info()
    
    return ModelVersionResponse(
        status=model_info["status"],
        version=model_info["version"],
        model_type=model_info["model_type"],
        loaded_at=model_info["loaded_at"],
        model_file=model_info["model_file"]
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Churn",
    description="Predict customer churn probability",
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict(customer: CustomerData):
    """
    Predict churn for a customer
    
    Takes customer data as input and returns churn prediction with
    probability and risk level
    
    Args:
        customer: Customer data with all required features
        
    Returns:
        Prediction response with churn probability and risk level
    """
    logger.info(f"Prediction request for customer {customer.customer_id}")
    
    try:
        # Check if model is loaded
        if not model_loader.is_loaded():
            logger.error("Prediction attempted but model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded. Service unavailable."
            )
        
        # Make prediction
        prediction = prediction_service.predict(customer)
        
        logger.info(f"Prediction successful for customer {customer.customer_id}")
        
        return prediction
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )