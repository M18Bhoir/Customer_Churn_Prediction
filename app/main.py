from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logger import logger
from app.api.routes import router
from app.models.model_loader import model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup: Load model
    logger.info("="*60)
    logger.info(f"Starting {settings.application.APP_NAME} v{settings.application.APP_VERSION}")
    logger.info("="*60)
    
    logger.info("Loading model from registry...")
    model_loaded = model_loader.load_latest_model()
    
    if model_loaded:
        logger.info("Model loaded successfully")
        model_info = model_loader.get_model_info()
        logger.info(f"  - Version: {model_info['version']}")
        logger.info(f"  - Type: {model_info['model_type']}")
        logger.info(f"  - File: {model_info['model_file']}")
    else:
        logger.warning("✗ Failed to load model - predictions will not be available")
    
    logger.info("="*60)
    logger.info(f"API is ready at http://{settings.api.HOST}:{settings.api.PORT}")
    logger.info(f"Documentation available at http://{settings.api.HOST}:{settings.api.PORT}/docs")
    logger.info("="*60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title=settings.application.APP_NAME,
    description="ML-powered API for predicting customer churn",
    version=settings.application.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix=settings.api.API_PREFIX)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.application.APP_NAME}",
        "version": settings.application.APP_VERSION,
        "docs": f"{settings.api.API_PREFIX}/docs",
        "health": f"{settings.api.API_PREFIX}/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api.HOST,
        port=settings.api.PORT,
        reload=settings.application.DEBUG,
        log_level=settings.logging.LOG_LEVEL.lower()
    )