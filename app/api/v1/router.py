from fastapi import APIRouter
from app.api.v1.endpoints import predict, health

router = APIRouter()
router.include_router(predict.router, tags=["Prediction"])
router.include_router(health.router, tags=["Health"])
