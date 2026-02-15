from fastapi import APIRouter

router = APIRouter()

@router.post("/predict")
def predict():
    return {"prediction": "placeholder"}
