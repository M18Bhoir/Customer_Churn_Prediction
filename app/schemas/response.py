from pydantic import BaseModel

class PredictionResponse(BaseModel):
    churn_probability: float
