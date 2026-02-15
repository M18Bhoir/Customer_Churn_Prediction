from pydantic import BaseModel

class PredictionRequest(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
