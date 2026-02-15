from pydantic import BaseModel
from typing import Dict, List


class EvaluationResult(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
