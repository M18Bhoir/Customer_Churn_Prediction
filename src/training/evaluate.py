import json
import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.logger.logger import get_logger
from src.schema.evaluate_schema import EvaluationResult


logger = get_logger(__name__)


class Evaluator:
    """
    Responsible for evaluation metrics calculation
    and persistence of evaluation reports.
    """

    def __init__(self):
        logger.info("Evaluator initialized")

    def evaluate(self, y_true, y_pred):

        logger.info("Calculating evaluation metrics")

        metrics = EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred)),
            recall=float(recall_score(y_true, y_pred)),
            f1_score=float(f1_score(y_true, y_pred)),
            roc_auc=float(roc_auc_score(y_true, y_pred)),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist()
        )

        logger.info(f"Evaluation completed: {metrics.dict()}")
        return metrics

    def save_report(self, metrics: EvaluationResult, version_path: str):

        report_path = os.path.join(version_path, "evaluation.json")

        with open(report_path, "w") as f:
            json.dump(metrics.dict(), f, indent=4)

        logger.info(f"Evaluation report saved at {report_path}")
