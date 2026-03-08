import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.logger.logger import get_logger
from src.schema.train_schema import TrainConfig
from src.training.evaluate import Evaluator
from src.features.transformers import (
    ColumnExcluder,
    MissingValueImputer,
    CategoricalLabelEncoder,
    NumericalTransformer,
)
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


logger = get_logger(__name__)


class Trainer:
    """
    Industrial-grade Trainer with unified preprocessing + model pipeline, SMOTE, and XGBoost support
    """

    def __init__(self, config: dict):

        self.config = TrainConfig(**config["training"])
        self.registry_path = self.config.model_registry_path
        os.makedirs(self.registry_path, exist_ok=True)

        logger.info("Trainer initialized successfully")

    # =========================================================
    # Unified Pipeline Builder (Preprocessing + Model)
    # =========================================================
    def _build_pipeline(self) -> Pipeline:
        """
        Build a unified preprocessing + model pipeline
        """

        # Preprocessing steps
        preprocessing_steps = [
            ("column_excluder", ColumnExcluder(exclude_columns=self.config.exclude_columns)),
            ("imputer", MissingValueImputer()),
            ("categorical_encoder", CategoricalLabelEncoder()),
            ("numerical_transformer", NumericalTransformer())
        ]

        # Select model based on config
        model_type = self.config.model.type
        model_config = self.config.model

        if model_type == "random_forest":
            rf_params = model_config.random_forest.dict()
            model = RandomForestClassifier(
                **rf_params,
                random_state=self.config.random_state
            )

        elif model_type == "logistic_regression":
            lr_params = model_config.logistic_regression.dict()
            model = LogisticRegression(
                **lr_params,
                random_state=self.config.random_state
            )

        elif model_type == "xgboost":
            xgb_params = model_config.xgboost.dict()
            xgb_params.pop('random_state', None)
            model = XGBClassifier(
                **xgb_params,
                random_state=self.config.random_state,
                eval_metric='logloss'
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Using model: {model_type}")

        # Add model to pipeline
        preprocessing_steps.append(("model", model))

        # Create unified pipeline
        pipeline = Pipeline(steps=preprocessing_steps)

        return pipeline

    # =========================================================
    # Extract Feature Names
    # =========================================================
    def _get_feature_names(self, pipeline: Pipeline, X_sample: pd.DataFrame) -> list:
        """
        Get feature names after preprocessing by fitting on sample
        """
        try:
            # Apply all transformers except the model
            X_transformed = X_sample.copy()
            
            for name, transformer in pipeline.named_steps.items():
                if name != "model":
                    X_transformed = transformer.transform(X_transformed)
            
            return X_transformed.columns.tolist() if isinstance(X_transformed, pd.DataFrame) else [f"feature_{i}" for i in range(X_transformed.shape[1])]
        except:
            return [f"feature_{i}" for i in range(100)]

    # =========================================================
    # Metadata Extraction
    # =========================================================
    def _extract_model_metadata(
        self,
        pipeline: Pipeline,
        feature_names: list
    ) -> Dict[str, Any]:

        model = pipeline.named_steps["model"]

        metadata = {
            "model_type": model.__class__.__name__,
            "training_timestamp": datetime.now().isoformat(),
            "random_state": self.config.random_state,
            "test_size": self.config.test_size,
            "cross_validation_folds": self.config.cross_validation_folds,
            "n_features": len(feature_names),
            "features": feature_names,
            "hyperparameters": model.get_params(),
            "smote_applied": self.config.apply_smote,
            "excluded_columns": self.config.exclude_columns,
            "preprocessing_pipeline": True,
        }

        if hasattr(model, "feature_importances_"):
            metadata["feature_importance"] = dict(
                zip(feature_names, model.feature_importances_.tolist())
            )

        elif hasattr(model, "coef_"):
            try:
                metadata["coefficients"] = dict(
                    zip(feature_names, model.coef_[0].tolist())
                )
            except:
                pass

        return metadata

    # =========================================================
    # Cross Validation
    # =========================================================
    def model_cross_validation(self, pipeline, X, y) -> Dict[str, float]:

        skf = StratifiedKFold(
            n_splits=self.config.cross_validation_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        acc_scores = []
        f1_scores = []
        auc_scores = []

        folds = list(skf.split(X, y))

        for fold_idx, (train_idx, val_idx) in enumerate(
            tqdm(folds, desc="CV Progress", total=len(folds))
        ):

            logger.info("Training fold %d", fold_idx + 1)

            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

            # Apply SMOTE to training fold if enabled
            if self.config.apply_smote:
                logger.info("Applying SMOTE to fold training data")
                smote = SMOTE(random_state=self.config.random_state)
                X_train_array = X_train.values
                X_train_smote, y_train = smote.fit_resample(X_train_array, y_train)
                X_train = pd.DataFrame(X_train_smote, columns=X_train.columns)

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_val)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds)

            acc_scores.append(acc)
            f1_scores.append(f1)

            # ROC-AUC if available
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, probs)
                auc_scores.append(auc)

        metrics = {
            "cv_accuracy_mean": float(np.mean(acc_scores)),
            "cv_accuracy_std": float(np.std(acc_scores)),
            "cv_f1_mean": float(np.mean(f1_scores)),
            "cv_f1_std": float(np.std(f1_scores)),
        }

        if auc_scores:
            metrics["cv_auc_mean"] = float(np.mean(auc_scores))
            metrics["cv_auc_std"] = float(np.std(auc_scores))

        return metrics
    
    # =========================================================
    # Training Execution
    # =========================================================
    def train(self, df: pd.DataFrame) -> Tuple[str, Dict]:

        logger.info("Training started")

        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_column}' not found"
            )

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        # -----------------------------------------------------
        # Train-Test Split
        # -----------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(y.unique()) <= 10 else None
        )

        logger.info("Train-test split completed")

        # Apply SMOTE to training data if enabled
        if self.config.apply_smote:
            logger.info("Applying SMOTE to training data")
            smote = SMOTE(random_state=self.config.random_state)
            X_train_array = X_train.values
            X_train_smote, y_train = smote.fit_resample(X_train_array, y_train)
            X_train = pd.DataFrame(X_train_smote, columns=X_train.columns)
            logger.info(f"SMOTE applied. New training set size: {X_train.shape}")

        # Build unified pipeline
        pipeline = self._build_pipeline()

        # -----------------------------------------------------
        # Stratified K-Fold Cross Validation
        # -----------------------------------------------------
        logger.info("Starting Stratified K-Fold Cross Validation")
        cv_metrics = self.model_cross_validation(
            pipeline,
            X_train,
            y_train
        )
        logger.info("Cross validation completed: %s", cv_metrics)


        # -----------------------------------------------------
        # Final Training on Full Training Data
        # -----------------------------------------------------
        logger.info("Final model training on complete training set")
        pipeline.fit(X_train, y_train)

        logger.info("Final model training completed")

        # -----------------------------------------------------
        # Evaluation on Hold-out Test Set
        # -----------------------------------------------------
        evaluator = Evaluator()

        predictions = pipeline.predict(X_test)

        metrics = evaluator.evaluate(y_test, predictions)

        # -----------------------------------------------------
        # Versioned Model Saving
        # -----------------------------------------------------
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_path = os.path.join(
            self.registry_path,
            f"model_{version}"
        )

        os.makedirs(version_path, exist_ok=True)

        # Save complete unified pipeline as .pkl
        model_path = os.path.join(version_path, "complete_pipeline.pkl")
        joblib.dump(pipeline, model_path)

        logger.info(f"Complete unified pipeline saved at {model_path}")

        # Save predictions
        predictions_df = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": predictions
        })
        predictions_df.to_csv(
            os.path.join(version_path, "predictions.csv"),
            index=False
        )

        evaluator.save_report(metrics, version_path)

        # -----------------------------------------------------
        # Metadata Saving
        # -----------------------------------------------------
        feature_names = self._get_feature_names(pipeline, X_train.head())

        metadata = self._extract_model_metadata(
            pipeline,
            feature_names
        )

        metadata["test_metrics"] = metrics.dict()
        metadata["cross_validation_metrics"] = cv_metrics

        metadata_path = os.path.join(version_path, "model.yaml")

        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, sort_keys=False)

        logger.info(f"Model metadata saved at {metadata_path}")
        logger.info("Training pipeline completed successfully")

        return model_path, metrics


