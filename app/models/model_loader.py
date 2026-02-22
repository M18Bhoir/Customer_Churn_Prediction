import joblib
from pathlib import Path
from typing import Optional, Tuple, Any
from datetime import datetime
import re

from app.core.config import settings
from app.core.logger import logger


class ModelLoader:
    """Class to load and manage ML models"""
    
    def __init__(self):
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.model_version: Optional[str] = None
        self.model_loaded_at: Optional[datetime] = None
        self.model_path: Optional[Path] = None
    
    def load_latest_model(self) -> bool:
        """
        Load the latest model from model registry
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            registry_path = Path(settings.model.MODEL_REGISTRY_PATH)
            
            if not registry_path.exists():
                logger.error(f"Model registry path does not exist: {registry_path}")
                return False
            
                        # Find folders matching pattern model_YYYYMMDD_HHMMSS
            model_folders = [
                f for f in registry_path.iterdir()
                if f.is_dir() and re.match(r"model_\d{8}_\d{6}", f.name)
            ]

            if not model_folders:
                logger.error("No model folders found in registry")
                return False

            # Sort folders by timestamp extracted from name
            def extract_timestamp(folder: Path):
                match = re.search(r"model_(\d{8}_\d{6})", folder.name)
                return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")

            latest_folder = max(model_folders, key=extract_timestamp)

            logger.info(f"Loading latest model folder: {latest_folder}")
            
            # Load the model
            latest_model_file = latest_folder / "model.pkl"
            logger.info(f"Loading model file: {latest_model_file}")
            self.model = joblib.load(latest_model_file)
            self.model_path = latest_model_file
            
            # Extract version from filename (e.g., churn_model_v1.0.0.joblib -> v1.0.0)
            version_match = '_'.join(str(latest_folder).split("\\")[-1].split("_")[-2:])  # Get timestamp part
            self.model_version = version_match if version_match else "unknown"
            
            self.model_loaded_at = datetime.now()
            
            logger.info(f"Model loaded successfully: version={self.model_version}, "
                       f"type={type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {
                "status": "not_loaded",
                "version": None,
                "model_type": None,
                "loaded_at": None
            }
        
        return {
            "status": "loaded",
            "version": self.model_version,
            "model_type": type(self.model).__name__,
            "loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None,
            "model_file": str(self.model_path) if self.model_path else None
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


# Global model loader instance
model_loader = ModelLoader()