from pathlib import Path
from pydantic import BaseModel
import yaml


class ApplicationSettings(BaseModel):
    APP_NAME: str
    APP_VERSION: str
    DEBUG: bool


class APISettings(BaseModel):
    API_PREFIX: str
    HOST: str
    PORT: int


class ModelSettings(BaseModel):
    MODEL_REGISTRY_PATH: str


class LoggingSettings(BaseModel):
    LOG_LEVEL: str
    LOG_FILE: str
    LOG_FORMAT: str


class PredictionSettings(BaseModel):
    PREDICTION_THRESHOLD: float


class Settings(BaseModel):
    application: ApplicationSettings
    api: APISettings
    model: ModelSettings
    logging: LoggingSettings
    prediction: PredictionSettings


# ----------------------------
# Load YAML Config
# ----------------------------

def load_settings() -> Settings:
    config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    return Settings(**config_data)


# Global settings instance
settings = load_settings()