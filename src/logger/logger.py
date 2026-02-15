"""
Central logging configuration for src modules.

Features:
- Console + file logging
- Rotating logs
- Standardized format
- Singleton logger creation
- Safe import across modules
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ==========================================================
# LOG CONFIGURATION
# ==========================================================


LOG_FILE = "src/logger/training_pipeline.log"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ==========================================================
# LOGGER FACTORY
# ==========================================================
def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Ensures:
    - No duplicate handlers
    - Consistent logging format
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    # ------------------------
    # Console Handler
    # ------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # ------------------------
    # File Handler (Rotating)
    # ------------------------
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger
