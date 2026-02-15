from pydantic import BaseModel
from typing import List


class DataConfig(BaseModel):
    input_path: str
    processed_path: str
    required_columns: List[str]
    target_column: str
    allowed_missing_pct: float = 0.2
