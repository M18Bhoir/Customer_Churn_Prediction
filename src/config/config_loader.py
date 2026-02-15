import yaml
from pathlib import Path


class ConfigLoader:

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

    def get_config(self):
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
