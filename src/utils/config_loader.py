"""
config_loader.py
Utility to load YAML configuration files.
"""
import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
