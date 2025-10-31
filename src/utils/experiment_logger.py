"""
experiment_logger.py
Handles run directory creation, timestamping, and file management.
"""
import os
from datetime import datetime
import shutil

def create_run_dir(base_dir="results/runs"):
    """Create timestamped run folder (e.g., results/runs/run_2025-10-31_001)."""
    os.makedirs(base_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    existing = [d for d in os.listdir(base_dir) if d.startswith(f"run_{date_str}")]
    run_num = len(existing) + 1
    run_dir = os.path.join(base_dir, f"run_{date_str}_{run_num:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_config_copy(config_path: str, run_dir: str):
    """Copy YAML config into the run directory for record keeping."""
    try:
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    except Exception as e:
        print(f"⚠️ Failed to copy config: {e}")
