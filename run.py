"""
Run Quant-ML-BTC — loads config, manages experiment logging, and executes pipeline.
"""

import argparse
import sys
from pathlib import Path

# Allow relative imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.experiment_logger import create_run_dir, save_config_copy
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Quant-ML-BTC Runner")
    parser.add_argument(
        "--config",
        default="configs/config_default.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # === Load config and initialize run directory ===
    cfg = load_config(args.config)
    run_dir = create_run_dir()
    save_config_copy(args.config, run_dir)

    # === Execute pipeline ===
    result = run_pipeline(cfg, run_dir=run_dir)

    # === Display summary ===
    print("\n✅ Quant-ML-BTC pipeline finished successfully")
    print(f"📊 Final equity: {float(result['equity'].iloc[-1]):,.2f}")
    print(f"📁 Run directory: {run_dir}")


if __name__ == "__main__":
    main()
