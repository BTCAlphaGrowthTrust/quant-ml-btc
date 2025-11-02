"""
Batch pipeline runner for Quant-ML-BTC
--------------------------------------
Runs the complete dataset build + ML training pipeline for all active strategies.

Usage:
    python -m src.pipeline.run_all_strategies
"""

import subprocess
from pathlib import Path
from datetime import datetime
from src.strategies import STRATEGY_REGISTRY

# --- Which strategies to include ---
ACTIVE_STRATEGIES = [
    "tom_makin_1w_osc_4h_pa",
    "tom_makin_1m_osc_4h_pa",
    "tom_makin_1d_golden_4h_pa",
    "tom_makin_4h_golden_4h_pa",
]

# --- Output folder for logs ---
LOG_DIR = Path("data/pipeline_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"run_all_{timestamp}.log"

# --- Helpers ---------------------------------------------------------
def run_step(cmd: list[str], title: str):
    """Run a subprocess command and append its output to the global log."""
    print(f"\n🚀 {title}\n{'='*len(title)}")
    with LOG_FILE.open("a") as log:
        log.write(f"\n\n==== {title} ====\n")
        result = subprocess.run(cmd, text=True, capture_output=True)
        log.write(result.stdout)
        log.write(result.stderr)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Step failed ({title}). Check {LOG_FILE}")
            exit(result.returncode)
    print(f"✅ {title} complete.\n")

# --- Main ------------------------------------------------------------
def main():
    print(f"🧠 Quant-ML-BTC Multi-Strategy Runner")
    print(f"🗓️  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🪶 Log file → {LOG_FILE.resolve()}\n")

    for strat_key in ACTIVE_STRATEGIES:
        if strat_key not in STRATEGY_REGISTRY:
            print(f"⚠️  Skipping {strat_key} — not in STRATEGY_REGISTRY.")
            continue

        print(f"⚙️  Running pipeline for: {strat_key}")
        dataset_name = f"{strat_key}_v1"

        # Step 1: build dataset
        run_step(
            ["python", "-m", "src.features.build_dataset", "--strategy", strat_key],
            f"Building dataset for {strat_key}"
        )

        # Step 2: train ML model
        run_step(
            ["python", "-m", "src.ml.train", "--dataset", dataset_name],
            f"Training ML model for {strat_key}"
        )

    print("\n🎯 All strategies completed successfully!")
    print(f"🪶 Log saved → {LOG_FILE.resolve()}")

# --- Entrypoint ------------------------------------------------------
if __name__ == "__main__":
    main()
