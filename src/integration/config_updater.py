"""
Phase 3C — Regime-Adaptive Config Updater
------------------------------------------
Loads the latest regime selector output and updates config_active.yaml
to use the top-ranked parameter set for live runs or backtesting.
"""

from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime


def update_active_config(recommendations_csv: Path, config_path: Path):
    # --- Load best parameters ---
    recs = pd.read_csv(recommendations_csv)
    if recs.empty:
        raise ValueError("No recommendations found in the selector CSV.")
    best = recs.iloc[0]

    stoch_threshold = float(best["stoch_threshold"])
    lookback_bars = int(best["lookback_bars"])
    atr_multiplier = float(best["atr_multiplier"])

    print("🔧 Updating config with best regime parameters:")
    print(f"   Stochastic Threshold: {stoch_threshold}")
    print(f"   Lookback Bars:       {lookback_bars}")
    print(f"   ATR Multiplier:      {atr_multiplier}")

    # --- Load existing config or create new one ---
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    config["strategy"] = config.get("strategy", {})
    config["strategy"].update({
        "stoch_threshold": stoch_threshold,
        "lookback_bars": lookback_bars,
        "atr_multiplier": atr_multiplier,
        "last_update": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": str(recommendations_csv),
    })

    # --- Write back to YAML ---
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"💾 Config updated → {config_path.resolve()}")
    print("✅ Regime-adaptive configuration active.")


def main():
    print("🧠 Phase 3C — Regime-Adaptive Config Updater...")

    # --- Locate latest recommendation file automatically ---
    results_dir = Path("results")
    rec_dirs = sorted(results_dir.glob("regime_selector_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not rec_dirs:
        raise FileNotFoundError("No regime_selector_* directories found. Run regime_selector first.")
    latest_dir = rec_dirs[0]
    csv_path = latest_dir / "regime_recommendations.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    config_path = Path("config_active.yaml")
    update_active_config(csv_path, config_path)


if __name__ == "__main__":
    main()
