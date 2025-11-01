"""
Parameter sweep runner for BTC | 1W Osc + 4H PA Execution.
Systematically tests parameter combinations and records performance metrics
(win rate, profit factor, Sharpe, etc.) for ML meta-model training.

Usage:
    python -m src.optimization.param_sweep
"""

from pathlib import Path
from itertools import product
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.strategies.weekly_oscillator_pa import WeeklyOscillatorPA
from src.data_loader import get_btc_data


# ---------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------
def sharpe_from_pnl(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    return float(np.mean(pnl) / (np.std(pnl, ddof=1) + 1e-12))


def profit_factor_from_pnl(pnl: np.ndarray) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return float(gains / (losses + 1e-12))


# ---------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------
def run_param_sweep():
    print("🚀 Running parameter sweep for BTC | 1W Osc + 4H PA Execution")

    # === 1️⃣ Define grid ===
    param_grid = {
        "stoch_threshold": [40, 50, 60, 70, 80],
        "lookback_bars": [5, 10, 20],
        "atr_multiplier": [0.5, 1.0, 1.5, 2.0],
    }

    combos = list(product(
        param_grid["stoch_threshold"],
        param_grid["lookback_bars"],
        param_grid["atr_multiplier"],
    ))
    print(f"🧮 Total combinations: {len(combos)}")

    # === 2️⃣ Prepare data ===
    df = get_btc_data()
    df.index = pd.to_datetime(df["timestamp"], utc=True)

    # === 3️⃣ Sweep ===
    rows = []
    for i, (stoch_threshold, lookback_bars, atr_multiplier) in enumerate(
        tqdm(combos, desc="Parameter combos")
    ):
        strat = WeeklyOscillatorPA(
            stoch_threshold=stoch_threshold,
            lookback_bars=lookback_bars,
            atr_multiplier=atr_multiplier,
            make_plot=False,
            save_trades_csv=False,
        )

        _ = strat.generate_signals(df)
        trades = pd.DataFrame(strat.trades)

        if trades.empty:
            rows.append({
                "run_id": i + 1,
                "stoch_threshold": stoch_threshold,
                "lookback_bars": lookback_bars,
                "atr_multiplier": atr_multiplier,
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "gross_pnl": 0.0,
            })
            continue

        pnl = trades["pnl"].dropna().values
        wins = (pnl > 0).sum()
        losses = (pnl < 0).sum()
        wr = wins / max(wins + losses, 1)
        pf = profit_factor_from_pnl(pnl)
        shp = sharpe_from_pnl(pnl)

        rows.append({
            "run_id": i + 1,
            "stoch_threshold": stoch_threshold,
            "lookback_bars": lookback_bars,
            "atr_multiplier": atr_multiplier,
            "trades": len(pnl),
            "win_rate": wr,
            "profit_factor": pf,
            "sharpe": shp,
            "gross_pnl": pnl.sum(),
        })

    # === 4️⃣ Save results ===
    results = pd.DataFrame(rows)
    out_dir = Path("data/param_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"btc_osc_pa_sweep_{ts}.csv"
    results.to_csv(out_path, index=False)

    print(f"\n💾 Saved sweep results → {out_path.resolve()}")
    print(results.describe().T)

    # === 5️⃣ Quick top-5 preview ===
    top = results.sort_values("sharpe", ascending=False).head(5)
    print("\n🏆 Top-5 Sharpe configurations:")
    print(top[["stoch_threshold", "lookback_bars", "atr_multiplier", "sharpe", "profit_factor", "win_rate"]])

    print("\n✅ Parameter sweep complete.")
    return results


if __name__ == "__main__":
    run_param_sweep()
