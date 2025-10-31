"""
metrics_reporter.py
Calculates Sharpe, Sortino, Profit Factor, Win Rate, and Max Drawdown.
"""

import pandas as pd
import numpy as np


def calc_metrics(equity: pd.Series) -> dict:
    """Compute risk/return statistics from an equity curve."""
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {}

    # Annualization factor ≈ 6 bars/day × 252 days/year
    ann_factor = np.sqrt(252 * 6)

    sharpe = ann_factor * ret.mean() / (ret.std() + 1e-9)
    downside = ret[ret < 0]
    sortino = ann_factor * ret.mean() / (downside.std() + 1e-9)

    gains = ret[ret > 0].sum()
    losses = -ret[ret < 0].sum()
    profit_factor = gains / (losses + 1e-9)
    win_rate = (ret > 0).mean()

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_dd = drawdown.min()

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "final_equity": equity.iloc[-1],
    }


def export_metrics(equity: pd.Series, path: str = "results/metrics.csv"):
    """Write metrics to CSV and print summary."""
    metrics = calc_metrics(equity)
    if not metrics:
        print("⚠️ No returns data for metrics.")
        return metrics

    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"✅ Saved metrics → {path}")
    for k, v in metrics.items():
        print(f"  {k:<15} {v:>10.4f}")
    return metrics
