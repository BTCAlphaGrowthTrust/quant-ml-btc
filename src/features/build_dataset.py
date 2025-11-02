"""
Universal Dataset Builder for Quant-ML-BTC
-----------------------------------------
Builds ML-ready feature datasets from any registered strategy.

Usage:
    python -m src.features.build_dataset --strategy tom_makin_1w_osc_4h_pa
    python -m src.features.build_dataset --strategy tom_makin_1m_osc_4h_pa

Outputs:
    data/datasets/<strategy_name>_v1.parquet
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import get_btc_data
from src.strategies import STRATEGY_REGISTRY


def build_features(df: pd.DataFrame, trades: list, strategy_name: str) -> pd.DataFrame:
    """Construct minimal features aligned to oscillator + PA logic (auto-detect timeframe)."""
    df = df.copy()

    # === Base candle features ===
    df["body"] = (df["close"] - df["open"]) / df["close"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    df["true_range_norm"] = (df["high"] - df["low"]) / df["close"]
    if "atr" in df.columns:
        df["atr_norm"] = df["atr"] / df["close"]

    # === Oscillator feature auto-detection ===
    if "k_week" in df.columns and "d_week" in df.columns:
        k_col, d_col, bias_col = "k_week", "d_week", "weekly_bullish"
    elif "k_month" in df.columns and "d_month" in df.columns:
        k_col, d_col, bias_col = "k_month", "d_month", "monthly_bullish"
    else:
        k_col, d_col, bias_col = None, None, None

    if k_col and d_col:
        df["kd_spread"] = df[k_col] - df[d_col]
        df["k_below_thr"] = (df[k_col] < 60).astype(int)
        df["d_below_thr"] = (df[d_col] < 60).astype(int)
        df["bullish_int"] = df.get(bias_col, False).fillna(False).astype(int)
    else:
        # fallback for non-oscillator strategies
        df["kd_spread"] = 0
        df["k_below_thr"] = 0
        df["d_below_thr"] = 0
        df["bullish_int"] = 0

    # === Temporal features ===
    dt = df.index
    df["dow"] = dt.dayofweek
    df["hour_bin"] = (dt.hour // 4) * 4

    # === Label creation (trade entries) ===
    df["y"] = 0
    for t in trades:
        if "entry_time" in t:
            idx = df.index.get_indexer([t["entry_time"]], method="nearest")
            if idx.size > 0 and idx[0] >= 0:
                df.iloc[idx[0], df.columns.get_loc("y")] = 1

    # === Regime tagging ===
    sma200 = df["close"].rolling(200).mean()
    df["regime"] = np.select(
        [(df["close"] > sma200), (df["close"] < sma200)],
        ["bull", "bear"],
        default="sideways",
    )

    # === Sample weighting (recency + regime bias) ===
    days_since = (df.index[-1] - df.index).days
    w_time = np.exp(-np.log(2) * days_since / 730)  # 2-year half-life decay
    w_regime = df["regime"].map({"bull": 1.0, "sideways": 1.15, "bear": 1.3}).fillna(1.0)
    df["weight"] = w_time * w_regime

    # Clean up
    df = df.dropna(subset=["close"]).copy()

    print(f"✅ Features built for {strategy_name}: {len(df)} rows | positives: {df['y'].sum()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build ML dataset from any registered strategy.")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy key from STRATEGY_REGISTRY (e.g. tom_makin_1m_osc_4h_pa)",
    )
    args = parser.parse_args()
    strategy_name = args.strategy

    if strategy_name not in STRATEGY_REGISTRY:
        print(f"❌ Unknown strategy: {strategy_name}")
        print(f"Available: {list(STRATEGY_REGISTRY.keys())}")
        return

    print(f"🚀 Building ML dataset for {strategy_name}")

    # 1️⃣ Load BTC 4H data
    df = get_btc_data()
    df.index = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["close"])
    print(f"📂 Loaded BTC 4H data → {len(df)} rows")

    # 2️⃣ Run selected strategy to generate trades and indicators
    StratClass = STRATEGY_REGISTRY[strategy_name]
    strat = StratClass()
    _ = strat.generate_signals(df)
    trades = strat.trades
    print(f"✅ Strategy generated {len(trades)} trades")

    # 3️⃣ Build ML features
    dataset = build_features(strat.df_with_indicators, trades, strategy_name)

    # 4️⃣ Save dataset
    out_path = Path(f"data/datasets/{strategy_name}_v1.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=True)
    print(f"💾 Saved → {out_path.resolve()}")

    # 5️⃣ Preview
    print(dataset.head(5).to_string())


if __name__ == "__main__":
    main()
