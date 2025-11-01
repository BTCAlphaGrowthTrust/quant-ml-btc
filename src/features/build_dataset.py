# src/features/build_dataset.py
"""
Builds the v1 labeled dataset for ML training from Tom Makin's
'1W Osc + 4H PA Execution' strategy.

Outputs:
    data/datasets/btc_osc_pa_v1.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import get_btc_data
from src.strategies.weekly_oscillator_pa import WeeklyOscillatorPA


def build_features(df: pd.DataFrame, trades: list) -> pd.DataFrame:
    """Construct minimal features aligned to your strategy logic."""
    df = df.copy()

    # === Base features ===
    df["body"] = (df["close"] - df["open"]) / df["close"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
    df["true_range_norm"] = (df["high"] - df["low"]) / df["close"]
    df["atr_norm"] = df["atr"] / df["close"]
    df["kd_spread"] = df["k_week"] - df["d_week"]
    df["k_below_thr"] = (df["k_week"] < 60).astype(int)
    df["d_below_thr"] = (df["d_week"] < 60).astype(int)
    df["weekly_bullish_int"] = df["weekly_bullish"].astype(int)

    # === Temporal ===
    dt = df.index
    df["dow"] = dt.dayofweek
    df["hour_bin"] = dt.hour // 4 * 4

    # === Label creation ===
    df["y"] = 0
    for t in trades:
        if "entry_time" in t:
            idx = df.index.get_indexer([t["entry_time"]], method="nearest")
            if idx.size > 0 and idx[0] >= 0:
                df.iloc[idx[0], df.columns.get_loc("y")] = 1

    # === Regime tagging ===
    sma200 = df["close"].rolling(200).mean()
    df["regime"] = np.select(
        [
            (df["weekly_bullish"]) & (df["close"] > sma200),
            (~df["weekly_bullish"]) & (df["close"] < sma200),
        ],
        ["bull", "bear"],
        default="sideways",
    )

    # === Sample weights (recency + regime) ===
    # Half-life weighting toward recent data (~2 years)
    days_since = (df.index[-1] - df.index).days
    w_time = np.exp(-np.log(2) * days_since / 730)

    w_regime = df["regime"].map({"bull": 1.0, "sideways": 1.15, "bear": 1.3}).fillna(1.0)
    df["weight"] = w_time * w_regime

    # Drop obvious NAs from rolling features
    df = df.dropna(subset=["atr_norm", "k_week", "d_week"]).copy()
    return df


def main():
    print("🚀 Building ML dataset for BTC | 1W Osc + 4H PA Execution")

    # 1️⃣ Load 4H BTC data
    df = get_btc_data()
    df.index = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["close"])
    print(f"📂 Loaded BTC 4H data → {len(df)} rows")

    # 2️⃣ Run base strategy to get labels
    strat = WeeklyOscillatorPA(use_forming_week=True, make_plot=False, save_trades_csv=False)
    _ = strat.generate_signals(df)
    trades = strat.trades
    print(f"✅ Strategy generated {len(trades)} trades")

    # 3️⃣ Build features + labels
    dataset = build_features(strat.df_with_indicators, trades)
    print(f"🧱 Dataset rows: {len(dataset):,} | positives: {dataset['y'].sum()}")

    # 4️⃣ Save to disk
    out_path = Path("data/datasets/btc_osc_pa_v1.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=True)
    print(f"💾 Saved → {out_path.resolve()}")

    # 5️⃣ Quick preview
    print(dataset.head(5).to_string())


if __name__ == "__main__":
    main()
