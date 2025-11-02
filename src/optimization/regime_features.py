"""
Regime Feature Computation Utility
----------------------------------
Computes extended regime descriptors used by meta-models:
volatility %, trend strength, volume z-score, range compression, etc.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling regime features from BTC 4h data (aligned with meta-model training)."""
    df = df.copy()

    # --- Ensure datetime index ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)

    # --- Basic returns ---
    df["returns"] = df["close"].pct_change()
    df["volatility_pct"] = df["returns"].rolling(48).std() * np.sqrt(48) * 100      # annualized-like %
    df["trend_strength"] = df["close"].pct_change(48)                               # 2-week momentum
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(500).mean()) / df["volume"].rolling(500).std()
    df["range_compression"] = (df["high"] - df["low"]) / df["close"].rolling(48).mean()
    df["volatility_zscore"] = (df["volatility_pct"] - df["volatility_pct"].rolling(500).mean()) / df["volatility_pct"].rolling(500).std()

    # --- Smoothed normalized versions ---
    df["trend_norm"] = df["trend_strength"] / (df["trend_strength"].rolling(500).std() + 1e-12)
    df["compression_norm"] = df["range_compression"] / (df["range_compression"].rolling(500).mean() + 1e-12)

    # --- Trades placeholder (meta-model expects it) ---
    df["trades"] = 0.0  # dynamically filled during inference if needed

    # --- Final cleaned regime DataFrame ---
    regime_df = df[[
        "volatility_pct",
        "trend_strength",
        "volume_zscore",
        "range_compression",
        "volatility_zscore",
        "trend_norm",
        "compression_norm",
        "trades"
    ]].dropna()

    return regime_df


if __name__ == "__main__":
    raw_path = Path("data/market/btcusdt_4h.parquet")
    if not raw_path.exists():
        raise FileNotFoundError("Run regime_adaptive_meta first to download full BTCUSDT history.")
    raw = pd.read_parquet(raw_path)
    feats = compute_regime_features(raw)
    print("✅ Regime feature computation complete.")
    print(feats.tail())
