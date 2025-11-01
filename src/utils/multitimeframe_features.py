"""
multitimeframe_features.py
Generates higher-timeframe indicator columns (e.g. 1W stochastic) and merges them into lower-timeframe data.
"""
import pandas as pd
import numpy as np

def add_weekly_stoch(df_4h: pd.DataFrame,
                     k_period: int = 14,
                     d_period: int = 6,
                     smooth_k: int = 3) -> pd.DataFrame:
    """Compute 1W stochastic from 4H data and align to 4H index."""
    # --- Resample to 1W ---
    df_w = df_4h.resample("1W", on="timestamp").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    # --- Weekly stochastic ---
    hh = df_w["high"].rolling(k_period).max()
    ll = df_w["low"].rolling(k_period).min()
    raw_k = 100 * (df_w["close"] - ll) / (hh - ll)
    k_smooth = raw_k.rolling(smooth_k).mean()
    d_line = k_smooth.rolling(d_period).mean()
    df_w["stoch_k_w"] = k_smooth.fillna(50)
    df_w["stoch_d_w"] = d_line.fillna(50)

    # --- Forward-fill weekly values into 4H data ---
    df_w_sub = df_w[["stoch_k_w", "stoch_d_w"]].copy()
    df_w_sub = df_w_sub.reindex(df_4h["timestamp"], method="ffill")

    # --- Merge into original 4H DataFrame ---
    df_out = df_4h.copy()
    df_out[["stoch_k_w", "stoch_d_w"]] = df_w_sub.values
    return df_out
