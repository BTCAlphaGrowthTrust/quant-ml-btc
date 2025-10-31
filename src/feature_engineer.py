"""
feature_engineer.py
Adds RSI, volatility, EMA cross state, and return features.
"""
import pandas as pd
import numpy as np

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=df.index)


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    return log_ret.rolling(window).std()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to dataframe."""
    df["rsi_14"] = add_rsi(df, 14)
    df["volatility_20"] = add_volatility(df, 20)
    df["ema_cross_state"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    df["return_1"] = df["close"].pct_change()
    return df
