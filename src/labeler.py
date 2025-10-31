"""
labeler.py
Applies dynamic, volatility-based labeling logic.
"""
import numpy as np
import pandas as pd

def label_by_volatility(df: pd.DataFrame,
                        horizon: int = 12,
                        vol_window: int = 20,
                        vol_factor: float = 0.5) -> np.ndarray:
    """
    Label using volatility-scaled thresholds.
    +1  → forward return > +vol_factor × rolling σ
    –1  → forward return < –vol_factor × rolling σ
     0  → otherwise
    """
    fwd = df["close"].shift(-horizon)
    ret = (fwd / df["close"]) - 1
    vol = df["close"].pct_change().rolling(vol_window).std()
    up_thr = vol * vol_factor
    down_thr = -vol * vol_factor

    labels = np.where(ret > up_thr, 1,
              np.where(ret < down_thr, -1, 0))
    return labels
