"""
btc_1w_osc_4h_pa.py
BTC | 1W Osc + 4H PA Positioning
Replicates your TradingView logic:
  - Uses weekly stochastic crossover bias
  - Executes on 4H structure breakout (new local low)
  - Exits on weekly bearish crossover
"""
from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class BTC1WOsc4HPA(BaseStrategy):
    def __init__(self, risk_percent: float = 2.0, stoch_threshold: float = 60.0):
        super().__init__("BTC | 1W Osc + 4H PA Positioning")
        self.risk_percent = risk_percent
        self.stoch_threshold = stoch_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Required columns:
            open, high, low, close, stoch_k_w, stoch_d_w
        """
        # --- Weekly bias ---
        is_bias_bullish = df["stoch_k_w"] > df["stoch_d_w"]
        bias_threshold = (df["stoch_k_w"] < self.stoch_threshold) & (df["stoch_d_w"] < self.stoch_threshold)
        weekly_bias = is_bias_bullish & bias_threshold

        # --- 4H structure ---
        bull_bar = df["close"] > df["open"]
        rolling_low = df["low"].rolling(5).min()
        new_local_low = (df["close"] > rolling_low.shift(1)) & bull_bar

        # --- Entry / Exit ---
        entry_condition = weekly_bias & new_local_low
        exit_condition = df["stoch_k_w"] < df["stoch_d_w"]

        signals = np.where(entry_condition, 1,
                   np.where(exit_condition, -1, 0))
        self.signals = pd.Series(signals, index=df.index, name="signal")
        return self.signals
