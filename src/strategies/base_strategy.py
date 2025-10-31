"""
base_strategy.py
Defines a template for rule-based trading strategies.
All user strategies should inherit from BaseStrategy.
"""
import pandas as pd
import numpy as np

class BaseStrategy:
    """Abstract base class for rule-based strategies."""
    def __init__(self, name: str = "UnnamedStrategy"):
        self.name = name
        self.signals = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Must be implemented by child class.
        Should return a pandas Series of trade signals:
        +1 for long, -1 for short, 0 for neutral.
        """
        raise NotImplementedError("Subclasses must implement generate_signals().")

    def backtest(self, df: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
        """Simple backtester for strategy signals."""
        if self.signals is None:
            raise ValueError("Run generate_signals() first.")

        ret = df["close"].pct_change().fillna(0)
        pnl = self.signals.shift(1).fillna(0) * ret
        equity = (1 + pnl).cumprod() * initial_capital
        result = pd.DataFrame({"equity": equity})
        return result
