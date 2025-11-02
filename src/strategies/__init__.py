"""
Strategy Registry
All rule-based strategies for Quant-ML-BTC are declared and imported here.
Each strategy must subclass BaseStrategy and be added to STRATEGY_REGISTRY.
"""

from importlib import import_module
from .base_strategy import BaseStrategy

# --- Direct imports for active strategies ---
from .weekly_oscillator_pa import WeeklyOscillatorPA            # Tom Makin | BTC | 1W Osc + 4H PA
from .btc_1w_osc_4h_pa import BTC1WOsc4HPA                      # Legacy 1W oscillator version
from .btc_1m_stoch_4h_pa import MonthlyOscillatorPA             # Tom Makin | BTC | 1M Osc + 4H PA
from .btc_1d_golden_4h_pa import DailyGoldenCrossPA             # Tom Makin | BTC | 1D Golden Cross + 4H PA
from .btc_4h_golden_4h_pa import FourHourGoldenCrossPA          # Tom Makin | BTC | 4H Golden Cross + 4H PA

# --- Registry of available strategies ---
STRATEGY_REGISTRY = {
    # ✅ Stochastic oscillator suite
    "btc_1w_osc_4h_pa": BTC1WOsc4HPA,              # Legacy reference
    "tom_makin_1w_osc_4h_pa": WeeklyOscillatorPA,  # Proprietary weekly version
    "tom_makin_1m_osc_4h_pa": MonthlyOscillatorPA, # Proprietary monthly version

    # ✅ Golden cross suite
    "tom_makin_1d_golden_4h_pa": DailyGoldenCrossPA,
    "tom_makin_4h_golden_4h_pa": FourHourGoldenCrossPA,
}

def load_strategy(name: str) -> BaseStrategy:
    """
    Dynamically instantiate a registered strategy.
    Example:
        strategy = load_strategy("tom_makin_1m_osc_4h_pa")
        signals = strategy.generate_signals(df)
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"❌ Unknown strategy name: {name}\nAvailable: {list(STRATEGY_REGISTRY.keys())}")
    cls = STRATEGY_REGISTRY[name]
    return cls()
