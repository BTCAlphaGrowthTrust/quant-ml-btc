"""
Strategy Registry
All rule-based strategies for Quant-ML-BTC are declared and imported here.
Each strategy must subclass BaseStrategy and be added to STRATEGY_REGISTRY.
"""

from .base_strategy import BaseStrategy
from .btc_1w_osc_4h_pa import BTC1WOsc4HPA
from .weekly_oscillator_pa import WeeklyOscillatorPA

# Central registry for all available strategies
STRATEGY_REGISTRY = {
    "btc_1w_osc_4h_pa": BTC1WOsc4HPA,  # legacy reference
    "tom_makin_1w_osc_4h_pa": WeeklyOscillatorPA,  # proprietary version
}
