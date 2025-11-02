"""
ML Conviction Gate
------------------
Universal conviction scoring interface for all strategies in Quant-ML-BTC.

Usage:
    from src.integration.ml_conviction_gate import get_entry_conviction, should_execute

    prob = get_entry_conviction("tom_makin_1m_osc_4h_pa")
    if should_execute("tom_makin_1m_osc_4h_pa", prob):
        execute_trade()
    else:
        print(f"❌ Skipped trade | Conviction={prob:.2f}")
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from src.data_loader import get_btc_data
from src.strategies import STRATEGY_REGISTRY


# ----------------------------------------------------------
# Core helpers
# ----------------------------------------------------------

def load_model_and_curve(strategy_name: str):
    """Load trained model and its conviction curve."""
    model_path = Path(f"models/{strategy_name}.pkl")
    curve_path = Path(f"data/results/{strategy_name}_conviction_curve.csv")

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    if not curve_path.exists():
        raise FileNotFoundError(f"❌ Conviction curve not found: {curve_path}")

    model_data = joblib.load(model_path)
    curve = pd.read_csv(curve_path)

    best_row = curve.loc[curve["sharpe_like"].idxmax()]
    best_threshold = float(best_row["threshold"])

    return model_data, best_threshold


def prepare_latest_features(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Extract the most recent row of features matching training schema."""
    missing = [f for f in model_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    latest_row = df.tail(1)[model_features]
    return latest_row


# ----------------------------------------------------------
# Public API
# ----------------------------------------------------------

def get_entry_conviction(strategy_name: str, symbol="BTCUSDT", timeframe="4h") -> float:
    """
    Compute the ML conviction (probability of profitable setup)
    for the latest bar of the given strategy.
    """
    # Load model + threshold
    model_data, _ = load_model_and_curve(strategy_name)
    model = model_data["xgb"]  # use XGBoost for probability
    scaler = model_data.get("scaler", None)
    features = model_data["features"]

    # Get fresh market data (4h candles)
    df = get_btc_data(symbol=symbol, timeframe=timeframe)
    df.index = pd.to_datetime(df["timestamp"], utc=True)

    # Run the associated strategy to regenerate indicators
    StratClass = STRATEGY_REGISTRY[strategy_name]
    strat = StratClass(save_trades_csv=False)
    _ = strat.generate_signals(df)

    feature_df = strat.df_with_indicators
    latest = prepare_latest_features(feature_df, features)

    # Scale if needed
    if scaler is not None:
        latest_scaled = scaler.transform(latest)
    else:
        latest_scaled = latest.values

    prob = float(model.predict_proba(latest_scaled)[:, 1])
    print(f"🤖 {strategy_name} | Conviction={prob:.3f}")
    return prob


def should_execute(strategy_name: str, prob: float = None) -> bool:
    """Return True if the given probability exceeds the best threshold."""
    _, best_threshold = load_model_and_curve(strategy_name)
    if prob is None:
        prob = get_entry_conviction(strategy_name)
    decision = prob >= best_threshold
    print(f"🔍 Decision: {'EXECUTE ✅' if decision else 'SKIP ❌'} | Threshold={best_threshold:.2f}, Conviction={prob:.2f}")
    return decision


# ----------------------------------------------------------
# Logging utility
# ----------------------------------------------------------

def log_decision(strategy_name: str, prob: float, executed: bool):
    """Log all conviction decisions for later review."""
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "conviction_decisions.csv"

    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy_name,
        "conviction": round(prob, 4),
        "executed": int(executed)
    }
    df = pd.DataFrame([row])
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=header, index=False)
