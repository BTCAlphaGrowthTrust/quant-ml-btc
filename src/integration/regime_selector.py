"""
Regime Selector | BTC | 1W Osc + 4H PA
-------------------------------------------------------
Uses trained meta-models (Sharpe & Profit-Factor predictors)
to dynamically choose the most promising parameter
combination for the current BTCUSDT 4h market regime.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

from src.optimization.regime_features import compute_regime_features  # <-- existing helper


def load_meta_models():
    """Load the trained regime-adaptive meta-models."""
    model_dir = Path("results/optimization")
    sharpe_model = max(model_dir.glob("regime_meta_*/*sharpe*.pkl"), key=lambda p: p.stat().st_mtime)
    pf_model = max(model_dir.glob("regime_meta_*/*profit_factor*.pkl"), key=lambda p: p.stat().st_mtime)
    sharpe = joblib.load(sharpe_model)
    pf = joblib.load(pf_model)
    print(f"📦 Loaded Sharpe meta-model → {sharpe_model.name}")
    print(f"📦 Loaded PF meta-model → {pf_model.name}")
    return sharpe, pf


def load_latest_regime():
    """Compute the latest market regime snapshot (vol, trend, volume, compression)."""
    raw_path = Path("data/market/btcusdt_4h.parquet")
    if not raw_path.exists():
        raise FileNotFoundError("Missing BTCUSDT 4h history — run regime_adaptive_meta first.")
    df = pd.read_parquet(raw_path)
    feats = compute_regime_features(df)
    return feats.iloc[-1]


def load_param_sweep():
    """Load the last parameter sweep results."""
    sweep_dir = Path("data/param_sweeps")
    last = max(sweep_dir.glob("btc_osc_pa_sweep_*.csv"), key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(last)
    print(f"📊 Loaded parameter grid → {last.name} ({len(df)} rows)")
    return df


def main():
    print("🧠 Regime Selector (Phase 3b)...")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/regime_selector_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models + data
    sharpe_model, pf_model = load_meta_models()
    current = load_latest_regime()
    params = load_param_sweep()

    # Prepare features for inference
    X_pred = params[["stoch_threshold", "lookback_bars", "atr_multiplier"]].copy()
    for col in current.index:
        X_pred[col] = current[col]

    # Predict
    params["pred_sharpe"] = sharpe_model.predict(X_pred)
    params["pred_pf"] = pf_model.predict(X_pred)
    params["score"] = (params["pred_sharpe"].rank(pct=True) + params["pred_pf"].rank(pct=True)) / 2

    # Rank and display top 5
    top = params.sort_values("score", ascending=False).head(5)
    print("\n🏆 Top 5 recommended parameter sets:")
    print(top[["stoch_threshold", "lookback_bars", "atr_multiplier",
               "pred_sharpe", "pred_pf", "score"]].round(4).to_string(index=False))

    # Save results
    csv_path = out_dir / "regime_recommendations.csv"
    top.to_csv(csv_path, index=False)
    print(f"\n💾 Saved → {csv_path.resolve()}")

    # Optional quick visual
    plt.figure(figsize=(7,4))
    plt.scatter(params["pred_sharpe"], params["pred_pf"], s=25, alpha=0.6)
    plt.scatter(top["pred_sharpe"], top["pred_pf"], color="red", s=40, label="Top 5")
    plt.xlabel("Predicted Sharpe"); plt.ylabel("Predicted PF"); plt.legend()
    plt.title("Meta-Model Predictions for Current Regime")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "regime_selector_scatter.png", dpi=130)
    plt.close()
    print("🖼️ Saved scatter plot → regime_selector_scatter.png")

    print("✅ Regime Selector complete.")


if __name__ == "__main__":
    main()
