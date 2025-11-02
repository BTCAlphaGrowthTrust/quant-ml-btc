# src/optimization/regime_adaptive_meta.py
"""
Regime-Adaptive Meta-Model Trainer (Full Historical Binance Downloader)

Downloads the complete 4H BTCUSDT history from Binance (2017→present),
computes volatility/trend/volume regime features, attaches them to the
parameter-sweep dataset, and trains XGBoost regressors for Sharpe and PF.

Usage:
    python -m src.optimization.regime_adaptive_meta
"""

from pathlib import Path
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance


# ==========================================================
# 📁 Utility: Create timestamped output directory
# ==========================================================
def make_output_dir() -> Path:
    root = Path("results/optimization")
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = root / f"regime_meta_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ==========================================================
# 📡 Download full 4H BTC/USDT history from Binance
# ==========================================================
def download_binance_btcusdt_4h_full(limit: int = 1000, pause: float = 0.5) -> pd.DataFrame:
    """
    Fetch the full historical BTCUSDT 4H candles from Binance (2017→now),
    caching to data/market/btcusdt_4h.parquet and incrementally updating.
    """
    out_path = Path("data/market")
    out_path.mkdir(parents=True, exist_ok=True)
    cache_file = out_path / "btcusdt_4h.parquet"

    # Load existing cache
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        last_ts = int(df["open_time"].iloc[-1]) + 1
        print(f"📂 Loaded cached BTCUSDT 4h data → {len(df):,} rows "
              f"(up to {pd.to_datetime(last_ts, unit='ms'):%Y-%m-%d})")
    else:
        df = pd.DataFrame()
        last_ts = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        print("📄 No cache found — downloading full BTCUSDT 4h history from 2017...")

    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    request_count = 0

    while True:
        params = {"symbol": "BTCUSDT", "interval": "4h", "limit": limit, "startTime": last_ts}
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"⚠️ Request failed: {e}. Retrying in 5s...")
            time.sleep(5)
            continue

        if not data:
            break

        batch = pd.DataFrame(
            data,
            columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qav","trades","tbbav","tbqav","ignore"
            ],
        )
        batch = batch[["open_time","open","high","low","close","volume"]].astype(float)
        all_rows.append(batch)
        request_count += 1

        start_dt = pd.to_datetime(batch["open_time"].iloc[0], unit="ms")
        end_dt = pd.to_datetime(batch["open_time"].iloc[-1], unit="ms")
        print(f"⬇️  Batch {request_count}: {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d} ({len(batch)} rows)")

        last_ts = int(batch["open_time"].iloc[-1]) + 1
        time.sleep(pause)

        if len(batch) < limit:
            break  # reached the end of available data

    if all_rows:
        new_df = pd.concat(all_rows, ignore_index=True)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time")

    # Standardize columns and save
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df.to_parquet(cache_file)
    print(f"✅ Saved updated full BTCUSDT 4h history → {cache_file} ({len(df):,} rows total)")
    return df


# ==========================================================
# 📊 Compute regime context (volatility, trend, etc.)
# ==========================================================
def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute live regime features and append to parameter sweep DataFrame."""
    raw = download_binance_btcusdt_4h_full()

    # Compute technical context
    raw["atr"] = (raw["high"] - raw["low"]).rolling(14).mean()
    raw["volatility_pct"] = (raw["atr"] / raw["close"]).fillna(0)
    raw["ema_fast"] = raw["close"].ewm(span=50).mean()
    raw["ema_slow"] = raw["close"].ewm(span=200).mean()
    raw["trend_strength"] = (raw["ema_fast"] / raw["ema_slow"]) - 1
    raw["volume_zscore"] = (raw["volume"] - raw["volume"].rolling(50).mean()) / (
        raw["volume"].rolling(50).std() + 1e-12
    )
    tr = raw["high"] - raw["low"]
    raw["range_compression"] = tr / (tr.rolling(30).median() + 1e-12)

    latest = raw.dropna().iloc[-1]
    print(
        f"📊 Latest regime snapshot:"
        f" vol={latest['volatility_pct']:.4f}, trend={latest['trend_strength']:.4f},"
        f" volZ={latest['volume_zscore']:.2f}, comp={latest['range_compression']:.3f}"
    )

    # Apply to every row in sweep dataset (same regime snapshot)
    for col in ["volatility_pct", "trend_strength", "volume_zscore", "range_compression"]:
        df[col] = latest[col]

    return df


# ==========================================================
# ⚙️ XGBoost Fitter
# ==========================================================
def fit_xgb(X, y):
    model = Pipeline([
        ("pre", ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])),
        ("xgb", XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="hist",
        )),
    ])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_val, y_val)
    return model, r2


# ==========================================================
# 📈 Visualization Helpers
# ==========================================================
def plot_perm_importance(model, X, y, tag, out_dir):
    r = permutation_importance(model, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    imp.to_csv(out_dir / f"feature_importance_{tag}.csv")
    plt.figure(figsize=(8, 4))
    imp.head(12).iloc[::-1].plot(kind="barh")
    plt.title(f"Feature Importance – {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"feature_importance_{tag}.png", dpi=130)
    plt.close()
    return imp


def heatmap(df, out_dir):
    piv = df.pivot_table(
        index=pd.qcut(df["volatility_pct"], 6, duplicates="drop"),
        columns=pd.qcut(df["trend_strength"], 6, duplicates="drop"),
        values="sharpe",
        aggfunc="mean",
    )
    plt.figure(figsize=(7, 6))
    im = plt.imshow(piv, aspect="auto", origin="lower")
    plt.title("Sharpe vs Volatility × Trend Strength")
    plt.xlabel("Trend Strength bins")
    plt.ylabel("Volatility bins")
    plt.colorbar(im, fraction=0.045)
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_sharpe_vol_trend.png", dpi=130)
    plt.close()


# ==========================================================
# 🚀 Main Routine
# ==========================================================
def main():
    out_dir = make_output_dir()
    print(f"🧠 Regime-adaptive meta-training → {out_dir}")

    sweep_file = max(Path("data/param_sweeps").glob("btc_osc_pa_sweep_*.csv"), key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(sweep_file)
    print(f"📂 Loaded sweep → {sweep_file.name} ({len(df)} rows)")

    # Attach regime context (latest full-history derived)
    df = add_regime_features(df)

    X = df[[
        "stoch_threshold","lookback_bars","atr_multiplier",
        "trades","volatility_pct","trend_strength","volume_zscore","range_compression"
    ]]
    y_sharpe = df["sharpe"].astype(float)
    y_pf = df["profit_factor"].astype(float)

    model_s, r2_s = fit_xgb(X, y_sharpe)
    model_p, r2_p = fit_xgb(X, y_pf)
    joblib.dump(model_s, out_dir / "regime_meta_sharpe.pkl")
    joblib.dump(model_p, out_dir / "regime_meta_profit_factor.pkl")

    print(f"🎯 Sharpe model R²={r2_s:.3f}")
    print(f"🎯 PF model R²={r2_p:.3f}")

    imp_s = plot_perm_importance(model_s, X, y_sharpe, "sharpe_regime", out_dir)
    imp_p = plot_perm_importance(model_p, X, y_pf, "pf_regime", out_dir)
    heatmap(df, out_dir)

    df.sort_values("sharpe", ascending=False).head(15).to_csv(out_dir / "top_contextual_zones.csv", index=False)
    print(f"✅ Outputs saved to {out_dir.resolve()}")


# ==========================================================
if __name__ == "__main__":
    main()
