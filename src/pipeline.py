"""
pipeline.py
Quant-ML-BTC v3 — config-driven pipeline
Loads BTC data → adds EMA/Stoch → enriches features → labels → trains XGB → backtests → reports metrics
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .data_loader import get_btc_data
from .metrics_reporter import export_metrics
from .feature_engineer import add_features


# === Indicator Functions ===
def ema(series, n):
    """Exponential Moving Average."""
    return series.ewm(span=n, adjust=False).mean()


def stoch(df, k=14, d=6, smooth_k=3):
    """Stochastic oscillator."""
    hh = df["high"].rolling(k).max()
    ll = df["low"].rolling(k).min()
    raw_k = 100 * (df["close"] - ll) / (hh - ll)
    k_smooth = raw_k.rolling(smooth_k).mean()
    d_line = k_smooth.rolling(d).mean()
    return k_smooth.fillna(50), d_line.fillna(50)


# === Labeling ===
def label_return(df, horizon=12, thr=0.01):
    """Generate -1 / 0 / 1 labels based on forward returns."""
    fwd = df["close"].shift(-horizon)
    ret = (fwd / df["close"]) - 1
    return np.where(ret > thr, 1, np.where(ret < -thr, -1, 0))


# === Label Encoder for XGB ===
def _encode_labels(y_series):
    """Remap labels from -1,0,1 → 0,1,2 for XGB compatibility."""
    mapping = {-1: 0, 0: 1, 1: 2}
    return y_series.map(mapping)


# === Main Pipeline ===
def run_pipeline(config: dict, run_dir: str = "results"):
    """Execute full Quant-ML-BTC pipeline."""
    # 1️⃣ Load Config Parameters
    ema_fast, ema_slow = config["features"]["ema_periods"]
    k, d, smooth_k = config["features"]["stochastic"]
    horizon = config["label"]["horizon"]
    thr = config["label"]["threshold"]
    init_cap = config["backtest"]["initial_capital"]
    model_params = config["model"]

    # 2️⃣ Load Data
    df = get_btc_data()
    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    k_line, d_line = stoch(df, k, d, smooth_k)
    df["stoch_k"], df["stoch_d"] = k_line, d_line

    # 2.5️⃣ Add engineered features (RSI, volatility, etc.)
    df = add_features(df)

    df = df.dropna().reset_index(drop=True)

    # 3️⃣ Label Data
    df["y"] = label_by_volatility(df, horizon=horizon, vol_window=20, vol_factor=0.5)
    features = [
        "ema_fast", "ema_slow", "stoch_k", "stoch_d",
        "rsi_14", "volatility_20", "ema_cross_state", "return_1"
    ]
    X, y = df[features], df["y"]

    # 4️⃣ Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    y_train_enc = _encode_labels(y_train)
    y_test_enc = _encode_labels(y_test)

    # 5️⃣ Train Model
    model = XGBClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        learning_rate=model_params["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train_enc)

    # 6️⃣ Predict + Evaluate
    preds_enc = model.predict(X_test)
    inv_map = {0: -1, 1: 0, 2: 1}
    preds = pd.Series(preds_enc).map(inv_map)
    acc = accuracy_score(y_test_enc, preds_enc)
    print(f"Model accuracy: {acc:.3f}")

    # 7️⃣ Backtest
    signal = pd.Series(preds, index=X_test.index)
    ret = df.loc[X_test.index, "close"].pct_change().fillna(0)
    pnl = signal.shift(1).fillna(0) * ret
    equity = (1 + pnl).cumprod() * init_cap
    out = pd.DataFrame({"equity": equity})
    out.to_csv(f"{run_dir}/equity_curve.csv", index=False)
    print(f"✅ Saved equity_curve.csv with {len(out)} rows")

    # 8️⃣ Save Full Dataset
    df_out = df.loc[X_test.index].copy()
    df_out["pred_class"] = preds
    df_out.to_csv(f"{run_dir}/backtest_dataset.csv", index=False)
    print(f"✅ Saved full dataset with features and predictions: {len(df_out)} rows")

    # 9️⃣ Export Metrics
    export_metrics(out["equity"], path=f"{run_dir}/metrics.csv")

    return out

from .labeler import label_by_volatility
