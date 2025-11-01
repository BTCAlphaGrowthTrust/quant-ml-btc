# src/strategies/weekly_oscillator_pa.py
from .base_strategy import BaseStrategy
import pandas as pd, numpy as np
from pathlib import Path

class WeeklyOscillatorPA(BaseStrategy):
    """Tom Makin Strategy | BTC | 1W Osc + 4H PA Execution (vTV sync)"""
    def __init__(self,
                 stoch_threshold=50,
                 lookback_bars=5,
                 signal_expiry=20,
                 atr_length=14,
                 atr_multiplier=1.0):
        super().__init__("Tom Makin | BTC | 1W Osc + 4H PA Execution (vTV sync)")
        self.stoch_threshold = stoch_threshold
        self.lookback_bars = lookback_bars
        self.signal_expiry = signal_expiry
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier

    def calculate_atr(self, df, n=14):
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    def resample_to_weekly(self, df):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df_weekly = df.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        df_weekly.reset_index(inplace=True)
        return df_weekly

    def calculate_stochastic(self, df, k_period=14, k_smooth=6, d_smooth=3):
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        k_raw = 100 * (df["close"] - low_min) / (high_max - low_min)
        k = k_raw.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()
        return pd.Series(k.values, index=pd.to_datetime(df["timestamp"])), \
               pd.Series(d.values, index=pd.to_datetime(df["timestamp"]))

    def find_lowest_bearish_candle(self, df, i, lookback):
        window = df.iloc[max(0, i - lookback):i + 1]
        bearish = window[window["close"] < window["open"]]
        if bearish.empty:
            return (None, None, None, None)
        lowest = bearish.loc[bearish["close"].idxmin()]
        return lowest["close"], lowest["high"], lowest["low"], i - df.index.get_loc(lowest.name)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df = df.set_index("timestamp")
        atr = self.calculate_atr(df, self.atr_length)
        df_weekly = self.resample_to_weekly(df.copy())
        kW, dW = self.calculate_stochastic(df_weekly)
        weekly_bullish = kW > dW

        # ✅ fix: make weekly data datetime indexed before reindexing
        kW.index = pd.to_datetime(kW.index)
        dW.index = pd.to_datetime(dW.index)
        weekly_bullish.index = pd.to_datetime(weekly_bullish.index)

        df["k_week"] = kW.reindex(df.index, method="ffill")
        df["d_week"] = dW.reindex(df.index, method="ffill")
        df["weekly_bullish"] = weekly_bullish.reindex(df.index, method="ffill")
        df["atr"] = atr

        sig = np.zeros(len(df))
        tracked = {}

        for i in range(self.lookback_bars, len(df)):
            if "bearish_high" in tracked:
                if i - tracked["detected_at"] > self.signal_expiry:
                    tracked.clear()
                elif df.iloc[i]["close"] > tracked["bearish_high"]:
                    if (df.iloc[i]["weekly_bullish"]
                        and df.iloc[i]["k_week"] < self.stoch_threshold
                        and df.iloc[i]["d_week"] < self.stoch_threshold):
                        sig[i] = 1
                        tracked.clear()
                        continue

            if not tracked:
                c, h, l, _ = self.find_lowest_bearish_candle(df, i, self.lookback_bars)
                if c is not None:
                    tracked = {"bearish_high": h, "bearish_low": l, "detected_at": i}

        self.signals = pd.Series(sig, index=df.index, name="signal")
        return self.signals


# === Manual test + TradingView comparison (enhanced) ===
if __name__ == "__main__":
    from src.data_loader import get_btc_data
    from datetime import timedelta

    print("🚀 Tom Makin 1W Osc + 4H PA Execution | TradingView Comparison")

    # 1️⃣ Load dataset
    df = get_btc_data()
    print(f"📂 Loaded BTCUSDT 4h data → {len(df)} rows")

    # 2️⃣ Generate strategy signals
    strat = WeeklyOscillatorPA()
    sig = strat.generate_signals(df)
    print(f"✅ Generated {int((sig == 1).sum())} long signals")

    # 3️⃣ Load TradingView export
    tv_path = Path("/mnt/data/BTC_1W_DERIBIT_BTCUSD.P_2025-11-01_3cca5.xlsx")
    if not tv_path.exists():
        print("⚠️ TradingView file not found.")
        exit()

    tv = pd.read_excel(tv_path, sheet_name="List of trades")
    tv_entries = pd.to_datetime(tv.loc[
        tv["Type"].str.contains("Entry", case=False), "Date/Time"
    ]).sort_values()

    py_entries = df.index[sig == 1].to_series().sort_values()

    # 4️⃣ Match to nearest 4-hour bar
    py_rounded = py_entries.dt.round("4H")
    tv_rounded = tv_entries.dt.round("4H")

    exact_matches = np.intersect1d(py_rounded, tv_rounded)

    # find near misses within ±4h window
    near_matches = []
    for t in tv_entries:
        diffs = (py_entries - t).dt.total_seconds().abs() / 3600
        close = diffs[diffs <= 4]
        if not close.empty and t not in exact_matches:
            near_matches.append((t, py_entries.loc[close.idxmin()]))

    print(f"\n📊 Comparison Summary")
    print(f" • TradingView entries: {len(tv_entries)}")
    print(f" • Python strategy entries: {len(py_entries)}")
    print(f" • Exact 4h matches: {len(exact_matches)}")
    print(f" • Near (≤4h) matches: {len(near_matches)}")

    if len(exact_matches):
        print("\n✅ Exact matches (first 10):")
        for t in exact_matches[:10]:
            print("  ", t)

    if len(near_matches):
        print("\n⚠️ Near-miss matches (≤4h difference, first 10):")
        for tv_t, py_t in near_matches[:10]:
            delta = (py_t - tv_t).total_seconds() / 3600
            print(f"  TV: {tv_t} ↔ PY: {py_t}  (Δ {delta:.1f}h)")
