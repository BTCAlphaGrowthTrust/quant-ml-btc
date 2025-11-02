from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class DailyGoldenCrossPA(BaseStrategy):
    """
    Tom Makin Strategy | BTC | 1D Golden Cross + 4H Price Action

    Entry:
      • Daily EMA50 > EMA200 (golden cross bias)
      • Price <= slow MA × (1 + price_filter_pct)
      • 4H breakout above the highest point of the lowest bearish candle (lookback window)

    Exit:
      • Daily bearish crossover (EMA50 < EMA200), or
      • Stop-loss = lowest bearish candle low − ATR × multiplier

    Risk:
      • 2% account risk per trade
      • Position size = risk / (entry − stop)
      • Non-pyramiding
    """

    def __init__(
        self,
        fast_len: int = 50,
        slow_len: int = 200,
        price_filter_pct: float = 10.0,
        lookback_bars: int = 5,
        signal_expiry: int = 20,
        atr_length: int = 14,
        atr_multiplier: float = 1.0,
        start_balance: float = 50_000.0,
        risk_fraction: float = 0.02,
        save_trades_csv: bool = True,
    ):
        super().__init__("Tom Makin | BTC | 1D Golden Cross + 4H PA Execution")
        self.fast_len = fast_len
        self.slow_len = slow_len
        self.price_filter_pct = price_filter_pct / 100.0
        self.lookback_bars = lookback_bars
        self.signal_expiry = signal_expiry
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.start_balance = float(start_balance)
        self.risk_fraction = float(risk_fraction)
        self.save_trades_csv = save_trades_csv

    def resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 4H candles to 1D for moving average bias."""
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df_daily = df.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        df_daily = df_daily.reset_index()
        df_daily.rename(columns={"index": "timestamp"}, inplace=True)
        return df_daily

    def ema(self, series: pd.Series, length: int) -> pd.Series:
        """Exponential moving average."""
        return series.ewm(span=length, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, length: int) -> pd.Series:
        """Average True Range (for stop loss buffer)."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    def find_lowest_bearish_candle(self, df: pd.DataFrame, i: int, lookback: int):
        """Find lowest close among bearish candles within lookback window."""
        window = df.iloc[max(0, i - lookback): i + 1]
        bearish = window[window["close"] < window["open"]]
        if bearish.empty:
            return None, None, None, None
        lowest = bearish.loc[bearish["close"].idxmin()]
        bars_ago = i - df.index.get_loc(lowest.name)
        return float(lowest["close"]), float(lowest["high"]), float(lowest["low"]), int(bars_ago)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Main trading logic for the 1D Golden Cross + 4H Price Action system."""
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df = df.set_index("timestamp")
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # === Compute daily bias ===
        df_daily = self.resample_to_daily(df)
        ma_fast = self.ema(df_daily["close"], self.fast_len)
        ma_slow = self.ema(df_daily["close"], self.slow_len)
        golden_cross = ma_fast > ma_slow

        # Map daily data back to 4H
        df["ma_fast_d"] = ma_fast.reindex(df.index, method="ffill")
        df["ma_slow_d"] = ma_slow.reindex(df.index, method="ffill")
        df["golden_cross_d"] = golden_cross.reindex(df.index, method="ffill")

        # Price filter (must be below slow MA × (1 + pct))
        df["price_filter"] = df["close"] <= df["ma_slow_d"] * (1 + self.price_filter_pct)

        # ATR for stop sizing
        df["atr"] = self.calculate_atr(df, self.atr_length)

        # Rolling lowest low
        df["lowest_low"] = df["low"].rolling(window=self.lookback_bars).min()

        # === Initialize state ===
        signals = np.zeros(len(df))
        tracked_low = {}
        in_position = False
        stop_loss = None
        trades = []

        account_balance = float(self.start_balance)
        risk_fraction = self.risk_fraction

        # === Iterate bars ===
        for i in range(self.lookback_bars, len(df)):
            row = df.iloc[i]
            price = float(row["close"])
            atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0

            # EXITS
            if in_position and stop_loss is not None:
                if row["low"] <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - trades[-1]["entry_price"]) * trades[-1]["size"]
                    account_balance += pnl
                    signals[i] = -1
                    trades[-1].update({
                        "exit_idx": i,
                        "exit_time": df.index[i],
                        "exit_price": round(exit_price, 2),
                        "exit_reason": "stop_loss",
                        "pnl": round(pnl, 2),
                        "account_balance": round(account_balance, 2),
                    })
                    in_position = False
                    stop_loss = None
                    continue

                # Daily bearish cross
                if row["ma_fast_d"] < row["ma_slow_d"]:
                    pnl = (price - trades[-1]["entry_price"]) * trades[-1]["size"]
                    account_balance += pnl
                    signals[i] = -1
                    trades[-1].update({
                        "exit_idx": i,
                        "exit_time": df.index[i],
                        "exit_price": round(price, 2),
                        "exit_reason": "daily_bearish_cross",
                        "pnl": round(pnl, 2),
                        "account_balance": round(account_balance, 2),
                    })
                    in_position = False
                    stop_loss = None
                    continue

            # ENTRIES
            if not in_position:
                if "bearish_high" in tracked_low:
                    bars_since = i - tracked_low["detected_at"]
                    if bars_since > self.signal_expiry:
                        tracked_low.clear()
                    elif price > tracked_low["bearish_high"]:
                        if bool(row["golden_cross_d"]) and bool(row["price_filter"]):
                            stop_loss_val = tracked_low["bearish_low"] - (atr * self.atr_multiplier)
                            risk_per_unit = price - stop_loss_val
                            if risk_per_unit > 0:
                                risk_usd = account_balance * risk_fraction
                                size = round(risk_usd / risk_per_unit, 4)
                                signals[i] = 1
                                in_position = True
                                stop_loss = round(stop_loss_val, 2)
                                trades.append({
                                    "entry_idx": i,
                                    "entry_time": df.index[i],
                                    "entry_price": round(price, 2),
                                    "stop_loss": stop_loss,
                                    "size": size,
                                    "risk_usd": round(risk_usd, 2),
                                    "risk_per_unit": round(risk_per_unit, 2),
                                    "account_balance": round(account_balance, 2),
                                })
                                tracked_low.clear()
                                continue

                if not tracked_low:
                    lowest_close, lowest_high, lowest_low, bars_ago = self.find_lowest_bearish_candle(df, i, self.lookback_bars)
                    if lowest_close is not None:
                        tracked_low = {
                            "bearish_close": lowest_close,
                            "bearish_high": lowest_high,
                            "bearish_low": lowest_low,
                            "detected_at": i,
                        }

        # === Save state ===
        self.signals = pd.Series(signals, index=df.index, name="signal")
        self.trades = trades
        self.final_balance = account_balance
        self.df_with_indicators = df[
            ["open", "high", "low", "close", "volume", "ma_fast_d", "ma_slow_d", "golden_cross_d", "price_filter", "atr"]
        ].copy()

        print(f"✅ Completed {len(trades)} trades | Final balance: ${account_balance:,.2f}")
        if trades:
            wins = sum(t.get("pnl", 0) > 0 for t in trades if "pnl" in t)
            print(f"🏆 Win rate: {wins / len(trades) * 100:.1f}%")

        if self.save_trades_csv and trades:
            out_dir = Path("results")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"trade_log_tom_makin_1d_golden_4h_pa_{ts}.csv"
            pd.DataFrame(trades).to_csv(out_path, index=False)
            print(f"💾 Trade log saved → {out_path}")

        return self.signals
