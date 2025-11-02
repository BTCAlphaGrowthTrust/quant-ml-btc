from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class FourHourGoldenCrossPA(BaseStrategy):
    """
    Tom Makin Strategy | BTC | 4H Golden Cross + 4H Price Action Execution

    Entry:
      • 4H EMA50 > EMA200 (golden cross bias)
      • Price <= slow MA × (1 + price_filter_pct)
      • Breakout above high of lowest bearish candle in lookback window

    Exit:
      • EMA50 < EMA200 (bearish cross)
      • Stop-loss = low of bearish candle − ATR × multiplier

    Risk:
      • 2% risk per trade
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
        super().__init__("Tom Makin | BTC | 4H Golden Cross + 4H PA Execution")
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

    def ema(self, series: pd.Series, length: int) -> pd.Series:
        """Exponential moving average."""
        return series.ewm(span=length, adjust=False).mean()

    def calculate_atr(self, df: pd.DataFrame, length: int) -> pd.Series:
        """Average True Range (for stop buffer)."""
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
        """Main trading logic for the 4H Golden Cross strategy."""
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df = df.set_index("timestamp")
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # === Moving averages (4H timeframe itself) ===
        df["ma_fast"] = self.ema(df["close"], self.fast_len)
        df["ma_slow"] = self.ema(df["close"], self.slow_len)
        df["golden_cross"] = df["ma_fast"] > df["ma_slow"]

        # Price filter (within % of slow MA)
        df["price_filter"] = df["close"] <= df["ma_slow"] * (1 + self.price_filter_pct)

        # ATR for stops
        df["atr"] = self.calculate_atr(df, self.atr_length)

        signals = np.zeros(len(df))
        tracked_low = {}
        in_position = False
        stop_loss = None
        trades = []

        account_balance = float(self.start_balance)
        risk_fraction = self.risk_fraction

        for i in range(self.lookback_bars, len(df)):
            row = df.iloc[i]
            price = float(row["close"])
            atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0

            # === EXIT CONDITIONS ===
            if in_position and stop_loss is not None:
                # Stop-loss hit
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

                # Bearish crossover exit
                if row["ma_fast"] < row["ma_slow"]:
                    pnl = (price - trades[-1]["entry_price"]) * trades[-1]["size"]
                    account_balance += pnl
                    signals[i] = -1
                    trades[-1].update({
                        "exit_idx": i,
                        "exit_time": df.index[i],
                        "exit_price": round(price, 2),
                        "exit_reason": "bearish_cross",
                        "pnl": round(pnl, 2),
                        "account_balance": round(account_balance, 2),
                    })
                    in_position = False
                    stop_loss = None
                    continue

            # === ENTRY CONDITIONS ===
            if not in_position:
                # Confirm breakout if we have a tracked local low
                if "bearish_high" in tracked_low:
                    bars_since = i - tracked_low["detected_at"]
                    if bars_since > self.signal_expiry:
                        tracked_low.clear()
                    elif price > tracked_low["bearish_high"]:
                        if bool(row["golden_cross"]) and bool(row["price_filter"]):
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

                # Track new local low if none active
                if not tracked_low:
                    lowest_close, lowest_high, lowest_low, bars_ago = self.find_lowest_bearish_candle(df, i, self.lookback_bars)
                    if lowest_close is not None:
                        tracked_low = {
                            "bearish_close": lowest_close,
                            "bearish_high": lowest_high,
                            "bearish_low": lowest_low,
                            "detected_at": i,
                        }

        # === Finalize ===
        self.signals = pd.Series(signals, index=df.index, name="signal")
        self.trades = trades
        self.final_balance = account_balance
        self.df_with_indicators = df[
            ["open", "high", "low", "close", "volume", "ma_fast", "ma_slow", "golden_cross", "price_filter", "atr"]
        ].copy()

        print(f"✅ Completed {len(trades)} trades | Final balance: ${account_balance:,.2f}")
        if trades:
            wins = sum(t.get("pnl", 0) > 0 for t in trades if "pnl" in t)
            print(f"🏆 Win rate: {wins / len(trades) * 100:.1f}%")

        if self.save_trades_csv and trades:
            out_dir = Path("results")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"trade_log_tom_makin_4h_golden_4h_pa_{ts}.csv"
            pd.DataFrame(trades).to_csv(out_path, index=False)
            print(f"💾 Trade log saved → {out_path}")

        return self.signals
