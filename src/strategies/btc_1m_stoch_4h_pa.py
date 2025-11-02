from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class MonthlyOscillatorPA(BaseStrategy):
    """
    Tom Makin Strategy | BTC | 1M Osc + 4H PA Execution

    Entry:
      • Monthly stochastic bias bullish (K > D) and both K,D < stoch_threshold
      • 4H breakout above the highest point of the lowest bearish candle in lookback window
    Exit:
      • Monthly bearish stochastic crossover (K < D), or
      • Stop-loss = lowest low of last 5 bars * stop_multiplier (intrabar execution)
    Risk:
      • 2% account risk per trade, sized by distance to stop
      • Non-pyramiding (one position at a time)
    """

    def __init__(
        self,
        stoch_threshold: float = 60.0,
        lookback_bars: int = 5,
        signal_expiry: int = 20,
        stop_multiplier: float = 0.995,  # 0.5% below lowest low
        start_balance: float = 50_000.0,
        risk_fraction: float = 0.02,
        save_trades_csv: bool = True,
    ):
        super().__init__("Tom Makin | BTC | 1M Osc + 4H PA Execution")
        self.stoch_threshold = stoch_threshold
        self.lookback_bars = lookback_bars
        self.signal_expiry = signal_expiry
        self.stop_multiplier = stop_multiplier
        self.start_balance = float(start_balance)
        self.risk_fraction = float(risk_fraction)
        self.save_trades_csv = save_trades_csv

    # === Monthly resample ===
    def resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to monthly timeframe for stochastic calculation"""
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        df_monthly = df.resample("ME").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        df_monthly = df_monthly.reset_index()
        df_monthly.rename(columns={"index": "timestamp"}, inplace=True)
        return df_monthly

    # === Stochastic calc ===
    def calculate_stochastic(self, df, k_period=14, k_smooth=6, d_smooth=3):
        """Calculate Stochastic Oscillator %K and %D"""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        k_raw = 100 * (df["close"] - low_min) / (high_max - low_min)
        k = k_raw.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()
        idx = pd.to_datetime(df["timestamp"], utc=True)
        return pd.Series(k.values, index=idx), pd.Series(d.values, index=idx)

    # === Find lowest bearish candle ===
    def find_lowest_bearish_candle(self, df, i, lookback):
        """Find lowest close among bearish candles in lookback window"""
        window = df.iloc[max(0, i - lookback): i + 1]
        bearish = window[window["close"] < window["open"]]
        if bearish.empty:
            return None, None, None, None
        lowest = bearish.loc[bearish["close"].idxmin()]
        bars_ago = i - df.index.get_loc(lowest.name)
        return float(lowest["close"]), float(lowest["high"]), float(lowest["low"]), int(bars_ago)

    # === Core logic ===
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals for Monthly oscillator + 4H price action"""
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df = df.set_index("timestamp")
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)

        # --- Monthly stochastic ---
        df_monthly = self.resample_to_monthly(df)
        k_month, d_month = self.calculate_stochastic(df_monthly)
        monthly_bias_bullish = k_month > d_month

        # --- Ensure UTC alignment before reindex ---
        k_month.index = pd.to_datetime(k_month.index, utc=True)
        d_month.index = pd.to_datetime(d_month.index, utc=True)
        monthly_bias_bullish.index = pd.to_datetime(monthly_bias_bullish.index, utc=True)

        df["k_month"] = k_month.reindex(df.index, method="ffill")
        df["d_month"] = d_month.reindex(df.index, method="ffill")
        df["monthly_bullish"] = monthly_bias_bullish.reindex(df.index, method="ffill")
        df["lowest_low_5"] = df["low"].rolling(window=5).min()

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
            lowest_low_5 = float(row["lowest_low_5"]) if pd.notna(row["lowest_low_5"]) else np.nan
            if np.isnan(lowest_low_5):
                continue

            # === EXIT CONDITIONS ===
            if in_position and stop_loss is not None:
                # Stop-loss
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

                # Monthly bearish crossover
                if row["k_month"] < row["d_month"]:
                    pnl = (price - trades[-1]["entry_price"]) * trades[-1]["size"]
                    account_balance += pnl
                    signals[i] = -1
                    trades[-1].update({
                        "exit_idx": i,
                        "exit_time": df.index[i],
                        "exit_price": round(price, 2),
                        "exit_reason": "monthly_bearish_crossover",
                        "pnl": round(pnl, 2),
                        "account_balance": round(account_balance, 2),
                    })
                    in_position = False
                    stop_loss = None
                    continue

            # === ENTRY CONDITIONS ===
            if not in_position:
                if "bearish_high" in tracked_low:
                    bars_since = i - tracked_low["detected_at"]
                    if bars_since > self.signal_expiry:
                        tracked_low.clear()
                    elif price > tracked_low["bearish_high"]:
                        monthly_bullish = bool(row["monthly_bullish"])
                        k_below = float(row["k_month"]) < self.stoch_threshold
                        d_below = float(row["d_month"]) < self.stoch_threshold

                        if monthly_bullish and k_below and d_below:
                            stop_loss_val = lowest_low_5 * self.stop_multiplier
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

        self.signals = pd.Series(signals, index=df.index, name="signal")
        self.trades = trades
        self.final_balance = account_balance

        print(f"✅ Completed {len(trades)} trades | Final balance: ${account_balance:,.2f}")
        if trades:
            wins = sum(t.get("pnl", 0) > 0 for t in trades if "pnl" in t)
            print(f"🏆 Win rate: {wins / len(trades) * 100:.1f}%")

        if self.save_trades_csv and trades:
            out_dir = Path("results")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"trade_log_tom_makin_1m_osc_4h_pa_{ts}.csv"
            pd.DataFrame(trades).to_csv(out_path, index=False)
            print(f"💾 Trade log saved → {out_path}")

        # Store indicators for ML dataset generation
        self.df_with_indicators = df[
            ["open", "high", "low", "close", "volume", "k_month", "d_month", "monthly_bullish", "lowest_low_5"]
        ].copy()

        return self.signals