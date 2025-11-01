from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class WeeklyOscillatorPA(BaseStrategy):
    """
    Tom Makin Strategy | BTC | 1W Osc + 4H PA Execution (vTV sync)

    Entry:
      • Weekly stochastic bias bullish (K > D) and both K,D < stoch_threshold
      • 4H breakout above the highest point of the lowest bearish candle in lookback window
    Exit:
      • Weekly bearish stochastic crossover (K < D), or
      • Stop-loss = bearish low − ATR × multiplier (intrabar execution)
    Risk:
      • 2% account risk per trade, sized by distance to stop
      • Non-pyramiding (one position at a time)
    """

    def __init__(
        self,
        stoch_threshold: float = 60.0,
        lookback_bars: int = 20,
        signal_expiry: int = 20,
        atr_length: int = 14,
        atr_multiplier: float = 1.0,
        start_balance: float = 100_000.0,
        risk_fraction: float = 0.02,
        save_trades_csv: bool = True,
    ):
        super().__init__("Tom Makin | BTC | 1W Osc + 4H PA Execution (vTV sync)")
        self.stoch_threshold = stoch_threshold
        self.lookback_bars = lookback_bars
        self.signal_expiry = signal_expiry
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.start_balance = float(start_balance)
        self.risk_fraction = float(risk_fraction)
        self.save_trades_csv = save_trades_csv

    def calculate_atr(self, df: pd.DataFrame, n: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    def resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to weekly timeframe for stochastic calculation"""
        # Ensure datetime index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)
        
        # Resample using the index directly
        df_weekly = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Reset index so timestamp becomes a column for calculate_stochastic
        df_weekly = df_weekly.reset_index()
        df_weekly.rename(columns={'index': 'timestamp'}, inplace=True)
        
        return df_weekly

    def calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, k_smooth: int = 6, d_smooth: int = 3
    ) -> tuple:
        """Calculate Stochastic Oscillator %K and %D"""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        k_raw = 100 * (df["close"] - low_min) / (high_max - low_min)
        k = k_raw.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()
        
        # Index by weekly timestamps for later alignment
        idx = pd.to_datetime(df["timestamp"])
        return pd.Series(k.values, index=idx), pd.Series(d.values, index=idx)

    def find_lowest_bearish_candle(
        self, df: pd.DataFrame, i: int, lookback: int
    ) -> tuple:
        """
        In a [i-lookback, i] window, find bearish candles (close<open),
        return that with the lowest close:
          (lowest_close, high_of_that_candle, low_of_that_candle, bars_ago)
        """
        window = df.iloc[max(0, i - lookback) : i + 1]
        bearish = window[window["close"] < window["open"]]
        if bearish.empty:
            return None, None, None, None
        lowest = bearish.loc[bearish["close"].idxmin()]
        bars_ago = i - df.index.get_loc(lowest.name)
        return float(lowest["close"]), float(lowest["high"]), float(lowest["low"]), int(bars_ago)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on weekly stochastic bias and local low breakouts.
        Enforces non-pyramiding and 2% risk-based sizing with intrabar stop loss execution.
        """
        # Ensure datetime index
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df = df.set_index("timestamp")
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # Calculate features
        atr = self.calculate_atr(df, self.atr_length)
        df_weekly = self.resample_to_weekly(df)
        k_week, d_week = self.calculate_stochastic(df_weekly)
        weekly_bias_bullish = k_week > d_week

        # Ensure datetime alignment
        k_week.index = pd.to_datetime(k_week.index)
        d_week.index = pd.to_datetime(d_week.index)
        weekly_bias_bullish.index = pd.to_datetime(weekly_bias_bullish.index)

        # Map weekly data back to 4H timeframe
        df["k_week"] = k_week.reindex(df.index, method="ffill")
        df["d_week"] = d_week.reindex(df.index, method="ffill")
        df["weekly_bullish"] = weekly_bias_bullish.reindex(df.index, method="ffill")
        df["atr"] = atr

        # Initialize state
        signals = np.zeros(len(df))
        tracked_low = {}
        in_position = False
        stop_loss = None
        trades = []

        account_balance = float(self.start_balance)
        risk_fraction = self.risk_fraction

        # Loop through bars
        for i in range(self.lookback_bars, len(df)):
            row = df.iloc[i]
            price = float(row["close"])
            atr_val = float(row["atr"]) if pd.notna(row["atr"]) else np.nan

            # Skip if no ATR yet
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # ===== EXIT CONDITIONS =====
            if in_position and stop_loss is not None:
                # Stop-loss exit - check if LOW touched stop (intrabar execution)
                if row["low"] <= stop_loss:
                    # Exit price = stop loss (market order filled at stop)
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

                # Weekly bearish stochastic crossover exit (close price)
                if row["k_week"] < row["d_week"]:
                    pnl = (price - trades[-1]["entry_price"]) * trades[-1]["size"]
                    account_balance += pnl
                    signals[i] = -1
                    trades[-1].update({
                        "exit_idx": i,
                        "exit_time": df.index[i],
                        "exit_price": round(price, 2),
                        "exit_reason": "weekly_bearish_crossover",
                        "pnl": round(pnl, 2),
                        "account_balance": round(account_balance, 2),
                    })
                    in_position = False
                    stop_loss = None
                    continue

            # ===== ENTRY CONDITIONS =====
            if not in_position:
                # Check if we have a tracked local low
                if "bearish_high" in tracked_low:
                    bars_since = i - tracked_low["detected_at"]
                    
                    # Check expiry
                    if bars_since > self.signal_expiry:
                        tracked_low.clear()
                    # Check for breakout
                    elif price > tracked_low["bearish_high"]:
                        # Check weekly bias conditions
                        wk_bullish = bool(row["weekly_bullish"])
                        k_below = float(row["k_week"]) < self.stoch_threshold
                        d_below = float(row["d_week"]) < self.stoch_threshold

                        if wk_bullish and k_below and d_below:
                            # Calculate stop loss and position size
                            stop_loss_val = tracked_low["bearish_low"] - (atr_val * self.atr_multiplier)
                            risk_per_unit = price - stop_loss_val
                            
                            if risk_per_unit > 0:
                                risk_usd = account_balance * risk_fraction
                                size = round(risk_usd / risk_per_unit, 4)

                                # ENTRY SIGNAL
                                signals[i] = 1
                                in_position = True
                                stop_loss = round(stop_loss_val, 2)

                                # Log trade
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

                # If no active tracked low, scan for new one
                if not tracked_low:
                    lowest_close, lowest_high, lowest_low, bars_ago = self.find_lowest_bearish_candle(
                        df, i, self.lookback_bars
                    )
                    
                    if lowest_close is not None:
                        tracked_low = {
                            "bearish_close": lowest_close,
                            "bearish_high": lowest_high,
                            "bearish_low": lowest_low,
                            "detected_at": i,
                        }

        # Finalize
        self.signals = pd.Series(signals, index=df.index, name="signal")
        self.trades = trades
        self.final_balance = account_balance
        self.df_with_indicators = df[
            ["open", "high", "low", "close", "volume", "k_week", "d_week", "weekly_bullish", "atr"]
        ].copy()

        print(f"✅ Completed {len(trades)} trades | Final balance: ${account_balance:,.2f}")
        if trades:
            wins = sum(t.get("pnl", 0) > 0 for t in trades if "pnl" in t)
            print(f"🏆 Win rate: {wins/len(trades)*100:.1f}%")

        # Optional trade CSV
        if self.save_trades_csv and len(trades):
            out_dir = Path("results")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"trade_log_tom_makin_1w_osc_4h_pa_{ts}.csv"
            pd.DataFrame(trades).to_csv(out_path, index=False)
            print(f"💾 Trade log saved → {out_path}")

        return self.signals


# ========== MANUAL TEST + TRADINGVIEW COMPARISON ==========
if __name__ == "__main__":
    from src.data_loader import get_btc_data

    print("🚀 Tom Makin 1W Osc + 4H PA Execution | TradingView Comparison")

    # Load BTC 4H data
    df = get_btc_data()
    print(f"📂 Loaded BTCUSDT 4h data → {len(df)} rows")

    # Generate signals
    strat = WeeklyOscillatorPA()
    sig = strat.generate_signals(df)
    print(f"✅ Generated {(sig == 1).sum()} long signals")

    # Locate TradingView export
    tv_dir = Path("data/reference")
    tv_files = sorted(tv_dir.glob("BTC_1W_*.xlsx"))
    if not tv_files:
        print(f"⚠️ No TradingView file found in {tv_dir.resolve()}")
        exit(0)
    tv_path = tv_files[-1]
    print(f"📊 Found TradingView export: {tv_path.name}")

    # Read TV trades sheet
    try:
        tv = pd.read_excel(tv_path, sheet_name="List of trades")
    except Exception as e:
        print(f"❌ Failed to read TradingView file: {e}")
        exit(1)

    tv_entries = pd.to_datetime(
        tv.loc[tv["Type"].str.contains("Entry", case=False, na=False), "Date/Time"], utc=True
    ).sort_values()

    # Align/round both to 4H UTC
    py_entries = pd.to_datetime(sig.index[sig == 1], utc=True).to_series().sort_values().dt.round("4h")
    tv_entries = tv_entries.dt.round("4h")

    exact = np.intersect1d(py_entries.values, tv_entries.values)

    # Near matches within ±8h
    near = []
    for t in tv_entries:
        diffs = (py_entries - t).dt.total_seconds().abs() / 3600
        close = diffs[diffs <= 8]
        if not close.empty and t not in exact:
            near.append((t, py_entries.loc[close.idxmin()]))

    print("\n📊 Comparison Summary")
    print(f" • TradingView entries: {len(tv_entries)}")
    print(f" • Python strategy entries: {len(py_entries)}")
    print(f" • Exact 4h matches: {len(exact)}")
    print(f" • Near (≤8h) matches: {len(near)}")

    if len(tv_entries) > 0:
        print("\n🕒 Example TradingView entries:")
        print(tv_entries.head(10).to_list())
    if len(py_entries) > 0:
        print("\n🕒 Example Python entries:")
        print(py_entries.head(10).to_list())