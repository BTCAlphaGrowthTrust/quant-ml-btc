# src/strategies/weekly_oscillator_pa.py
from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


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
    Notes:
      • Weekly bars emulate TradingView `request.security("W")` behaviour,
        meaning the current (forming) week is included in stochastic bias.
    """

    def __init__(
        self,
        stoch_threshold: float = 60.0,
        lookback_bars: int = 5,
        signal_expiry: int = 20,
        atr_length: int = 14,
        atr_multiplier: float = 1.0,
        start_balance: float = 50_000.0,
        risk_fraction: float = 0.02,
        save_trades_csv: bool = True,
        make_plot: bool = True,
        plot_limit_bars: int = 4000,
        use_forming_week: bool = True,  # mirror TV's request.security("W")
    ):
        super().__init__("Tom Makin | BTC | 1W Osc + 4H PA Execution (vTV sync)")
        self.stoch_threshold = float(stoch_threshold)
        self.lookback_bars = int(lookback_bars)
        self.signal_expiry = int(signal_expiry)
        self.atr_length = int(atr_length)
        self.atr_multiplier = float(atr_multiplier)
        self.start_balance = float(start_balance)
        self.risk_fraction = float(risk_fraction)
        self.save_trades_csv = bool(save_trades_csv)
        self.make_plot = bool(make_plot)
        self.plot_limit_bars = int(plot_limit_bars)
        self.use_forming_week = bool(use_forming_week)

        # populated after run
        self.signals = None
        self.trades = []
        self.final_balance = None
        self.df_with_indicators = None

    # ---------- Helpers ----------
    @staticmethod
    def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df = df.set_index("timestamp")
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def calculate_atr(self, df: pd.DataFrame, n: int = 14) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(int(n)).mean()

    def resample_to_weekly(self, df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        Emulate TradingView request.security('W') behaviour.
        Includes forming (current) week so stochastic updates intra-week.
        """
        df = self._ensure_dt_index(df_4h)

        if self.use_forming_week:
            # Group by week start (Mon) and include the current (incomplete) week
            df = df.copy()
            df["week_start"] = df.index.to_period("W-MON").start_time
            grouped = (
                df.groupby("week_start")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .reset_index()
                .rename(columns={"week_start": "timestamp"})
            )
            return grouped
        else:
            # Strict completed-week mode
            wk = (
                df.resample("W-MON", closed="left", label="left")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
                .rename(columns={"index": "timestamp"})
            )
            return wk

    def calculate_stochastic(
        self, df_wk: pd.DataFrame, k_period: int = 14, k_smooth: int = 6, d_smooth: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        low_min = df_wk["low"].rolling(window=k_period).min()
        high_max = df_wk["high"].rolling(window=k_period).max()
        denom = (high_max - low_min).replace(0, np.nan)
        k_raw = 100.0 * (df_wk["close"] - low_min) / denom
        k = k_raw.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()
        idx = pd.to_datetime(df_wk["timestamp"], utc=True)
        return pd.Series(k.values, index=idx, name="k_week"), pd.Series(d.values, index=idx, name="d_week")

    def find_lowest_bearish_candle(self, df: pd.DataFrame, i: int, lookback: int):
        """
        Match TV logic: find lowest bearish *close* (not low) within lookback window.
        Return:
          (lowest_close, high_of_that_candle, low_of_that_candle, bars_ago)
        """
        window = df.iloc[max(0, i - lookback): i + 1]
        bearish = window[window["close"] < window["open"]]
        if bearish.empty:
            return None, None, None, None
        lowest = bearish.loc[bearish["close"].idxmin()]
        bars_ago = i - df.index.get_loc(lowest.name)
        return float(lowest["close"]), float(lowest["high"]), float(lowest["low"]), int(bars_ago)

    # ---------- Core Strategy ----------
    def generate_signals(self, df_in: pd.DataFrame) -> pd.Series:
        df = self._ensure_dt_index(df_in)
        df["atr"] = self.calculate_atr(df, self.atr_length)

        df_wk = self.resample_to_weekly(df)
        k_week, d_week = self.calculate_stochastic(df_wk)
        weekly_bullish = (k_week > d_week)

        # Map to 4H
        df["k_week"] = k_week.reindex(df.index, method="ffill")
        df["d_week"] = d_week.reindex(df.index, method="ffill")
        df["weekly_bullish"] = weekly_bullish.reindex(df.index, method="ffill")

        signals = np.zeros(len(df), dtype=np.int8)
        trades = []
        tracked_low = {}
        in_position = False
        stop_loss = None
        account_balance = float(self.start_balance)
        risk_fraction = float(self.risk_fraction)

        for i in range(self.lookback_bars, len(df)):
            row = df.iloc[i]
            price = float(row["close"])
            atr_val = float(row["atr"]) if pd.notna(row["atr"]) else np.nan
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # === EXIT ===
            if in_position and stop_loss is not None:
                # Intrabar stop
                if row["low"] <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - trades[-1]["entry_price"]) * trades[-1]["size"]
                    account_balance += pnl
                    signals[i] = -1
                    trades[-1].update({
                        "exit_idx": i, "exit_time": df.index[i],
                        "exit_price": round(exit_price, 2),
                        "exit_reason": "stop_loss",
                        "pnl": round(pnl, 2),
                        "account_balance": round(account_balance, 2)
                    })
                    in_position = False
                    stop_loss = None
                    continue

                # Weekly bearish cross
                if row["k_week"] < row["d_week"]:
                    pnl = (price - trades[-1]["entry_price"]) * trades[-1]["size"]
                    account_balance += pnl
                    signals[i] = -1
                    trades[-1].update({
                        "exit_idx": i, "exit_time": df.index[i],
                        "exit_price": round(price, 2),
                        "exit_reason": "weekly_bearish_crossover",
                        "pnl": round(pnl, 2),
                        "account_balance": round(account_balance, 2)
                    })
                    in_position = False
                    stop_loss = None
                    continue

            # === ENTRY ===
            if not in_position:
                if "bearish_high" in tracked_low:
                    bars_since = i - tracked_low["detected_at"]
                    if bars_since > self.signal_expiry:
                        tracked_low.clear()
                    elif price > tracked_low["bearish_high"]:
                        wk_bull = bool(row["weekly_bullish"])
                        k_below = float(row["k_week"]) < self.stoch_threshold
                        d_below = float(row["d_week"]) < self.stoch_threshold
                        if wk_bull and k_below and d_below:
                            stop_val = tracked_low["bearish_low"] - (atr_val * self.atr_multiplier)
                            risk_per_unit = price - stop_val
                            if risk_per_unit > 0:
                                risk_usd = account_balance * risk_fraction
                                size = round(risk_usd / risk_per_unit, 6)
                                signals[i] = 1
                                in_position = True
                                stop_loss = round(stop_val, 2)
                                trades.append({
                                    "entry_idx": i, "entry_time": df.index[i],
                                    "entry_price": round(price, 2),
                                    "stop_loss": stop_loss,
                                    "size": size,
                                    "risk_usd": round(risk_usd, 2),
                                    "risk_per_unit": round(risk_per_unit, 2),
                                    "account_balance": round(account_balance, 2)
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

        # Book final PnL if open
        if in_position and len(trades) > 0 and "exit_idx" not in trades[-1]:
            last_close = float(df["close"].iloc[-1])
            pnl = (last_close - trades[-1]["entry_price"]) * trades[-1]["size"]
            account_balance += pnl
            trades[-1].update({
                "exit_idx": len(df) - 1,
                "exit_time": df.index[-1],
                "exit_price": round(last_close, 2),
                "exit_reason": "end_of_test",
                "pnl": round(pnl, 2),
                "account_balance": round(account_balance, 2),
            })

        self.signals = pd.Series(signals, index=df.index, name="signal")
        self.trades = trades
        self.final_balance = float(account_balance)
        self.df_with_indicators = df[
            ["open","high","low","close","volume","k_week","d_week","weekly_bullish","atr"]
        ].copy()

        print(f"✅ Completed {len(trades)} trades | Final balance: ${account_balance:,.2f}")
        if trades:
            wins = sum(t.get("pnl", 0) > 0 for t in trades if "pnl" in t)
            print(f"🏆 Win rate: {wins/len(trades)*100:.1f}%")

        if self.save_trades_csv and len(trades):
            out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"trade_log_tom_makin_1w_osc_4h_pa_{ts}.csv"
            pd.DataFrame(trades).to_csv(out_path, index=False)
            print(f"💾 Trade log saved → {out_path}")

        if self.make_plot:
            try:
                self._plot_results()
            except Exception as e:
                print(f"⚠️ Plot failed: {e}")

        return self.signals

    # ---------- Visualization ----------
    def _plot_results(self):
        if self.df_with_indicators is None:
            return
        df = self.df_with_indicators.copy()
        sig = self.signals
        trades = self.trades

        if self.plot_limit_bars and len(df) > self.plot_limit_bars:
            df = df.iloc[-self.plot_limit_bars:]
            sig = sig.loc[df.index[0]:]
            trades = [t for t in trades if df.index[0] <= t["entry_time"] <= df.index[-1] or ("exit_time" in t and df.index[0] <= t["exit_time"] <= df.index[-1])]

        out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = out_dir / f"chart_tom_makin_1w_osc_4h_pa_{ts}.png"

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)

        # Price panel
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df["close"], linewidth=1.1)
        ax1.set_title("BTC 4H — Entries/Exits/Stops")
        ax1.set_ylabel("Price")

        for t in trades:
            ax1.scatter(t["entry_time"], t["entry_price"], marker="^", s=50)
            if "exit_time" in t:
                ax1.scatter(t["exit_time"], t["exit_price"], marker="v", s=50)
            # Stop line for trade duration (if exit exists)
            if "exit_idx" in t:
                i0, i1 = t["entry_idx"], t["exit_idx"]
                idx_slice = self.df_with_indicators.iloc[i0:i1+1].index
                ax1.plot(idx_slice, [t["stop_loss"]]*len(idx_slice), linestyle="--", linewidth=0.9)

        ax1.grid(True, linewidth=0.3)

        # Stoch panel
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(df.index, df["k_week"], linewidth=1.0, label="%K (weekly)")
        ax2.plot(df.index, df["d_week"], linewidth=1.0, label="%D (weekly)")
        ax2.axhline(self.stoch_threshold, linestyle="--", linewidth=0.8)
        ax2.set_ylabel("%K / %D")
        ax2.grid(True, linewidth=0.3)
        ax2.legend(loc="upper left")

        plt.setp(ax1.get_xticklabels(), visible=False)
        fig.autofmt_xdate()
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"🖼️ Plot saved → {png_path}")


# ========== MANUAL TEST + TRADINGVIEW COMPARISON ==========
if __name__ == "__main__":
    from src.data_loader import get_btc_data

    print("🚀 Tom Makin 1W Osc + 4H PA Execution | TradingView Comparison")

    # Load BTC 4H data
    df = get_btc_data()
    print(f"📂 Loaded BTCUSDT 4h data → {len(df)} rows")

    # Generate signals
    strat = WeeklyOscillatorPA(use_forming_week=True)
    sig = strat.generate_signals(df)
    print(f"✅ Generated {(sig == 1).sum()} long signals")

    # Locate TradingView export - prioritize BINANCE file
    tv_dir = Path("data/reference")
    binance_files = sorted(tv_dir.glob("BTC_1W_BINANCE*.xlsx"))

    if binance_files:
        tv_path = binance_files[-1]  # Use most recent Binance file
        print(f"📊 Found TradingView Binance export: {tv_path.name}")
    else:
        # Fall back to any BTC file
        tv_files = sorted(tv_dir.glob("BTC_1W_*.xlsx"))
        if not tv_files:
            print(f"⚠️ No TradingView file found in {tv_dir.resolve()}")
            raise SystemExit(0)
        tv_path = tv_files[-1]
        print(f"📊 Found TradingView export: {tv_path.name}")

    # Read TV trades sheet
    try:
        tv = pd.read_excel(tv_path, sheet_name="List of trades")
    except Exception as e:
        print(f"❌ Failed to read TradingView file: {e}")
        raise SystemExit(1)

    # Extract TV entries and normalize to 4H UTC grid
    tv_entries = pd.to_datetime(
        tv.loc[tv["Type"].astype(str).str.contains("Entry", case=False, na=False), "Date/Time"],
        utc=True,
        errors="coerce",
    ).dropna().sort_values()

    py_entries = pd.to_datetime(sig.index[sig == 1], utc=True).to_series().sort_values().dt.round("4h")
    tv_entries = tv_entries.dt.round("4h")

    exact = np.intersect1d(py_entries.values, tv_entries.values)

    # Near matches within ±8h
    near = []
    for t in tv_entries:
        diffs = (py_entries - t).dt.total_seconds().abs() / 3600.0
        close = diffs[diffs <= 8]
        if not close.empty and t not in exact:
            near.append((t, py_entries.loc[close.idxmin()]))

    print("\n📊 Comparison Summary")
    print(f" • TradingView entries: {len(tv_entries)}")
    print(f" • Python strategy entries: {int((sig==1).sum())}")
    print(f" • Exact 4h matches: {len(exact)}")
    print(f" • Near (≤8h) matches: {len(near)}")

    if len(tv_entries) > 0:
        print("\n🕒 Example TradingView entries:")
        print(tv_entries.head(10).to_list())
    if len(py_entries) > 0:
        print("\n🕒 Example Python entries:")
        print(py_entries.head(10).to_list())
