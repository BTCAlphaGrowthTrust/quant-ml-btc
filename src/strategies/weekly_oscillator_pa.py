from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class WeeklyOscillatorPA(BaseStrategy):
    """
    BTC Weekly Oscillator + 4H Price Action Local Low Strategy
    
    Logic:
    1. Weekly stochastic bias: K > D and both below threshold (bullish setup)
    2. Find lowest bearish candle close in lookback period
    3. Wait for price to close above that bearish candle's high (breakout confirmation)
    4. Stop loss: Low of that bearish candle - ATR buffer
    """
    
    def __init__(self, 
                 stoch_threshold=60,
                 lookback_bars=20,
                 signal_expiry=20,
                 atr_length=14,
                 atr_multiplier=1.0):
        super().__init__("Weekly Oscillator + PA Local Low")
        self.stoch_threshold = stoch_threshold
        self.lookback_bars = lookback_bars
        self.signal_expiry = signal_expiry
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def resample_to_weekly(self, df):
        """Resample data to weekly timeframe for stochastic calculation"""
        df_weekly = df.resample('W', on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df_weekly
    
    def calculate_stochastic(self, df, k_period=14, k_smooth=6, d_smooth=3):
        """Calculate Stochastic Oscillator %K and %D"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # Raw stochastic
        k_raw = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        # Smoothed %K
        k = k_raw.rolling(window=k_smooth).mean()
        
        # %D
        d = k.rolling(window=d_smooth).mean()
        
        return k, d
    
    def find_lowest_bearish_candle(self, df, idx, lookback):
        """
        Find the bearish candle with lowest close in the lookback period
        Returns: (lowest_close, high_of_that_candle, low_of_that_candle, bars_ago)
        """
        start_idx = max(0, idx - lookback)
        window = df.iloc[start_idx:idx+1]
        
        # Filter bearish candles only
        bearish = window[window['close'] < window['open']]
        
        if len(bearish) == 0:
            return None, None, None, None
        
        # Find the one with lowest close
        lowest_idx = bearish['close'].idxmin()
        lowest_close = bearish.loc[lowest_idx, 'close']
        lowest_high = bearish.loc[lowest_idx, 'high']
        lowest_low = bearish.loc[lowest_idx, 'low']
        
        # Calculate bars ago
        bars_ago = idx - df.index.get_loc(lowest_idx)
        
        return lowest_close, lowest_high, lowest_low, bars_ago
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on weekly stochastic bias and local low breakouts
        
        Returns:
            pd.Series: +1 for long entry, 0 for no signal
        """
        # Ensure timestamp is index
        if 'timestamp' in df.columns and df.index.name != 'timestamp':
            df = df.set_index('timestamp')
        
        # Calculate ATR for stop loss buffer
        atr = self.calculate_atr(df, self.atr_length)
        
        # Resample to weekly and calculate stochastic
        df_weekly = self.resample_to_weekly(df.copy())
        k_week, d_week = self.calculate_stochastic(
            df_weekly, 
            k_period=14, 
            k_smooth=6, 
            d_smooth=3
        )
        
        # Determine weekly bias
        weekly_bias_bullish = k_week > d_week
        
        # Map weekly data back to original timeframe
        df['k_week'] = k_week.reindex(df.index, method='ffill')
        df['d_week'] = d_week.reindex(df.index, method='ffill')
        df['weekly_bullish'] = weekly_bias_bullish.reindex(df.index, method='ffill')
        df['atr'] = atr
        
        # Initialize signal tracking
        signals = np.zeros(len(df))
        tracked_low = {}  # Store tracked local low info
        
        for i in range(self.lookback_bars, len(df)):
            current_idx = i
            
            # Check if we have a tracked local low that hasn't expired
            if 'bearish_high' in tracked_low:
                bars_since = current_idx - tracked_low['detected_at']
                
                # Check expiry
                if bars_since > self.signal_expiry:
                    tracked_low.clear()
                else:
                    # Check for breakout (close above bearish candle's high)
                    if df.iloc[current_idx]['close'] > tracked_low['bearish_high']:
                        # Check weekly bias conditions
                        weekly_bullish = df.iloc[current_idx]['weekly_bullish']
                        k_below = df.iloc[current_idx]['k_week'] < self.stoch_threshold
                        d_below = df.iloc[current_idx]['d_week'] < self.stoch_threshold
                        
                        if weekly_bullish and k_below and d_below:
                            # ENTRY SIGNAL
                            signals[current_idx] = 1
                            
                            # Store trade info for potential backtesting
                            entry_price = df.iloc[current_idx]['close']
                            stop_loss = tracked_low['bearish_low'] - (df.iloc[current_idx]['atr'] * self.atr_multiplier)
                            
                            # Clear tracked low after entry
                            tracked_low.clear()
                            continue
            
            # If no active tracked low, scan for new one
            if not tracked_low:
                lowest_close, lowest_high, lowest_low, bars_ago = self.find_lowest_bearish_candle(
                    df, current_idx, self.lookback_bars
                )
                
                if lowest_close is not None:
                    tracked_low = {
                        'bearish_close': lowest_close,
                        'bearish_high': lowest_high,
                        'bearish_low': lowest_low,
                        'detected_at': current_idx,
                        'bars_ago': bars_ago
                    }
        
        # Create signal series
        self.signals = pd.Series(signals, index=df.index, name="signal")
        
        # Store additional info for analysis
        self.df_with_indicators = df[['open', 'high', 'low', 'close', 'volume', 
                                       'k_week', 'd_week', 'weekly_bullish', 'atr']].copy()
        
        return self.signals
    
    def get_stop_loss(self, df: pd.DataFrame, entry_idx: int) -> float:
        """
        Calculate stop loss for a given entry point
        Used by backtesting framework
        """
        # Find the local low that triggered this entry
        lowest_close, lowest_high, lowest_low, bars_ago = self.find_lowest_bearish_candle(
            df, entry_idx, self.lookback_bars
        )
        
        if lowest_low is not None:
            atr_value = df.iloc[entry_idx]['atr'] if 'atr' in df.columns else self.calculate_atr(df, self.atr_length).iloc[entry_idx]
            stop_loss = lowest_low - (atr_value * self.atr_multiplier)
            return round(stop_loss, 2)
        
        return None


# Usage example
if __name__ == "__main__":
    from src.data_loader import get_btc_data
    
    # Load data
    df = get_btc_data(timeframe='4h')  # 4-hour timeframe
    
    # Initialize strategy
    strategy = WeeklyOscillatorPA(
        stoch_threshold=60,
        lookback_bars=20,
        signal_expiry=20,
        atr_length=14,
        atr_multiplier=1.0
    )
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    # Print signal summary
    print(f"Total Long Signals: {(signals == 1).sum()}")
    print(f"\nSignal Dates:")
    print(df[signals == 1][['close']].head(10))
    
    # Run backtest (if base_strategy has backtest method)
    # results = strategy.backtest(df)
    # print(results)