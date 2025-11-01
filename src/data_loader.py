from pathlib import Path
import pandas as pd
import requests
import time

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="4h", limit=1000, max_bars=None, sleep=0.1):
    """
    Fetch OHLCV data from Binance API with full history support.

    Args:
        symbol (str): Trading pair, e.g., BTCUSDT.
        interval (str): Timeframe, e.g., '1h', '4h', '1d'.
        limit (int): Max bars per request (Binance max = 1000).
        max_bars (int): Optional cap on total bars to fetch.
        sleep (float): Delay between requests (avoid rate limit).
    """
    all_rows = []
    end_time = int(time.time() * 1000)

    print(f"⬇️  Fetching {symbol} {interval} data from Binance...")

    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "endTime": end_time}
        r = requests.get(BINANCE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        all_rows.extend(data)
        end_time = data[0][0] - 1  # move window back one candle

        print(f"   → Downloaded {len(all_rows)} rows so far...", end="\r")

        if len(data) < limit:
            break

        if max_bars and len(all_rows) >= max_bars:
            break

        time.sleep(sleep)

    print(f"\n✅  Downloaded total {len(all_rows)} bars.")
    df = pd.DataFrame(all_rows, columns=[
        "timestamp","open","high","low","close","volume",
        "_close_time","_quote_asset_vol","_trades","_taker_base_vol","_taker_quote_vol","_ignore"
    ])

    df = df[["timestamp","open","high","low","close","volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_btc_data(csv_path="data/raw/BTCUSDT_4h.csv", force=False):
    """
    Load cached BTCUSDT data or fetch full history from Binance.
    """
    path = Path(csv_path)
    if path.exists() and not force:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        print(f"📂 Loaded cached BTCUSDT 4h data → {len(df)} rows")
        return df

    df = fetch_binance_ohlcv(symbol="BTCUSDT", interval="4h")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅  Saved {len(df)} rows to {path}")
    return df


if __name__ == "__main__":
    df = get_btc_data(force=True)
    print(df.tail())
    print(f"\nData coverage: {df['timestamp'].min()} → {df['timestamp'].max()} ({len(df)} bars)")
