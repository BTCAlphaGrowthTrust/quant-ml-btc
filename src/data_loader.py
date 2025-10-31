from pathlib import Path
import pandas as pd
import requests
import time

BINANCE_URL = "https://api.binance.com/api/v3/klines"
def fetch_binance_ohlcv(symbol="BTCUSDT", interval="4h", limit=1000, max_bars=3000):
    all_rows = []
    end_time = int(time.time() * 1000)
    while len(all_rows) < max_bars:
        params = {"symbol": symbol, "interval": interval, "limit": limit, "endTime": end_time}
        r = requests.get(BINANCE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_rows.extend(data)
        end_time = data[0][0] - 1
        if len(data) < limit:
            break
        time.sleep(0.1)
    df = pd.DataFrame(all_rows, columns=[
        "timestamp","open","high","low","close","volume",
        "_c","_q","_t","_b","_a","_i"
    ])
    df = df[["timestamp","open","high","low","close","volume"]].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
def get_btc_data(csv_path="data/raw/BTCUSDT_4h.csv", force=False):
    path = Path(csv_path)
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["timestamp"])
    print("⬇️  Downloading BTCUSDT 4h from Binance ...")
    df = fetch_binance_ohlcv()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅  Saved {len(df)} rows to {path}")
    return df
