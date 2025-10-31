from pathlib import Path
import json, yaml, pandas as pd
def ensure_dir(p): Path(p).parent.mkdir(parents=True, exist_ok=True); return p
def load_yaml(path): return yaml.safe_load(open(path, "r"))
def save_yaml(obj, path): ensure_dir(path); yaml.safe_dump(obj, open(path,"w"))
def save_json(obj, path): ensure_dir(path); json.dump(obj, open(path,"w"), indent=2)
def read_csv(path, parse_dates=None): return pd.read_csv(path, parse_dates=parse_dates)
def save_parquet(df, path): ensure_dir(path); df.to_parquet(path, index=False)
def save_csv(df, path): ensure_dir(path); df.to_csv(path, index=False)
