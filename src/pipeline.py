from pathlib import Path
import pandas as pd, numpy as np
from .utils.io import load_yaml, read_csv, save_parquet, save_csv, save_json
# --- tiny indicator helpers
def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def stoch(df, k=14, d=3, smooth_k=6):
    hh=df["high"].rolling(k,min_periods=1).max(); ll=df["low"].rolling(k,min_periods=1).min()
    raw=100*(df["close"]-ll).replace(0,np.nan)/(hh-ll).replace(0,np.nan); raw=raw.fillna(50)
    kline=raw.rolling(smooth_k,min_periods=1).mean(); dline=kline.rolling(d,min_periods=1).mean()
    return kline.rename("stoch_k"), dline.rename("stoch_d")
def build_features(df, feats):
    out=df.copy()
    for spec in feats["indicators"]:
        if spec["kind"]=="ema": out[f'ema_{spec["length"]}']=ema(out["close"], spec["length"])
        if spec["kind"]=="stoch":
            k,d=stoch(out, spec["k"], spec["d"], spec["smooth_k"])
            out[f'stochK_{spec.get("tf","base")}']=k; out[f'stochD_{spec.get("tf","base")}']=d
    return out
def label_return_fwd(df, horizon=12, thr=0.01):
    fwd=df["close"].shift(-horizon)
    ret=(fwd/df["close"]-1.0)
    y=np.where(ret>=thr,1,np.where(ret<=-thr,-1,0))
    df2=df.copy(); df2["y"]=y; return df2.dropna().copy()
def train_xgb(X,y,params):
    from xgboost import XGBClassifier
    m=XGBClassifier(**params); m.fit(X,y); return m
def train_logreg(X,y,params):
    from sklearn.linear_model import LogisticRegression
    m=LogisticRegression(**params); m.fit(X,y); return m
def walk_fit_predict(df, splits, model_type, params):
    tr=df.loc[df.index>=splits["train_start"]]; tr=tr.loc[tr.index<splits["valid_start"]]
    va=df.loc[df.index>=splits["valid_start"]]; va=va.loc[va.index<splits["test_start"]]
    te=df.loc[df.index>=splits["test_start"]]
    features=[c for c in df.columns if c not in ["open","high","low","close","volume","y"]]
    Xtr,ytr=tr[features],tr["y"]; Xte, yte=te[features], te["y"]
    model=train_xgb(Xtr,ytr,params) if model_type=="xgboost" else train_logreg(Xtr,ytr,params)
    proba=pd.Series(model.predict_proba(Xte)[:,1], index=te.index, name="p_up")
    signal=(proba>0.55).astype(int)-(proba<0.45).astype(int)  # +1,0,-1
    return signal
def backtest(close, signal, cash0=100000, fee=0.0005, slip=0.0):
    pos=signal.shift(1).fillna(0)  # enter next bar
    ret=close.pct_change().fillna(0)
    strat=pos*ret - abs(pos.diff().fillna(0))*fee - abs(pos.diff().fillna(0))*slip
    eq=(1+strat).cumprod()*cash0
    return pd.DataFrame({"equity":eq, "ret":strat})
def run(config_path):
    cfg=load_yaml(config_path)
    base=read_csv(cfg["data"]["csv_path"], parse_dates=[cfg["data"]["ts_col"]]).rename(columns={cfg["data"]["ts_col"]:"timestamp"})
    base=base[cfg["data"]["price_cols"]+["timestamp"]].set_index("timestamp").sort_index()
    feats=load_yaml("configs/features.yaml")[cfg["features"]["feature_set"]]
    df=build_features(base, feats).dropna()
    df=label_return_fwd(df, cfg["labels"]["return_horizon_bars"], cfg["labels"]["threshold"])
    # split + model
    params=load_yaml("configs/model_params.yaml")[cfg["model"]["params_name"]]
    signal=walk_fit_predict(df, cfg["split"], cfg["model"]["type"], params)
    bt=backtest(df.loc[signal.index,"close"], signal, cfg["backtest"]["initial_cash"], cfg["backtest"]["fee_bps"]/1e4, cfg["backtest"]["slippage_bps"]/1e4)
    # saves
    save_parquet(df, cfg["output"]["processed_parquet"])
    save_csv(bt.reset_index().rename(columns={"index":"timestamp"}), f'{cfg["output"]["results_dir"]}/equity_curve.csv')
    save_json({"config":cfg,"features":list(df.columns)}, f'{cfg["output"]["results_dir"]}/params.json')
    return bt
