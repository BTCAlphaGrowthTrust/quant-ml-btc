# src/integration/ml_filter.py
"""
PNL-aware ML gating for the '1W Osc + 4H PA Execution' strategy.
Uses real trade PnL and sweeps probability thresholds to evaluate value-add,
then outputs which trades were accepted vs rejected by the ML filter.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data_loader import get_btc_data
from src.strategies.weekly_oscillator_pa import WeeklyOscillatorPA


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def load_models():
    path = Path("models/btc_osc_pa_v1.pkl")
    bundle = joblib.load(path)
    return bundle["scaler"], bundle["logreg_cal"], bundle["xgb"], bundle["features"]


def sharpe_from_pnl(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    std = pnl.std(ddof=1)
    return float(pnl.mean() / (std + 1e-12))


def profit_factor_from_pnl(pnl: np.ndarray) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return float(gains / (losses + 1e-12))


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def main():
    print("🚀 PNL-aware ML filter evaluation...")

    # 1️⃣ Load dataset
    ds_path = Path("data/datasets/btc_osc_pa_v1.parquet")
    df = pd.read_parquet(ds_path)
    print(f"📂 Dataset → {len(df):,} rows | positives: {df['y'].sum()}")

    # 2️⃣ Load trained models
    scaler, logreg_cal, xgb, feat_cols = load_models()

    # 3️⃣ Score probabilities
    X = df[feat_cols].values
    Xs = scaler.transform(X)
    df["p_logreg"] = logreg_cal.predict_proba(Xs)[:, 1]
    df["p_xgb"] = xgb.predict_proba(X)[:, 1]
    df["p_mean"] = (df["p_logreg"] + df["p_xgb"]) / 2

    # 4️⃣ Sanity check
    auc_lr = roc_auc_score(df["y"], df["p_logreg"])
    auc_xgb = roc_auc_score(df["y"], df["p_xgb"])
    print(f"🎯 ROC-AUC  logreg={auc_lr:.3f} | xgb={auc_xgb:.3f}")

    # 5️⃣ Recompute REAL trades & PnL from base strategy
    raw = get_btc_data()
    raw.index = pd.to_datetime(raw["timestamp"], utc=True)
    strat = WeeklyOscillatorPA(use_forming_week=True, make_plot=False, save_trades_csv=False)
    _ = strat.generate_signals(raw)
    trades = pd.DataFrame(strat.trades)

    if trades.empty:
        print("⚠️ No trades found from baseline strategy. Exiting.")
        return

    trades = trades.dropna(subset=["entry_time"]).copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["pnl"] = trades["pnl"].fillna(0.0)

    # 6️⃣ Merge realized PnL onto dataset (align by entry time)
    df["trade_pnl"] = np.nan
    entry_map = trades.set_index("entry_time")["pnl"]
    mask_y1 = df["y"] == 1
    common_times = df.index[mask_y1].intersection(entry_map.index)
    df.loc[common_times, "trade_pnl"] = entry_map.loc[common_times].values
    realized = df.loc[df["y"] == 1, "trade_pnl"].dropna()
    print(f"💰 Realized trade PnL joined: {len(realized)} / {int(df['y'].sum())} entries")

    # 7️⃣ Threshold sweep (PnL-based)
    thresholds = np.linspace(0.01, 0.50, 50)
    rows = []
    for τ in thresholds:
        take = (df["y"] == 1) & (df["p_mean"] >= τ)
        pnl = df.loc[take, "trade_pnl"].dropna().values
        if pnl.size == 0:
            continue
        wins = (pnl > 0).sum()
        losses = (pnl < 0).sum()
        wr = wins / max(wins + losses, 1)
        pf = profit_factor_from_pnl(pnl)
        shp = sharpe_from_pnl(pnl)
        rows.append({
            "threshold": float(τ),
            "trades": int(pnl.size),
            "win_rate": float(wr),
            "profit_factor": float(pf),
            "sharpe": float(shp),
            "gross_pnl": float(pnl.sum()),
            "avg_pnl": float(pnl.mean()),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("⚠️ No thresholds produced trades; check probabilities or labels.")
        return

    best_idx = res["sharpe"].idxmax()
    best = res.loc[best_idx]
    print("\n📊 PnL-based Threshold Summary (head):")
    print(res.head(10).to_string(index=False))
    print(f"\n🏁 Best by Sharpe: τ={best['threshold']:.3f} | trades={best['trades']} | "
          f"WR={best['win_rate']:.2%} | PF={best['profit_factor']:.2f} | "
          f"Sharpe={best['sharpe']:.3f} | Gross=${best['gross_pnl']:.2f}")

    # 8️⃣ Save sweep CSV + plot
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ml_filter_pnl_sweep.csv"
    res.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV → {csv_path.resolve()}")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(res["threshold"], res["sharpe"], label="Sharpe (PnL)", linewidth=1.5)
    ax1.plot(res["threshold"], res["profit_factor"], label="Profit Factor", linewidth=1.0)
    ax1.set_xlabel("Threshold τ"); ax1.set_ylabel("Sharpe / PF")
    ax2 = ax1.twinx(); ax2.plot(res["threshold"], res["trades"], label="# Trades", color="orange", linewidth=1.0)
    ax2.set_ylabel("# Trades")
    fig.legend(loc="upper right"); ax1.grid(True, linewidth=0.3)
    plt.tight_layout()
    png_path = out_dir / f"ml_filter_pnl_sweep_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(png_path, dpi=130)
    print(f"🖼️ Plot saved → {png_path.resolve()}")

    # -----------------------------------------------------------------
    # 9️⃣ Identify which trades were accepted vs rejected by the ML gate
    # -----------------------------------------------------------------
    τ_opt = float(best["threshold"])
    accepted = df[(df["y"] == 1) & (df["p_mean"] >= τ_opt)]
    rejected = df[(df["y"] == 1) & (df["p_mean"] < τ_opt)]
    print(f"\n✅ Accepted trades: {len(accepted)} | ❌ Rejected trades: {len(rejected)} (τ={τ_opt:.2f})")

    filtered_log = pd.concat([
        accepted.assign(status="accepted"),
        rejected.assign(status="rejected")
    ])[[
        "p_mean", "trade_pnl", "weekly_bullish", "k_week", "d_week",
        "atr", "body", "true_range_norm", "regime", "status"
    ]].sort_index()

    out_filtered = out_dir / "ml_filtered_trades.csv"
    filtered_log.to_csv(out_filtered)
    print(f"💾 Detailed filtered-trade log saved → {out_filtered.resolve()}")


if __name__ == "__main__":
    main()
