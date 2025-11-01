# src/analysis/ml_filter_report.py
"""
Explain WHAT the ML filter accepted/rejected and WHY.

Outputs:
  • results/ml_filter_trade_report.csv  (tabular per-trade audit)
  • results/ml_filter_trade_report.txt  (human-readable per-trade notes)
  • results/ml_filter_reason_counts.png (bar chart of reason frequencies)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.data_loader import get_btc_data
from src.strategies.weekly_oscillator_pa import WeeklyOscillatorPA


# ---------------------------
# Config
# ---------------------------
THRESHOLD = 0.47  # τ chosen from PnL-based sweep


# ---------------------------
# Utilities
# ---------------------------
def load_models():
    bundle = joblib.load(Path("models/btc_osc_pa_v1.pkl"))
    return bundle["scaler"], bundle["logreg_cal"], bundle["xgb"], bundle["features"]


def get_logreg_coefficients(calibrated_model):
    """Version-proof extraction of LR coefficients from CalibratedClassifierCV."""
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        try:
            return calibrated_model.calibrated_classifiers_[0].estimator.coef_[0]
        except Exception:
            pass
    if hasattr(calibrated_model, "_calibrated_classifiers_"):
        try:
            return calibrated_model._calibrated_classifiers_[0].estimator.coef_[0]
        except Exception:
            pass
    if hasattr(calibrated_model, "base_estimator"):  # very old sklearn
        return calibrated_model.base_estimator.coef_[0]
    raise AttributeError("Unable to access logistic regression coefficients.")


def safe_shap_values(explainer, X):
    """Return a plain numpy array of SHAP values (handles Explanation objects)."""
    sv = explainer(X)
    try:
        return sv.values if hasattr(sv, "values") else np.array(sv)
    except Exception:
        return np.array(sv)


def percentile_label(val, q25, q75, name, low_better=None):
    """
    Make a readable label based on quartiles.
    low_better: True (small is good), False (large is good), None (mid is good)
    """
    if val <= q25:
        tag = "low"
        vibe = "✅" if low_better is True else ("⚠️" if low_better is False else "⚠️")
    elif val >= q75:
        tag = "high"
        vibe = "✅" if low_better is False else ("⚠️" if low_better is True else "⚠️")
    else:
        tag = "moderate"
        vibe = "✅" if low_better is None else "➖"
    return f"{vibe} {name}: {tag}"


# ---------------------------
# Main
# ---------------------------
def main():
    out_dir = Path("results"); out_dir.mkdir(parents=True, exist_ok=True)
    rpt_dir = out_dir / "feature_importance"; rpt_dir.mkdir(parents=True, exist_ok=True)

    print("🧾 ML Filter Trade Report...")

    # 1) Load dataset + models
    ds_path = Path("data/datasets/btc_osc_pa_v1.parquet")
    df = pd.read_parquet(ds_path)  # index = timestamp (UTC)
    scaler, logreg_cal, xgb, feat_cols = load_models()

    # 2) Score probabilities
    X = df[feat_cols].values
    Xs = scaler.transform(X)
    p_lr = logreg_cal.predict_proba(Xs)[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1]
    df["p_mean"] = (p_lr + p_xgb) / 2

    # 3) Compute realized PnL joining (recreate baseline trades)
    raw = get_btc_data()
    raw.index = pd.to_datetime(raw["timestamp"], utc=True)
    strat = WeeklyOscillatorPA(use_forming_week=True, make_plot=False, save_trades_csv=False)
    _ = strat.generate_signals(raw)
    trades = pd.DataFrame(strat.trades)
    trades = trades.dropna(subset=["entry_time"]).copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["pnl"] = trades["pnl"].fillna(0.0)
    entry_pnl = trades.set_index("entry_time")["pnl"]

    # Restrict to actual entry bars (y==1) and attach PnL
    entries = df[df["y"] == 1].copy()
    entries["trade_pnl"] = np.nan
    common = entries.index.intersection(entry_pnl.index)
    entries.loc[common, "trade_pnl"] = entry_pnl.loc[common].values

    # 4) Decide accepted vs rejected with threshold τ
    entries["accepted"] = (entries["p_mean"] >= THRESHOLD).astype(int)

    # 5) SHAP per-entry to see *why*
    print("⚙️ Computing SHAP values for entry bars...")
    explainer = shap.TreeExplainer(xgb)
    shap_matrix = safe_shap_values(explainer, entries[feat_cols].values)  # shape: (n_entries, n_features)

    # For later: top +/- drivers per trade
    shap_df = pd.DataFrame(shap_matrix, columns=feat_cols, index=entries.index)

    # 6) Human-readable reasons using simple rules + top SHAP features
    # Quartiles for continuous features (computed over entry bars)
    kd_q = entries["kd_spread"].quantile([0.25, 0.75]).to_dict()
    body_q = entries["body"].quantile([0.25, 0.75]).to_dict()
    tr_q = entries["true_range_norm"].quantile([0.25, 0.75]).to_dict()
    atr_q = entries["atr_norm"].quantile([0.25, 0.75]).to_dict()

    # Dow heuristics (dataset uses 0..6, where 0≈Mon, 6≈Sun; early-week is good)
    def dow_tag(dow_int):
        if dow_int in [0, 1, 2]:
            return "✅ early-week timing"
        if dow_int in [5, 6]:
            return "⚠️ weekend timing"
        return "➖ mid-week timing"

    human_summaries = []
    reason_bucket_counts = {}

    for ts, row in entries.iterrows():
        bits = []
        # Rules on discrete flags
        if int(row.get("weekly_bullish_int", 0)) == 1:
            bits.append("✅ weekly bias bullish")
            reason_bucket_counts["weekly_bullish"] = reason_bucket_counts.get("weekly_bullish", 0) + 1
        else:
            bits.append("⚠️ weekly bias not bullish")
            reason_bucket_counts["weak_bias"] = reason_bucket_counts.get("weak_bias", 0) + 1

        if int(row.get("k_below_thr", 0)) and int(row.get("d_below_thr", 0)):
            bits.append("✅ early-cycle: K & D below threshold")
            reason_bucket_counts["early_cycle"] = reason_bucket_counts.get("early_cycle", 0) + 1
        elif int(row.get("k_below_thr", 0)) or int(row.get("d_below_thr", 0)):
            bits.append("➖ partial early-cycle (one below threshold)")
            reason_bucket_counts["partial_cycle"] = reason_bucket_counts.get("partial_cycle", 0) + 1
        else:
            bits.append("⚠️ late-cycle: K & D above threshold")
            reason_bucket_counts["late_cycle"] = reason_bucket_counts.get("late_cycle", 0) + 1

        # Continuous heuristics (quartile-based)
        bits.append(percentile_label(row["kd_spread"], kd_q[0.25], kd_q[0.75], "K–D spread", low_better=None))
        bits.append(percentile_label(row["body"], body_q[0.25], body_q[0.75], "candle body", low_better=True))
        bits.append(percentile_label(row["true_range_norm"], tr_q[0.25], tr_q[0.75], "true range", low_better=True))
        bits.append(percentile_label(row["atr_norm"], atr_q[0.25], atr_q[0.75], "ATR", low_better=True))
        bits.append(dow_tag(int(row.get("dow", -1))))

        # Top +/- SHAP drivers
        row_shap = shap_df.loc[ts]
        top_pos = row_shap.sort_values(ascending=False).head(2)
        top_neg = row_shap.sort_values(ascending=True).head(2)

        pos_txt = ", ".join([f"{k} (+{v:.2f})" for k, v in top_pos.items()])
        neg_txt = ", ".join([f"{k} ({v:.2f})" for k, v in top_neg.items()])

        status = "ACCEPTED" if row["accepted"] == 1 else "REJECTED"
        ptxt = f"p={row['p_mean']:.2f}"
        pnl_txt = "" if pd.isna(row["trade_pnl"]) else f" | pnl={row['trade_pnl']:.2f}"

        summary = (
            f"{'🟢' if status=='ACCEPTED' else '🔴'} {ts} | {status} | {ptxt}{pnl_txt}\n"
            f"   • {'; '.join(bits)}\n"
            f"   • Top + drivers: {pos_txt}\n"
            f"   • Top − drivers: {neg_txt}\n"
        )
        human_summaries.append(summary)

    # 7) Assemble CSV table
    report = entries.copy()
    report["status"] = report["accepted"].map({1: "accepted", 0: "rejected"})
    # Include only the most useful columns + probabilities
    keep_cols = [
        "status", "p_mean", "trade_pnl",
        "weekly_bullish_int", "k_below_thr", "d_below_thr",
        "kd_spread", "dow", "hour_bin", "body", "true_range_norm", "atr_norm"
    ]
    missing = [c for c in keep_cols if c not in report.columns]
    for c in missing:
        report[c] = np.nan
    report = report[keep_cols]

    # Add SHAP top +/- columns
    # (Use the same loop again for clarity)
    top_pos_list, top_neg_list = [], []
    for ts in report.index:
        row_shap = shap_df.loc[ts]
        pos = row_shap.sort_values(ascending=False).head(2)
        neg = row_shap.sort_values(ascending=True).head(2)
        top_pos_list.append("; ".join([f"{k}:+{v:.2f}" for k, v in pos.items()]))
        top_neg_list.append("; ".join([f"{k}:{v:.2f}" for k, v in neg.items()]))
    report["top_pos_drivers"] = top_pos_list
    report["top_neg_drivers"] = top_neg_list

    # 8) Save files
    csv_path = out_dir / "ml_filter_trade_report.csv"
    txt_path = out_dir / "ml_filter_trade_report.txt"
    report.to_csv(csv_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(human_summaries))

    print(f"💾 CSV  → {csv_path.resolve()}")
    print(f"💾 Text → {txt_path.resolve()}")

    # 9) Reason frequency plot
    if reason_bucket_counts:
        keys = list(reason_bucket_counts.keys())
        vals = [reason_bucket_counts[k] for k in keys]
        plt.figure(figsize=(8, 4))
        plt.bar(keys, vals)
        plt.title("Common reason tags (entries)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        png_path = out_dir / "ml_filter_reason_counts.png"
        plt.savefig(png_path, dpi=130)
        plt.close()
        print(f"🖼️ Saved → {png_path.resolve()}")

    print("✅ ML filter report complete.")


if __name__ == "__main__":
    main()
