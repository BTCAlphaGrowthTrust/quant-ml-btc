# src/ml/train.py
"""
Conviction-Based ML Trainer (v2.1)
---------------------------------
Trains probabilistic models to detect rare, high-value trade events.
Now robust to folds with only one class.

Usage:
    python -m src.ml.train --dataset tom_makin_1m_osc_4h_pa_v1
"""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


# === Utility =================================================================

def pseudo_pnl(y_true, y_pred, reward=1.0, penalty=-0.5):
    pnl = np.where(y_pred == 1, np.where(y_true == 1, reward, penalty), 0)
    return pnl.sum(), pnl.mean(), pnl.std(), pnl


def evaluate_thresholds(y_true, proba, thresholds):
    records = []
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        pnl_sum, pnl_mean, pnl_std, _ = pseudo_pnl(y_true, preds)
        trades = preds.sum()
        sharpe = 0 if pnl_std == 0 else pnl_mean / pnl_std
        records.append({
            "threshold": thr,
            "trades": int(trades),
            "mean_pnl": round(pnl_mean, 4),
            "total_pnl": round(pnl_sum, 2),
            "sharpe_like": round(sharpe, 3)
        })
    return pd.DataFrame(records)


# === Main ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset filename (without path). Example: tom_makin_1m_osc_4h_pa_v1")
    args = parser.parse_args()
    dataset_name = args.dataset.replace(".parquet", "")
    data_path = Path(f"data/datasets/{dataset_name}.parquet")

    print(f"🚀 Training conviction-based ML models on {dataset_name}")
    df = pd.read_parquet(data_path)
    print(f"📂 Loaded dataset: {len(df):,} rows | positives: {df['y'].sum()}")

    # --- features -------------------------------------------------------------
    feature_cols = [
        "body", "upper_wick", "lower_wick", "true_range_norm",
        "kd_spread", "k_below_thr", "d_below_thr", "bullish_int",
        "dow", "hour_bin"
    ]
    X = df[feature_cols].copy().fillna(0)
    y = df["y"].astype(int).values
    sample_weight = df.get("weight", pd.Series(1, index=df.index)).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- CV -------------------------------------------------------------------
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"🧪 Performing {n_splits}-fold walk-forward CV...")

    fold_metrics = []
    all_val_probs, all_val_true = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), start=1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weight[train_idx]

        # Skip if only one class in training data
        if len(np.unique(y_train)) < 2:
            print(f"⚠️  Skipping fold {fold}: only one class present in training data.")
            continue

        # Compute balanced class weights
        classes = np.unique(y_train)
        cw_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        cw = dict(zip(classes, cw_values))

        # Train calibrated logistic regression
        logreg = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear",
            class_weight=cw, max_iter=300
        )
        logreg.fit(X_train, y_train, sample_weight=w_train)
        cal = CalibratedClassifierCV(logreg, cv='prefit', method='isotonic')
        cal.fit(X_val, y_val)

        y_pred_proba = cal.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        all_val_probs.extend(y_pred_proba)
        all_val_true.extend(y_val)
        fold_metrics.append({"fold": fold, "auc": auc})
        print(f"Fold {fold}: AUC={auc:.3f}")

    if not fold_metrics:
        raise RuntimeError("No valid folds contained both classes. Try fewer splits or resampling.")

    print("\n📊 CV Summary:")
    res_df = pd.DataFrame(fold_metrics)
    print(res_df.mean().round(3))

    # --- Train final models ---------------------------------------------------
    print("\n🏗️ Training final Logistic Regression + XGBoost models...")
    logreg_final = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear",
        class_weight="balanced", max_iter=300
    ).fit(X_scaled, y, sample_weight=sample_weight)
    cal_final = CalibratedClassifierCV(logreg_final, cv=3, method="isotonic")
    cal_final.fit(X_scaled, y)

    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(len(y) - y.sum()) / (y.sum() + 1e-9),
        eval_metric="logloss", n_jobs=-1, random_state=42
    )
    xgb.fit(X, y, sample_weight=sample_weight, verbose=False)

    # --- Evaluate conviction curve -------------------------------------------
    all_val_probs = np.array(all_val_probs)
    all_val_true = np.array(all_val_true)
    thresholds = np.linspace(0.05, 0.95, 19)
    curve = evaluate_thresholds(all_val_true, all_val_probs, thresholds)

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / f"{dataset_name}_conviction_curve.csv"
    curve.to_csv(curve_path, index=False)
    print(f"💾 Conviction curve saved → {curve_path}")

    # --- Plot ----------------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(curve["threshold"], curve["mean_pnl"], marker="o", label="Mean PnL")
    plt.plot(curve["threshold"], curve["sharpe_like"], marker="x", label="Sharpe-like")
    plt.xlabel("Probability Threshold")
    plt.title("Conviction vs Profitability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_conviction_curve.png", dpi=150)
    plt.close()
    print(f"📈 Plot saved → {dataset_name}_conviction_curve.png")

    # --- Save models ---------------------------------------------------------
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{dataset_name}.pkl"
    joblib.dump({
        "scaler": scaler,
        "logreg_cal": cal_final,
        "xgb": xgb,
        "features": feature_cols
    }, model_file)
    print(f"💾 Saved models → {model_file}")

    meta = {
        "dataset": dataset_name,
        "n_rows": len(df),
        "n_pos": int(df["y"].sum()),
        "cv_auc_mean": float(res_df["auc"].mean()),
        "conviction_curve": str(curve_path)
    }
    with open(model_dir / f"{dataset_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"🏁 Mean CV AUC: {meta['cv_auc_mean']:.3f}")
    print("✅ Conviction-based training complete.")


if __name__ == "__main__":
    main()
