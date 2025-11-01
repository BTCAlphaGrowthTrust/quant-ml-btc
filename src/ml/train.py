# src/ml/train.py
"""
Train v1 machine learning models on the BTC 1W Osc + 4H PA dataset.
- Loads data from: data/datasets/btc_osc_pa_v1.parquet
- Trains Logistic Regression + XGBoost
- Walk-forward CV (time-based)
- Saves best calibrated model to: models/btc_osc_pa_v1.pkl
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier


# --- utility -----------------------------------------------------------------
def sharpe_like(y_true, y_pred_proba, threshold=0.5):
    """Approximate Sharpe: (mean win - mean loss) / std."""
    preds = (y_pred_proba >= threshold).astype(int)
    pnl = preds * (2 * y_true - 1)  # +1 for correct long, -1 for wrong
    if pnl.std() == 0:
        return 0
    return pnl.mean() / pnl.std()


# --- main --------------------------------------------------------------------
def main():
    print("🚀 Training ML models on btc_osc_pa_v1 dataset")

    data_path = Path("data/datasets/btc_osc_pa_v1.parquet")
    df = pd.read_parquet(data_path)
    print(f"📂 Loaded dataset: {len(df):,} rows, positives: {df['y'].sum()}")

    # === features ===
    feature_cols = [
        "body", "upper_wick", "lower_wick", "true_range_norm", "atr_norm",
        "kd_spread", "k_below_thr", "d_below_thr", "weekly_bullish_int",
        "dow", "hour_bin"
    ]
    X = df[feature_cols].copy()
    y = df["y"].astype(int).values
    sample_weight = df["weight"].values

    # === scaling ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === Time-based CV ===
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    print(f"🧪 Running {n_splits}-fold walk-forward validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), start=1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weight[train_idx]

        # Handle imbalance
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        cw_dict = dict(zip(classes, class_weights))

        # --- Logistic Regression ---
        logreg = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            class_weight=cw_dict,
            max_iter=200
        )
        logreg.fit(X_train, y_train, sample_weight=w_train)

        cal = CalibratedClassifierCV(logreg, cv='prefit', method='isotonic')
        cal.fit(X_val, y_val)

        y_pred_proba = cal.predict_proba(X_val)[:, 1]
        prec = precision_score(y_val, y_pred_proba >= 0.5)
        rec = recall_score(y_val, y_pred_proba >= 0.5)
        auc = roc_auc_score(y_val, y_pred_proba)
        sharpe = sharpe_like(y_val, y_pred_proba, 0.5)

        results.append({
            "fold": fold,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "sharpe": sharpe
        })
        print(f"Fold {fold}: prec={prec:.3f}, rec={rec:.3f}, auc={auc:.3f}, sharpe={sharpe:.3f}")

    res_df = pd.DataFrame(results)
    print("\n📊 CV Summary:")
    print(res_df.mean().round(3))

    # === Train final models on full data ===
    print("\n🏗️ Training final Logistic Regression + XGBoost models...")

    # Logistic Regression (calibrated)
    logreg_final = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=200
    ).fit(X_scaled, y, sample_weight=sample_weight)
    cal_final = CalibratedClassifierCV(logreg_final, cv=3, method="isotonic")
    cal_final.fit(X_scaled, y)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y) - y.sum()) / y.sum(),
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(X, y, sample_weight=sample_weight, verbose=False)

    # === Save models ===
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "scaler": scaler,
        "logreg_cal": cal_final,
        "xgb": xgb,
        "features": feature_cols
    }, out_dir / "btc_osc_pa_v1.pkl")

    meta = {
        "n_rows": int(len(df)),
        "n_pos": int(df["y"].sum()),
        "features": feature_cols,
        "cv_results": res_df.to_dict(orient="records")
    }
    with open(out_dir / "btc_osc_pa_v1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Saved models to {out_dir.resolve()}")
    print(f"🏁 Mean CV Sharpe: {res_df['sharpe'].mean():.3f}")


if __name__ == "__main__":
    main()
