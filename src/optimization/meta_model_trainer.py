# src/optimization/meta_model_trainer.py
"""
Meta-model learning for BTC | 1W Osc + 4H PA Execution parameter sweeps.

- Aggregates all sweep CSVs under data/param_sweeps/
- Trains regressors to predict Sharpe & Profit Factor from parameters
- Produces lookback-sliced heatmaps of Sharpe vs (stoch_threshold x atr_multiplier)
- Ranks top parameter zones and saves models + plots + CSVs

Usage:
    python -m src.optimization.meta_model_trainer
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# -----------------------------
# Helpers
# -----------------------------
def load_all_sweeps(sweep_dir: Path) -> pd.DataFrame:
    files = sorted(sweep_dir.glob("btc_osc_pa_sweep_*.csv"))
    if not files:
        raise FileNotFoundError(f"No sweep files found in {sweep_dir.resolve()}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["sweep_file"] = f.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Normalise column types
    out["stoch_threshold"] = out["stoch_threshold"].astype(int)
    out["lookback_bars"] = out["lookback_bars"].astype(int)
    out["atr_multiplier"] = out["atr_multiplier"].astype(float)
    # De-percent win_rate if needed
    if out["win_rate"].dtype == object:
        # Accept forms like '48%' or '0.48'
        out["win_rate"] = out["win_rate"].astype(str).str.replace("%", "", regex=False)
        out["win_rate"] = pd.to_numeric(out["win_rate"], errors="coerce")
        out.loc[out["win_rate"] > 1.5, "win_rate"] /= 100.0
    return out


def make_output_dir() -> Path:
    root = Path("results/optimization")
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = root / f"meta_model_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def fit_regressor(X: pd.DataFrame, y: pd.Series, model_name: str):
    """
    Trains two models (RF + XGB) and returns the best by R^2 on validation.
    Saves the best model pipeline as a .pkl.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    num_cols = X.columns.tolist()
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)],
        remainder="drop",
    )

    rf = Pipeline(
        steps=[
            ("pre", pre),
            ("rf", RandomForestRegressor(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=2
            )),
        ]
    )

    xgb = Pipeline(
        steps=[
            ("pre", pre),
            ("xgb", XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )),
        ]
    )

    models = [("rf", rf), ("xgb", xgb)]
    scores = []

    for name, pipe in models:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        r2 = r2_score(y_val, pred)
        mae = mean_absolute_error(y_val, pred)
        scores.append((name, r2, mae, pipe))

    # Pick best by R^2
    best = sorted(scores, key=lambda t: t[1], reverse=True)[0]
    best_name, best_r2, best_mae, best_pipe = best
    return best_name, best_r2, best_mae, best_pipe


def save_permutation_importance(pipe, X: pd.DataFrame, y: pd.Series, out_dir: Path, tag: str):
    # Compute on val-sized split to keep it snappy
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipe.fit(X_train, y_train)
    result = permutation_importance(
        pipe, X_val, y_val, n_repeats=25, random_state=42, n_jobs=-1
    )

    imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    imp.to_csv(out_dir / f"feature_importance_{tag}.csv")

    # Plot
    plt.figure(figsize=(7, 4))
    imp.head(10).iloc[::-1].plot(kind="barh")
    plt.title(f"Top 10 Feature Importance (Permutation) – {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"feature_importance_{tag}.png", dpi=130)
    plt.close()


def heatmaps_by_lookback(df: pd.DataFrame, out_dir: Path):
    """
    For each lookback, draw a heatmap of mean Sharpe over
    (stoch_threshold x atr_multiplier).
    """
    lkb_vals = sorted(df["lookback_bars"].unique())
    for lkb in lkb_vals:
        sub = df[df["lookback_bars"] == lkb].copy()
        if sub.empty:
            continue
        piv = sub.pivot_table(
            index="stoch_threshold",
            columns="atr_multiplier",
            values="sharpe",
            aggfunc="mean",
        ).sort_index(ascending=True)
        plt.figure(figsize=(7, 5))
        # Manual heatmap using imshow + axis ticks (no seaborn)
        ax = plt.gca()
        mat = piv.values
        im = ax.imshow(mat, aspect="auto", origin="lower")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([str(c) for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([str(i) for i in piv.index])
        ax.set_xlabel("ATR Multiplier")
        ax.set_ylabel("Stoch Threshold")
        plt.title(f"Sharpe Heatmap (lookback={lkb})")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_sharpe_lookback_{lkb}.png", dpi=130)
        plt.close()


def rank_top_zones(df: pd.DataFrame, out_dir: Path, k: int = 15):
    """
    Save a CSV of the best parameter rows by Sharpe (with tie metrics).
    """
    cols = [
        "stoch_threshold", "lookback_bars", "atr_multiplier",
        "trades", "win_rate", "profit_factor", "sharpe", "gross_pnl", "sweep_file"
    ]
    top = df.sort_values("sharpe", ascending=False)[cols].head(k)
    top.to_csv(out_dir / "top_parameter_zones.csv", index=False)
    return top


# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = make_output_dir()
    print(f"🧠 Meta-model training → {out_dir}")

    # 1) Load data
    sweep_dir = Path("data/param_sweeps")
    sweeps = load_all_sweeps(sweep_dir)
    print(f"📦 Loaded sweeps: {len(sweeps)} rows from {sweep_dir}")

    # 2) Basic X, y
    # Core param features; you can add more later (e.g., regime features)
    X = sweeps[["stoch_threshold", "lookback_bars", "atr_multiplier", "trades"]].copy()
    # Targets
    y_sharpe = sweeps["sharpe"].astype(float)
    y_pf = sweeps["profit_factor"].astype(float)

    # 3) Fit meta-models
    best_name_s, r2_s, mae_s, model_sharpe = fit_regressor(X, y_sharpe, "sharpe")
    best_name_p, r2_p, mae_p, model_pf = fit_regressor(X, y_pf, "profit_factor")

    # 4) Save models
    joblib.dump(model_sharpe, out_dir / "meta_model_sharpe.pkl")
    joblib.dump(model_pf, out_dir / "meta_model_profit_factor.pkl")

    print(f"🎯 Best (Sharpe): {best_name_s} | R^2={r2_s:.3f} | MAE={mae_s:.4f}")
    print(f"🎯 Best (PF):     {best_name_p} | R^2={r2_p:.3f} | MAE={mae_p:.4f}")

    # 5) Permutation importance (saved as CSV + PNG)
    save_permutation_importance(model_sharpe, X, y_sharpe, out_dir, tag="sharpe")
    save_permutation_importance(model_pf, X, y_pf, out_dir, tag="profit_factor")

    # 6) Heatmaps per lookback
    heatmaps_by_lookback(sweeps, out_dir)

    # 7) Rank top parameter zones & save
    top = rank_top_zones(sweeps, out_dir, k=20)
    print("\n🏆 Top parameter zones by Sharpe:")
    print(top.to_string(index=False))

    # 8) Quick summary file
    meta = {
        "rows": int(len(sweeps)),
        "files": sorted(list({f for f in sweeps["sweep_file"].unique()})),
        "best_models": {
            "sharpe": {"name": best_name_s, "r2": float(r2_s), "mae": float(mae_s)},
            "profit_factor": {"name": best_name_p, "r2": float(r2_p), "mae": float(mae_p)},
        },
    }
    (out_dir / "summary.json").write_text(
        pd.Series(meta, dtype="object").to_json(indent=2), encoding="utf-8"
    )

    print(f"\n✅ Meta-model artifacts saved → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
