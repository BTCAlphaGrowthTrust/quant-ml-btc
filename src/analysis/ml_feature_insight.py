# src/analysis/ml_feature_insight.py
"""
Feature importance and SHAP analysis for BTC | 1W Osc + 4H PA ML filter.
Explains which features drive the model's trade confidence
and visualises their effects on accepted vs rejected trades.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def load_models():
    path = Path("models/btc_osc_pa_v1.pkl")
    bundle = joblib.load(path)
    return bundle["scaler"], bundle["logreg_cal"], bundle["xgb"], bundle["features"]


def get_logreg_coefficients(calibrated_model):
    """
    Retrieve coefficients from a CalibratedClassifierCV, regardless of sklearn version.
    """
    # Modern sklearn (≥1.3)
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        try:
            inner_est = calibrated_model.calibrated_classifiers_[0].estimator
            return inner_est.coef_[0]
        except Exception as e:
            print(f"⚠️ Could not access inner estimator via calibrated_classifiers_: {e}")

    # Older versions
    if hasattr(calibrated_model, "_calibrated_classifiers_"):
        try:
            inner_est = calibrated_model._calibrated_classifiers_[0].estimator
            return inner_est.coef_[0]
        except Exception as e:
            print(f"⚠️ Could not access inner estimator via _calibrated_classifiers_: {e}")

    # Fallback (very old sklearn)
    if hasattr(calibrated_model, "base_estimator"):
        try:
            return calibrated_model.base_estimator.coef_[0]
        except Exception as e:
            print(f"⚠️ Could not access base_estimator: {e}")

    raise AttributeError("❌ Unable to extract logistic regression coefficients from CalibratedClassifierCV.")


# -----------------------------
# Main execution
# -----------------------------
def main():
    print("🧠 ML Feature Insight Analysis...")

    out_dir = Path("results/feature_importance")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Load dataset + models
    ds_path = Path("data/datasets/btc_osc_pa_v1.parquet")
    df = pd.read_parquet(ds_path)
    scaler, logreg_cal, xgb, feat_cols = load_models()
    print(f"📂 Dataset loaded → {len(df):,} rows | features={len(feat_cols)}")

    X = df[feat_cols].values
    Xs = scaler.transform(X)

    # 2️⃣ Compute feature importances
    print("🔍 Calculating feature importances...")

    # XGBoost importances
    xgb_imp = pd.Series(
        xgb.feature_importances_, index=feat_cols, name="xgb_importance"
    ).sort_values(ascending=False)

    # Logistic regression coefficients
    lr_coef = get_logreg_coefficients(logreg_cal)
    lr_imp = pd.Series(
        np.abs(lr_coef), index=feat_cols, name="logreg_weight"
    ).sort_values(ascending=False)

    # Combine and rank
    fi_df = pd.concat([xgb_imp, lr_imp], axis=1).fillna(0)
    fi_df["avg_rank"] = fi_df.rank(ascending=False).mean(axis=1)
    fi_df.sort_values("avg_rank", inplace=True)

    csv_path = out_dir / "feature_importance_summary.csv"
    fi_df.to_csv(csv_path)
    print(f"💾 Feature importance saved → {csv_path.resolve()}")

    # Plot top 10
    top = fi_df.head(10)
    plt.figure(figsize=(8, 4))
    top.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances (XGB + LogReg)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance_top10.png", dpi=130)
    plt.close()
    print("🖼️ Saved → feature_importance_top10.png")

    # 3️⃣ SHAP analysis (XGBoost only)
    print("⚙️ Running SHAP explainability on XGBoost...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X)

    # Handle SHAP Explanation object safely
    shap_array = (
        shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    )

    # Global summary plot
    shap.summary_plot(shap_array, features=df[feat_cols], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=130)
    plt.close()
    print("🖼️ Saved → shap_summary.png")

    # Dependence plots for top 5 features
    top5 = xgb_imp.head(5).index.tolist()
    for f in top5:
        try:
            shap.dependence_plot(f, shap_array, df[feat_cols], show=False)
            plt.tight_layout()
            plt.savefig(out_dir / f"shap_dependence_{f}.png", dpi=130)
            plt.close()
        except Exception as e:
            print(f"⚠️ Skipped {f} due to SHAP plotting issue: {e}")

    print(f"🖼️ Saved dependence plots for top 5 features: {top5}")
    print("✅ Feature insight analysis complete.")


if __name__ == "__main__":
    main()
