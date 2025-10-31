# 🧠 Quant-ML-BTC

### Modular, config-driven machine learning backtesting framework for BTC

---

## 📘 Overview
Quant-ML-BTC is a **research and backtesting framework** designed to build and evaluate trading systems on Bitcoin (BTCUSDT) data.

It combines:
- **Multi-timeframe technical indicators** (Stochastic, EMA, VWAP, volatility)
- **Machine learning models** (Logistic Regression, XGBoost)
- **Labeling frameworks** (return-based or pivot-based)
- **Backtesting and metrics** (PnL, Sharpe, Profit Factor, MDD)
- **Config-driven modular design** — no hardcoded paths or parameters

You can swap data, features, labels, or models just by editing YAML config files.

---

## 🏗️ Project Structure
quant-ml-btc/
├── README.md
├── configs/                 # configuration menus (features, labeling, models)
│   ├── config_default.yaml  # main run config (references everything else)
│   ├── features.yaml        # defines which indicators to use
│   └── model_params.yaml    # stores hyperparameter sets
├── data/
│   ├── raw/                 # unprocessed OHLCV CSVs (e.g. from Binance)
│   ├── processed/           # feature-enriched data
│   └── reference/           # lookup tables, mappings, metadata
├── src/                     # Python source modules
│   ├── data_loader.py
│   ├── feature_engineer.py
│   ├── labeler.py
│   ├── model_trainer.py
│   ├── backtester.py
│   ├── metrics_reporter.py
│   ├── utils/
│   │   ├── io.py
│   │   ├── math_tools.py
│   │   └── plotting.py
│   └── pipeline.py
├── results/                 # outputs of each run
│   ├── metrics.csv
│   ├── equity_curve.csv
│   ├── trades.csv
│   ├── feature_importance.csv
│   └── charts/
├── notebooks/               # optional EDA or diagnostics
│   ├── exploratory_analysis.ipynb
│   └── feature_visuals.ipynb
└── run.py                   # CLI entrypoint — calls pipeline with config

---

## ⚙️ Installation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

---

## 🚀 Usage

python run.py --config configs/config_default.yaml

### Output files
| File | Description |
|------|--------------|
| data/processed/BTCUSDT_4h_features.parquet | Processed features and labels |
| results/equity_curve.csv | Strategy equity curve |
| results/metrics.csv | Key performance metrics |
| results/charts/ | Auto-generated plots (equity, rolling Sharpe, feature importance) |

---

## 🧩 Configuration System
All settings are stored in YAML under `/configs`.

Example:
run_name: btcusdt_4h_base
data:
  csv_path: data/raw/BTCUSDT_4h.csv
  symbol: BTCUSDT
features:
  feature_set: base_stoch_ema
labels:
  mode: return_fwd
  return_horizon_bars: 12
  threshold: 0.01
model:
  type: xgboost
  params_name: xgb_light

Edit `features.yaml` or `model_params.yaml` to change features or hyperparameters.

---

## 📊 Metrics (coming soon)
The framework will automatically calculate:
- Win rate, profit factor, and Sharpe ratio
- Drawdown and exposure statistics
- Feature importance (for XGBoost)
- Rolling performance diagnostics

---

## 🧱 Development Notes
| Area | Purpose |
|------|----------|
| src/data_loader.py | Handles loading, resampling, and missing data |
| src/feature_engineer.py | Builds indicator features from YAML menu |
| src/labeler.py | Creates supervised learning targets |
| src/model_trainer.py | Trains models & cross-validates |
| src/backtester.py | Simulates trades and equity curve |
| src/metrics_reporter.py | Summarises results |
| src/utils/ | Shared helper functions |

---

## 🧑‍💻 Contributing
1. Every module should have a clear docstring and self-contained functions.
2. Keep the README updated after **every material change** (new module, config structure, or logic refactor).
3. Always commit with descriptive messages, e.g.  
   git commit -m "feat: add pivot labeling mode to labeler.py"

---

## 🪜 Next Steps
1. Add dataset (`data/raw/BTCUSDT_4h.csv`)
2. Verify basic pipeline runs without errors
3. Implement step-by-step:
   - Feature engineering
   - Label generation
   - Model training
   - Backtesting & metrics
4. Expand documentation with actual output examples

---

Maintainer: **Tom Makin**  
Repo: **BTCAlphaGrowthTrust/quant-ml-btc**
