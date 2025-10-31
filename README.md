# 🧠 Quant-ML-BTC
Modular, config-driven machine-learning backtesting framework for BTC.

## Overview
Quant-ML-BTC lets you build, test, and compare trading systems on BTCUSDT using:
- Multi-timeframe indicators (EMA, Stochastic, RSI, volatility)
- ML models (Logistic Regression, XGBoost)
- Configurable labeling (return-based, volatility-scaled)
- Backtesting + metrics (PnL, Sharpe, Profit Factor, MDD)
- Experiment logging to timestamped run folders

You control the pipeline by editing YAML configs (no hardcoded params).

## Project Structure
quant-ml-btc/
├── README.md
├── configs/
│   ├── config_default.yaml       # main run config
│   ├── features.yaml             # feature menus (optional)
│   └── model_params.yaml         # model params (optional)
├── data/
│   ├── raw/                      # downloaded OHLCV
│   ├── processed/                # feature-enriched data
│   └── reference/                # lookups/metadata
├── src/
│   ├── data_loader.py            # Binance loader & caching
│   ├── feature_engineer.py       # RSI, volatility, EMA cross, etc.
│   ├── labeler.py                # dynamic volatility-based labels
│   ├── metrics_reporter.py       # Sharpe, Sortino, PF, MDD
│   ├── strategies/               # rule-based systems
│   │   ├── base_strategy.py
│   │   └── (add your strategies here)
│   └── pipeline.py               # end-to-end orchestrator
├── results/
│   ├── equity_curve.csv
│   ├── metrics.csv
│   ├── backtest_dataset.csv
│   └── runs/run_YYYY-MM-DD_NNN/  # per-experiment artifacts
├── docs/
│   └── strategy_builder_context.txt
└── run.py                        # CLI entrypoint

## Installation
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Usage
python run.py --config configs/config_default.yaml

Outputs (also mirrored under results/runs/...):
- results/equity_curve.csv
- results/metrics.csv
- results/backtest_dataset.csv

## Config (example)
data:
  symbol: BTCUSDT
  timeframe: 4h
features:
  ema_periods: [50, 200]
  stochastic: [14, 6, 3]
label:
  horizon: 12
  threshold: 0.01
model:
  type: xgboost
  n_estimators: 150
  max_depth: 4
  learning_rate: 0.05
backtest:
  initial_capital: 100000

## Strategy Builder
See docs/strategy_builder_context.txt for how to author strategies under src/strategies/.

Maintainer: Tom Makin
Repo: BTCAlphaGrowthTrust/quant-ml-btc
