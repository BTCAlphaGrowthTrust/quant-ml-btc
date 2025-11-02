# 🧠 Quant-ML-BTC
**Modular, config-driven machine-learning backtesting framework for BTC.**

---

## 🧩 Overview
Quant-ML-BTC provides a complete end-to-end research stack for BTC algorithmic systems:
- Multi-timeframe signal generation (1M / 1W / 1D / 4H)
- Config-driven backtesting and trade simulation
- Machine learning conviction scoring (Logistic Regression, XGBoost)
- Walk-forward validation for robustness
- Easy expansion with plug-and-play strategy classes

Designed and maintained by **Tom Makin (BTC Alpha Growth Trust)**.

---

## 📁 Project Structure

quant-ml-btc/
├── README.md
├── configs/
│ ├── config_default.yaml
│ ├── features.yaml
│ └── model_params.yaml
├── data/
│ ├── raw/ # downloaded OHLCV data
│ ├── processed/ # feature-enriched frames
│ ├── datasets/ # ML training datasets (.parquet)
│ └── results/ # conviction curves & summaries
├── models/ # saved ML models (.pkl)
├── results/ # backtest logs & trade CSVs
├── src/
│ ├── features/ # dataset builder logic
│ ├── ml/ # model training + calibration
│ ├── strategies/ # rule-based strategy modules
│ │ ├── base_strategy.py
│ │ ├── btc_1m_stoch_4h_pa.py
│ │ ├── weekly_oscillator_pa.py
│ │ ├── btc_1d_golden_4h_pa.py
│ │ └── btc_4h_golden_4h_pa.py
│ └── pipeline.py
└── run.py

## ⚙️ Installation
```bash
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

🚀 Core Capabilities
Layer	Function	Example
Strategy	Defines rules (MA, Oscillator, PA)	src/strategies/
Feature Builder	Generates trade labels + indicators	python -m src.features.build_dataset --strategy <name>
ML Trainer	Fits calibrated ML models per strategy	python -m src.ml.train --dataset <name>_v1
Conviction Curve	Produces confidence-weighted probabilities	Saved in /data/results/
Registry	Unified strategy loader	load_strategy(name)
🧠 Current Strategies
Strategy	File	Type	Purpose
tom_makin_1m_osc_4h_pa	btc_1m_stoch_4h_pa.py	Momentum Oscillator	Monthly stochastic bias + 4H execution
tom_makin_1w_osc_4h_pa	weekly_oscillator_pa.py	Swing Oscillator	Weekly stochastic bias + 4H execution
tom_makin_1d_golden_4h_pa	btc_1d_golden_4h_pa.py	Trend-Following	Daily EMA50/200 golden cross + 4H breakout
tom_makin_4h_golden_4h_pa	btc_4h_golden_4h_pa.py	Momentum Continuation	4H EMA50/200 golden cross + 4H breakout

All four run through the same ML + conviction pipeline.

🧪 One-Click Pipeline Commands

🔹 1. Build Datasets

python -m src.features.build_dataset --strategy tom_makin_1m_osc_4h_pa
python -m src.features.build_dataset --strategy tom_makin_1w_osc_4h_pa
python -m src.features.build_dataset --strategy tom_makin_1d_golden_4h_pa
python -m src.features.build_dataset --strategy tom_makin_4h_golden_4h_pa

Each produces a timestamped dataset in:

data/datasets/<strategy_name>_v1.parquet

🔹 2. Train Conviction Models
python -m src.ml.train --dataset tom_makin_1m_osc_4h_pa_v1
python -m src.ml.train --dataset tom_makin_1w_osc_4h_pa_v1
python -m src.ml.train --dataset tom_makin_1d_golden_4h_pa_v1
python -m src.ml.train --dataset tom_makin_4h_golden_4h_pa_v1


Each outputs:
models/<strategy_name>_v1.pkl
data/results/<strategy_name>_v1_conviction_curve.csv
results/trade_log_<strategy_name>_timestamp.csv

🔹 3. Inspect Conviction Curves
ls data/results/*conviction_curve.csv

🔹 4. (Optional) Backtest All Strategies
python -m src.backtest.run --strategy all

🧬 Next Steps
- 🧩 Ensemble the 4 conviction models (Meta-Alpha Combiner)
- 📊 Add unified performance dashboard (equity curves, risk overlays)
- 🔔 Connect TradingView webhooks for live execution filtering

🚑 Recovery Procedure

If your environment resets or Codespace is wiped:

# 1️⃣ Recreate the environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2️⃣ Pull the repo
git clone https://github.com/BTCAlphaGrowthTrust/quant-ml-btc.git
cd quant-ml-btc

# 3️⃣ Run any strategy again
python -m src.features.build_dataset --strategy tom_makin_1m_osc_4h_pa
python -m src.ml.train --dataset tom_makin_1m_osc_4h_pa_v1

👨‍💻 Maintainer

Tom Makin
BTC Alpha Growth Trust
github.com/BTCAlphaGrowthTrust/quant-ml-btc


---

✅ Just copy and paste that entire block into `README.md` (replacing the old content),  
then run:

```bash
git add README.md
git commit -m "🧠 Updated README to document 4-strategy ML pipeline and commands"
git push origin main