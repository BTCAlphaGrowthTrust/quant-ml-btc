# src/pipeline/run_full_ml_cycle.py
"""
Run the full ML lifecycle for BTC | 1W Osc + 4H PA Execution:
  1. Build dataset
  2. Train ML models
  3. Run ML filter evaluation
  4. Generate interpretability + trade report
All outputs are timestamped under results/ml_cycle_<YYYY-MM-DD_HH-MM>/.
"""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import json


def run_step(cmd: str, label: str, log_path: Path):
    print(f"\n🚀 [{label}] Running: {cmd}")
    with open(log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    if proc.returncode == 0:
        print(f"✅ [{label}] complete.\n")
    else:
        print(f"❌ [{label}] failed (exit code {proc.returncode}).\n")
        raise SystemExit(proc.returncode)


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    root = Path("results") / f"ml_cycle_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    print(f"\n🧩 Starting full ML cycle → {root}")

    log_file = root / "run_log.txt"

    # 1️⃣ Build dataset
    run_step("python -m src.features.build_dataset", "Build Dataset", log_file)

    # 2️⃣ Train ML models
    run_step("python -m src.ml.train", "Train Models", log_file)

    # 3️⃣ Evaluate ML filter (PnL-based gating)
    run_step("python -m src.integration.ml_filter", "ML Filter Evaluation", log_file)

    # 4️⃣ Generate interpretability + trade report
    run_step("python -m src.analysis.ml_feature_insight", "Feature Insight", log_file)
    run_step("python -m src.analysis.ml_filter_report", "Filter Report", log_file)

    # 5️⃣ Collect and archive outputs into timestamped folder
    latest_results = Path("results")
    for f in latest_results.glob("ml_filter_*.csv"):
        shutil.move(f, root / f.name)
    for f in latest_results.glob("ml_filter_*.png"):
        shutil.move(f, root / f.name)
    for f in latest_results.glob("feature_importance/*.png"):
        shutil.copy(f, root / f.name)
    for f in latest_results.glob("feature_importance/*.csv"):
        shutil.copy(f, root / f.name)
    for f in latest_results.glob("ml_filter_trade_report.*"):
        shutil.move(f, root / f.name)
    for f in latest_results.glob("ml_filter_reason_counts.png"):
        shutil.move(f, root / f.name)

    # 6️⃣ Write summary metadata
    meta = {
        "timestamp": ts,
        "results_dir": str(root.resolve()),
        "steps": ["dataset", "train", "filter", "insight", "report"],
    }
    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"🏁 ML cycle complete. Outputs saved under → {root}")


if __name__ == "__main__":
    main()
