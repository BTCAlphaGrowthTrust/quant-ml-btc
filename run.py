import argparse
from src.pipeline import run
if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_default.yaml")
    args=p.parse_args()
    bt=run(args.config)
    print("Done. Last equity:", float(bt["equity"].iloc[-1]))
