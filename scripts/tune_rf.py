# scripts/tune_rf.py
import argparse
from pathlib import Path

import yaml

from code_usg.tuning import tune_random_forest


def main():
    ap = argparse.ArgumentParser(description="Tune RandomForest hyperparameters and save results.")
    ap.add_argument("--config", "-c", default="conf/params.yml", help="Path to YAML config.")
    ap.add_argument(
        "--out", "-o", default="reports", help="Output directory for tuned params and CV results."
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    best = tune_random_forest(cfg, Path(args.out))
    print("Best RF parameters:", best)


if __name__ == "__main__":
    main()
