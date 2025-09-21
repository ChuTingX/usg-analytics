#!/usr/bin/env python
import argparse
from code_usg.pipeline import run_analysis

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/params.yml")
    args = ap.parse_args()
    out = run_analysis(args.config)
    for k, v in out.items():
        print(f"{k}: {v}")