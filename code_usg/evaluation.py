import platform
import subprocess
import time

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_pred_vs_actual(y_true, y_pred, out_png: str, title: str = "Predicted vs Actual"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims)
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_residuals(y_true, y_pred, out_png: str, title: str = "Residuals"):
    res = y_true - y_pred
    plt.figure(figsize=(8, 3))
    plt.plot(res)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_metadata_dict():
    pkg_versions = {}
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import yaml

        pkg_versions = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "yaml": yaml.__version__,
        }
    except Exception:
        pass
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = None
    return {
        "packages": pkg_versions,
        "git_commit": git_hash,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
