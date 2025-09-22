import json
import os

import yaml

from code_usg.pipeline import run_analysis, run_eda

from .conftest import write_synth_csv


def test_end_to_end_pipeline(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "proc"
    rep = tmp_path / "rep"
    fig = rep / "fig"
    raw.mkdir()
    proc.mkdir()
    fig.mkdir(parents=True, exist_ok=True)
    csv_path = write_synth_csv(str(tmp_path))

    cfg_yaml = {
        "data": {
            "input_csv": str(csv_path),
            "date_column": "Date",
            "parse_dates": True,
            "index_as_date": True,
            "trend_suffix": "_Trend",
        },
        "features": {
            "target": "Adj Close",
            "variables": ["Adj Close", "SP_close", "DJ_close", "USO_Close"],
        },
        "cv": {"strategy": "kfold", "n_splits": 3, "shuffle": True, "random_state": 1},
        "models": {
            "random_forest": {"n_estimators": 20, "random_state": 1},
            "lasso": {"alphas_logspace": [-4, 1, 5], "max_iter": 5000, "random_state": 1},
            "ridge": {"alphas_logspace": [-4, 1, 5], "max_iter": 5000, "random_state": 1},
        },
        "output": {"processed_dir": str(proc), "reports_dir": str(rep), "figures_dir": str(fig)},
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_yaml))

    eda_out = run_eda(str(cfg_path))
    assert os.path.exists(eda_out["stats_csv"])

    ana_out = run_analysis(str(cfg_path))
    for k in ["metrics_json", "meta_json", "lagged_csv", "pred_plot", "resid_plot"]:
        assert os.path.exists(ana_out[k])
    with open(ana_out["metrics_json"]) as f:
        metrics = json.load(f)
    assert "Baseline" in metrics and "Ensemble_mean" in metrics
