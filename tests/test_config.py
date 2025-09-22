import yaml

from code_usg.config import load_config


def test_config_load_and_validate(tmp_path):
    cfg_yaml = {
        "data": {"input_csv": "data/raw/FINAL_USO.csv", "date_column": "Date"},
        "features": {"target": "Adj Close", "variables": ["Adj Close", "SP_close"]},
        "cv": {"strategy": "kfold", "n_splits": 3, "shuffle": True, "random_state": 123},
        "models": {},
        "output": {
            "processed_dir": "data/processed",
            "reports_dir": "reports",
            "figures_dir": "reports/figures",
        },
    }
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg_yaml))
    cfg = load_config(str(p))
    assert cfg.cv.strategy == "kfold"
    assert cfg.features.target == "Adj Close"
    assert "SP_close" in cfg.features.variables
