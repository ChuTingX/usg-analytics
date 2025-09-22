from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    input_csv: str
    date_column: str = "Date"
    parse_dates: bool = True
    index_as_date: bool = True
    trend_suffix: str = "_Trend"


@dataclass
class FeatureConfig:
    target: str = "Adj Close"
    variables: List[str] = field(default_factory=list)
    rolling_means: List[int] = field(default_factory=lambda: [7, 30])
    lag_all_predictors: bool = True


@dataclass
class CVConfig:
    strategy: str = "kfold"  # 'kfold' or 'timeseries'
    n_splits: int = 10
    shuffle: bool = True
    random_state: Optional[int] = 42


@dataclass
class OutputConfig:
    processed_dir: str = "data/processed"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"


@dataclass
class MasterConfig:
    data: DataConfig
    features: FeatureConfig
    cv: CVConfig
    models: Dict[str, Any]
    output: OutputConfig


def load_config(path: str) -> MasterConfig:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    data = DataConfig(**y["data"])
    features = FeatureConfig(**y["features"])
    cv = CVConfig(**y["cv"])
    output = OutputConfig(**y["output"])
    models = y.get("models", {})
    if cv.strategy not in {"kfold", "timeseries"}:
        raise ValueError("cv.strategy must be 'kfold' or 'timeseries'")
    if not features.target:
        raise ValueError("features.target must be set")
    if not isinstance(features.variables, list) or len(features.variables) == 0:
        raise ValueError("features.variables must be a non-empty list")
    return MasterConfig(data=data, features=features, cv=cv, models=models, output=output)
