# code_usg/tuning.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer


def _load_raw_csv(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = cfg["data"]["input_csv"]
    date_col = cfg["data"].get("date_column", "Date")
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
    # drop trend columns if requested
    trend_suffix = cfg["data"].get("trend_suffix", "_Trend")
    trend_cols = [c for c in df.columns if c.endswith(trend_suffix)]
    df = df.drop(columns=trend_cols, errors="ignore")
    return df


def _build_supervised(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, list]:
    """Mirror the pipeline: rolling means + 1-day lags for all predictors."""
    target = cfg["features"]["target"]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe columns.")

    # rolling means on the target
    for w in cfg["features"].get("rolling_means", []):
        df[f"Adj_Close_{w}d"] = df[target].rolling(window=w, min_periods=w).mean()

    # predictor list
    variables = cfg["features"]["variables"][:]
    # include generated rolling columns if they exist
    for w in cfg["features"].get("rolling_means", []):
        cname = f"Adj_Close_{w}d"
        if cname not in variables and cname in df.columns:
            variables.append(cname)

    # build lagged dataset
    lagged = pd.DataFrame(index=df.index)
    lagged[target] = df[target]
    for v in variables:
        if v in df.columns:
            lagged[f"{v}_prev"] = df[v].shift(1)

    lagged = lagged.dropna(axis=0)

    y = lagged[target].values
    X_cols = [c for c in lagged.columns if c.endswith("_prev")]
    X = lagged[X_cols].values
    return X, y, X_cols


def _make_cv(cfg: Dict[str, Any]):
    cv_cfg = cfg.get("cv", {})
    strategy = cv_cfg.get("strategy", "kfold").lower()
    n_splits = int(cv_cfg.get("n_splits", 10))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", 42))

    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    # default: KFold
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def tune_random_forest(
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run GridSearchCV for RF using config; write best params + cv results."""
    df = _load_raw_csv(cfg)
    X, y, _ = _build_supervised(df, cfg)
    cv = _make_cv(cfg)

    grid = cfg["models"]["random_forest"]["tuning"]["grid"]
    scoring = cfg["models"]["random_forest"]["tuning"].get("scoring", "neg_mean_squared_error")
    n_jobs = cfg["models"]["random_forest"]["tuning"].get("n_jobs", -1)
    verbose = cfg["models"]["random_forest"]["tuning"].get("verbose", 1)

    # Define scorer explicitly to be safe
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    scorer = make_scorer(mse, greater_is_better=False)

    base_rf = RandomForestRegressor(random_state=int(cfg["models"]["random_forest"]["params"].get("random_state", 42)))

    gs = GridSearchCV(
        estimator=base_rf,
        param_grid=grid,
        scoring=scorer if scoring == "neg_mean_squared_error" else scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    gs.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save best params
    best_params_path = out_dir / "tuned_rf_params.json"
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(gs.best_params_, f, indent=2)

    # Save full CV results for auditing
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(out_dir / "rf_cv_results.csv", index=False)

    return gs.best_params_


def load_tuned_rf_params(reports_dir: Path) -> Dict[str, Any] | None:
    p = reports_dir / "tuned_rf_params.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None