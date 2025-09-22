# code_usg/modeling.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --- helpers to read dict or dataclass-like config ---
def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _nested(cfg, path: List[str], default=None):
    cur = cfg
    for k in path:
        cur = _field(cur, k, None)
        if cur is None:
            return default
    return cur


def build_cv(
    strategy: str = "kfold", n_splits: int = 10, shuffle: bool = True, random_state: int | None = 42
):
    """Return a CV splitter."""
    if str(strategy).lower() == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def baseline_mse_cv(y: np.ndarray, y_prev: np.ndarray, cv):
    """Baseline: predict today with yesterday's value."""
    mses: List[float] = []
    for _, test_i in cv.split(y_prev):
        y_test = y[test_i]
        y_pred = y_prev[test_i]
        mses.append(mean_squared_error(y_test, y_pred))
    return mses, float(np.mean(mses))


def _load_tuned_rf_params_if_any(reports_dir: str | Path) -> Dict[str, Any] | None:
    """Read tuned RF params if the JSON file exists."""
    p = Path(reports_dir) / "tuned_rf_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _rf_params_from_cfg_or_tuned(cfg, reports_dir: str | Path) -> Dict[str, Any]:
    """Start with config defaults; merge tuned params if present (supports dict or object cfg)."""
    params = (_nested(cfg, ["models", "random_forest", "params"], {}) or {}).copy()
    params.setdefault("random_state", 42)

    tuned = _load_tuned_rf_params_if_any(reports_dir)
    if tuned:
        params.update({k: v for k, v in tuned.items()})
        print(f"[INFO] Using tuned RF params from {Path(reports_dir) / 'tuned_rf_params.json'}")
    else:
        print("[INFO] Using RF params from conf/params.yml (no tuned params found)")
    return params


def random_forest_cv(X: np.ndarray, y: np.ndarray, cv, rf_params: Dict[str, Any]):
    """CV MSE for RF; returns per-fold MSEs, mean, and a fitted estimator (last fold)."""
    rf = RandomForestRegressor(**rf_params)
    mses: List[float] = []
    for train_i, test_i in cv.split(X):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mses.append(mean_squared_error(y_test, y_pred))
    return mses, float(np.mean(mses)), rf


def all_subset_lr_cv(X: np.ndarray, y: np.ndarray, cv, feature_names: List[str]):
    """Exhaustive subset selection for OLS, then CV on the best subset."""
    lr = LinearRegression()

    def neg_mse_scorer(estimator, X_fs, y_fs):
        return -mean_squared_error(y_fs, estimator.predict(X_fs))

    efs = EFS(
        estimator=lr,
        min_features=1,
        max_features=X.shape[1],
        scoring=neg_mse_scorer,
        cv=cv,
        n_jobs=-1,
        print_progress=False,  # or verbose=0 depending on mlxtend version
    ).fit(X, y)

    best_idx = efs.best_idx_
    best_feats = [feature_names[i] for i in best_idx]

    X_best = X[:, best_idx]
    mses: List[float] = []
    for train_i, test_i in cv.split(X_best):
        X_train, X_test = X_best[train_i], X_best[test_i]
        y_train, y_test = y[train_i], y[test_i]
        lr2 = LinearRegression().fit(X_train, y_train)
        y_pred = lr2.predict(X_test)
        mses.append(mean_squared_error(y_test, y_pred))

    return mses, float(np.mean(mses)), best_idx, best_feats


def lasso_ridge_cv(model_type: str, X: np.ndarray, y: np.ndarray, cv, model_conf: Dict[str, Any]):
    """CV for Lasso/Ridge with scaling and alpha grid search."""
    assert model_type in {"lasso", "ridge"}
    al_start, al_end, al_num = _field(model_conf, "alphas_logspace", [-5, 2, 20])
    alphas = np.logspace(al_start, al_end, al_num)

    if model_type == "lasso":
        estimator = Lasso(
            max_iter=_field(model_conf, "max_iter", 10000),
            random_state=_field(model_conf, "random_state", 42),
        )
    else:
        estimator = Ridge(
            max_iter=_field(model_conf, "max_iter", 10000),
            random_state=_field(model_conf, "random_state", 42),
        )

    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    grid = {"model__alpha": alphas}
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    gs = GridSearchCV(pipe, grid, scoring=mse_scorer, cv=cv, n_jobs=-1)
    gs.fit(X, y)

    best_alpha = float(gs.best_params_["model__alpha"])
    best_est = gs.best_estimator_

    mses: List[float] = []
    for train_i, test_i in cv.split(X):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]
        best_est.fit(X_train, y_train)
        y_pred = best_est.predict(X_test)
        mses.append(mean_squared_error(y_test, y_pred))

    return mses, float(np.mean(mses)), best_alpha, best_est


def ensemble_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    rf_est: RandomForestRegressor,
    lasso_est: Pipeline,
    ridge_est: Pipeline,
    best_subset_idx: tuple,
):
    """Simple average of RF + Lasso + Ridge + best-subset OLS."""
    mses: List[float] = []
    for train_i, test_i in cv.split(X):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]

        rf_est.fit(X_train, y_train)
        pred_rf = rf_est.predict(X_test)
        lasso_est.fit(X_train, y_train)
        pred_lasso = lasso_est.predict(X_test)
        ridge_est.fit(X_train, y_train)
        pred_ridge = ridge_est.predict(X_test)

        X_train_lr = X_train[:, best_subset_idx]
        X_test_lr = X_test[:, best_subset_idx]
        lr = LinearRegression().fit(X_train_lr, y_train)
        pred_lr = lr.predict(X_test_lr)

        ensemble_pred = (pred_rf + pred_lasso + pred_ridge + pred_lr) / 4.0
        mses.append(mean_squared_error(y_test, ensemble_pred))

    return mses, float(np.mean(mses)), float(np.std(mses))


def run_modeling_suite(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg,
    reports_dir: str | Path = "reports",
):
    """
    Run baseline, RF, all-subset OLS, Lasso, Ridge, and ensemble.
    Returns metrics and fitted objects.
    """

    # CV config (works with dict or object)
    cv_strategy = _nested(cfg, ["cv", "strategy"], "kfold")
    cv_n_splits = int(_nested(cfg, ["cv", "n_splits"], 10))
    cv_shuffle = bool(_nested(cfg, ["cv", "shuffle"], True))
    cv_random = int(_nested(cfg, ["cv", "random_state"], 42))

    cv = build_cv(cv_strategy, cv_n_splits, cv_shuffle, cv_random)

    # Baseline uses previous day's target; prefer 'Adj Close_prev' if present
    if "Adj Close_prev" in feature_names:
        y_prev = X[:, feature_names.index("Adj Close_prev")]
    else:
        y_prev = X[:, 0]

    base_mses, base_mean = baseline_mse_cv(y, y_prev, cv)

    rf_params = _rf_params_from_cfg_or_tuned(cfg, reports_dir)
    rf_mses, rf_mean, rf_est = random_forest_cv(X, y, cv, rf_params)

    lr_mses, lr_mean, best_idx, best_feats = all_subset_lr_cv(X, y, cv, feature_names)

    lasso_conf = _nested(cfg, ["models", "lasso"], {}) or {}
    ridge_conf = _nested(cfg, ["models", "ridge"], {}) or {}
    lasso_mses, lasso_mean, lasso_alpha, lasso_est = lasso_ridge_cv("lasso", X, y, cv, lasso_conf)
    ridge_mses, ridge_mean, ridge_alpha, ridge_est = lasso_ridge_cv("ridge", X, y, cv, ridge_conf)

    ens_mses, ens_mean, ens_std = ensemble_cv(X, y, cv, rf_est, lasso_est, ridge_est, best_idx)

    metrics = {
        "Baseline": {"cv_mse": base_mses, "mean_mse": base_mean},
        "RandomForest": {"cv_mse": rf_mses, "mean_mse": rf_mean, "params": rf_params},
        "AllSubsetLR": {
            "cv_mse": lr_mses,
            "mean_mse": lr_mean,
            "best_subset_idx": list(best_idx),
            "best_subset_feats": best_feats,
        },
        "Lasso": {"cv_mse": lasso_mses, "mean_mse": lasso_mean, "alpha": lasso_alpha},
        "Ridge": {"cv_mse": ridge_mses, "mean_mse": ridge_mean, "alpha": ridge_alpha},
        "Ensemble": {"cv_mse": ens_mses, "mean_mse": ens_mean, "std_mse": ens_std},
    }

    fitted = {
        "rf": rf_est,
        "lasso": lasso_est,
        "ridge": ridge_est,
        "best_subset_idx": list(best_idx),
        "best_subset_feats": best_feats,
    }

    return {"metrics": metrics, "fitted": fitted}
