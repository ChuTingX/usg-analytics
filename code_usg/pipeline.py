# code_usg/pipeline.py
import os
import json
from typing import Dict, Any

from .config import load_config
from .io_utils import load_dataframe
from .features import drop_trend_columns, select_variables, add_target_rollings, make_lagged
from .modeling import run_modeling_suite
from .plots import save_corr_heatmap, save_timeseries, save_multi_timeseries
from .evaluation import save_pred_vs_actual, save_residuals, run_metadata_dict
from .seed import seed_everything


def run_eda(cfg_path: str) -> Dict[str, str]:
    """Basic EDA: summary tables + correlation heatmap + time-series figures."""
    cfg = load_config(cfg_path)
    seed_everything(cfg.cv.random_state)

    df = load_dataframe(
        cfg.data.input_csv,
        cfg.data.date_column,
        cfg.data.parse_dates,
        cfg.data.index_as_date,
    )
    df = drop_trend_columns(df, cfg.data.trend_suffix)
    df_sel = select_variables(df, cfg.features.variables)

    os.makedirs(cfg.output.reports_dir, exist_ok=True)
    os.makedirs(cfg.output.figures_dir, exist_ok=True)

    stats_path = os.path.join(cfg.output.reports_dir, "descriptive_stats.csv")
    med_path = os.path.join(cfg.output.reports_dir, "median_values.csv")
    df_sel.describe().T.to_csv(stats_path)
    df_sel.median(numeric_only=True).to_csv(med_path, header=["median"])

    save_corr_heatmap(df_sel, os.path.join(cfg.output.figures_dir, "corr_heatmap.png"))

    if cfg.features.target in df_sel.columns:
        save_timeseries(
            df_sel,
            cfg.features.target,
            os.path.join(cfg.output.figures_dir, "target_ts.png"),
            f"{cfg.features.target} Over Time",
        )

    other_cols = [c for c in cfg.features.variables if c != cfg.features.target and c in df_sel.columns]
    if other_cols:
        save_multi_timeseries(
            df_sel,
            other_cols,
            os.path.join(cfg.output.figures_dir, "predictors_ts.png"),
            "Predictors Over Time",
        )

    return {"stats_csv": stats_path, "medians_csv": med_path}


def run_analysis(cfg_path: str) -> Dict[str, str]:
    """
    Full pipeline:
      - load -> clean -> select -> rolling target -> lag features
      - run modeling suite (baseline, RF, OLS-subset, Lasso, Ridge, Ensemble)
      - save metrics, metadata, diagnostic plots, and processed dataset
    """
    cfg = load_config(cfg_path)
    seed_everything(cfg.cv.random_state)

    # Load + clean + select
    df = load_dataframe(
        cfg.data.input_csv,
        cfg.data.date_column,
        cfg.data.parse_dates,
        cfg.data.index_as_date,
    )
    df = drop_trend_columns(df, cfg.data.trend_suffix)
    df_sel = select_variables(df, cfg.features.variables)

    if cfg.features.target not in df_sel.columns:
        raise ValueError(f"Target '{cfg.features.target}' missing from selected columns.")

    # Rolling means on target then drop leading NaNs
    df_sel = add_target_rollings(df_sel, cfg.features.target, cfg.features.rolling_means).dropna()

    # Build supervised, 1-day lagged dataset
    base_predictors = list(df_sel.columns)
    lagged = make_lagged(df_sel, base_predictors, cfg.features.target, lag=1)

    # Arrange inputs
    y = lagged[cfg.features.target].values
    feature_names = [c for c in lagged.columns if c.endswith("_prev")]
    X = lagged[feature_names].values

    # Ensure output folders
    os.makedirs(cfg.output.reports_dir, exist_ok=True)
    os.makedirs(cfg.output.figures_dir, exist_ok=True)
    os.makedirs(cfg.output.processed_dir, exist_ok=True)

    # Run all models (RF will automatically use tuned params if reports/tuned_rf_params.json exists)
    results = run_modeling_suite(X, y, feature_names, cfg, reports_dir=cfg.output.reports_dir)
    metrics = results["metrics"]
    fitted = results["fitted"]

    # Persist metrics + run metadata
    metrics_path = os.path.join(cfg.output.reports_dir, "metrics.json")
    meta_path = os.path.join(cfg.output.reports_dir, "run_metadata.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata_dict(), f, indent=2)

    # Fit RF on full data for diagnostic plots (cleaner than last-fold model)
    rf_full = fitted["rf"]
    rf_full.fit(X, y)
    y_pred = rf_full.predict(X)

    pred_plot = os.path.join(cfg.output.figures_dir, "pred_vs_actual.png")
    resid_plot = os.path.join(cfg.output.figures_dir, "residuals.png")
    save_pred_vs_actual(y, y_pred, pred_plot)
    save_residuals(y, y_pred, resid_plot)

    # Save processed supervised table
    lagged_path = os.path.join(cfg.output.processed_dir, "lagged_dataset.csv")
    lagged.to_csv(lagged_path)

    return {
        "metrics_json": metrics_path,
        "meta_json": meta_path,
        "lagged_csv": lagged_path,
        "pred_plot": pred_plot,
        "resid_plot": resid_plot,
    }