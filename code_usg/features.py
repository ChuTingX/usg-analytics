from __future__ import annotations

import pandas as pd


def drop_trend_columns(df: pd.DataFrame, suffix: str = "_Trend") -> pd.DataFrame:
    trend_columns = [c for c in df.columns if c.endswith(suffix)]
    return df.drop(columns=trend_columns, errors="ignore")


def select_variables(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    keep = [v for v in variables if v in df.columns]
    return df[keep].copy()


def add_target_rollings(df: pd.DataFrame, target: str, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"{target.replace(' ', '_')}_{w}d"] = df[target].rolling(window=w, min_periods=w).mean()
    return df


def make_lagged(
    df: pd.DataFrame, base_predictors: list[str], target: str, lag: int = 1
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[target] = df[target]
    for var in base_predictors:
        out[f"{var}_prev"] = df[var].shift(lag)
    out = out.dropna()
    return out
