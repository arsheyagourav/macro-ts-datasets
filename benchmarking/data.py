from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from benchmarking.config import BenchmarkConfig


@dataclass(slots=True)
class MatchedPanels:
    collected: pd.DataFrame
    fred_md: pd.DataFrame
    variables: list[str]


def _ensure_monthly_period_index(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = frame.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    df.index = pd.PeriodIndex(df[date_col], freq="M")
    df = df.drop(columns=[date_col])
    return df


def _trim_full_panel(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.notna().all(axis=1)
    first = valid.idxmax()
    last = valid[::-1].idxmax()
    return frame.loc[first:last].copy()


def _reindex_contiguous(frame: pd.DataFrame) -> pd.DataFrame:
    idx = pd.period_range(frame.index.min(), frame.index.max(), freq="M")
    return frame.reindex(idx)


def _impute_panel(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.astype(float)
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )


def load_collected_monthly(config: BenchmarkConfig) -> pd.DataFrame:
    labor = pd.read_csv(config.processed_root / "monthly_labor_market.csv")
    ip = pd.read_csv(
        config.processed_root / "monthly_industrial_production_capacity_utilization.csv"
    )
    merged = labor.merge(ip, on="date", how="outer")
    variables = list(config.matched_variables.keys())
    merged = merged[["date", *variables]].copy()
    merged["date"] = pd.PeriodIndex(merged["date"].str.replace(":", "-", regex=False), freq="M")
    panel = _ensure_monthly_period_index(merged, "date")
    panel = _trim_full_panel(panel)
    panel = _reindex_contiguous(panel)
    return _impute_panel(panel)


def ensure_fred_md_file(config: BenchmarkConfig) -> Path:
    path = config.fred_md_path
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(config.fred_md_url, path)
    return path


def load_fred_md_monthly(config: BenchmarkConfig) -> pd.DataFrame:
    path = ensure_fred_md_file(config)
    raw = pd.read_csv(path)
    raw = raw[raw["sasdate"] != "Transform:"].copy()
    fred_columns = list(config.matched_variables.values())
    rename_map = {"sasdate": "date", **{v: k for k, v in config.matched_variables.items()}}
    df = raw[["sasdate", *fred_columns]].rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M")
    panel = _ensure_monthly_period_index(df, "date")
    panel = _trim_full_panel(panel)
    panel = _reindex_contiguous(panel)
    return _impute_panel(panel)


def build_matched_panels(config: BenchmarkConfig) -> MatchedPanels:
    collected = load_collected_monthly(config)
    fred_md = load_fred_md_monthly(config)
    common_vars = sorted(set(collected.columns).intersection(fred_md.columns))
    common_start = max(collected.index.min(), fred_md.index.min())
    common_end = min(collected.index.max(), fred_md.index.max())
    common_index = pd.period_range(common_start, common_end, freq="M")
    collected = collected.loc[common_index, common_vars].copy()
    fred_md = fred_md.loc[common_index, common_vars].copy()
    collected = _impute_panel(collected)
    fred_md = _impute_panel(fred_md)
    if config.max_observations is not None:
        collected = collected.iloc[-config.max_observations :].copy()
        fred_md = fred_md.iloc[-config.max_observations :].copy()
    return MatchedPanels(collected=collected, fred_md=fred_md, variables=common_vars)


def train_standardize(
    train_values: np.ndarray,
    eval_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_scaled = (train_values - mean) / std
    eval_scaled = (eval_values - mean) / std
    return train_scaled, eval_scaled, mean, std
