#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, adfuller


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "monthly_labor_market.csv"
OUT_DIR = REPO_ROOT / "results" / "report_assets"
FIG_DIR = OUT_DIR / "figures"

TARGET = "EMPLOY"
PANEL_COLUMNS = ["POP", "LFC", "LFPART", "RUC", "EMPLOY", "H", "HG", "HS"]
EVENTS = {
    "2001-09-01": "9/11 / Dot-com bust",
    "2008-09-01": "Financial crisis",
    "2020-03-01": "COVID-19 shock",
}

# These forecasts were already produced in the saved notebooks. Keeping them here
# lets the repo export one consistent report bundle without reinstalling the
# heavyweight model dependencies.
NOTEBOOK_MODELS = {
    "Chronos": {
        "family": "foundation",
        "source": "notebooks/model_experiments.ipynb",
        "prediction": [
            136522.14,
            137325.22,
            137325.22,
            137325.22,
            137325.22,
            137325.22,
            137325.22,
            137325.22,
            138128.28,
            138128.28,
            138128.28,
            138128.28,
        ],
    },
    "N-BEATS": {
        "family": "task_specific_neural",
        "source": "notebooks/model_experiments.ipynb",
        "prediction": [
            136952.58,
            137217.12,
            137367.47,
            137551.61,
            137827.90,
            137965.25,
            138167.86,
            138400.62,
            138609.38,
            138740.70,
            139000.72,
            139166.52,
        ],
    },
    "Moirai": {
        "family": "foundation",
        "source": "draft report / notebooks/moirai.ipynb",
        "prediction": None,
        "mae": 459.09,
        "mse": 264565.34,
        "note": "Notebook source exists, but executed forecast outputs were not saved in the repo.",
    },
    "TimesFM": {
        "family": "foundation",
        "source": "notebooks/timesfm_experiment.ipynb",
        "prediction": [
            136287.66,
            136369.25,
            136937.88,
            137119.84,
            137485.60,
            137739.48,
            138136.56,
            138135.72,
            138516.33,
            138932.48,
            139143.95,
            139220.19,
        ],
    },
    "TTM": {
        "family": "foundation",
        "source": "notebooks/ttm_experiment.ipynb",
        "prediction": [
            138098.31,
            137928.53,
            137846.98,
            137681.95,
            137412.84,
            137313.25,
            137245.14,
            137126.38,
            136943.95,
            136669.31,
            136769.17,
            136790.77,
        ],
    },
    "DeepAR": {
        "family": "task_specific_neural",
        "source": "notebooks/deepar_experiment.ipynb",
        "prediction": [
            136774.17,
            136865.72,
            136895.00,
            136850.19,
            136728.89,
            136530.48,
            136254.94,
            135902.05,
            135471.03,
            134962.48,
            134384.81,
            133760.90,
        ],
    },
}


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"date": "ds"})
    df["ds"] = pd.to_datetime(df["ds"], format="%Y:%m", errors="coerce")
    df = df.dropna(how="all", subset=df.columns[1:]).ffill().dropna()
    df = df.set_index("ds").asfreq("MS")
    return df


def compute_dataset_summary(df: pd.DataFrame, split: int) -> dict[str, object]:
    emp = df[TARGET].astype(float)
    train = df.iloc[:split]
    test = df.iloc[split : split + 12]
    monthly_change = emp.diff().dropna()
    yearly_change = emp.diff(12).dropna()

    return {
        "rows": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "target": TARGET,
        "variables": PANEL_COLUMNS,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": str(train.index.min().date()),
        "train_end": str(train.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "target_min": float(emp.min()),
        "target_max": float(emp.max()),
        "target_mean": float(emp.mean()),
        "target_std": float(emp.std()),
        "monthly_change_mean": float(monthly_change.mean()),
        "monthly_change_std": float(monthly_change.std()),
        "yearly_change_mean": float(yearly_change.mean()),
        "yearly_change_std": float(yearly_change.std()),
        "acf": {
            "lag_1": float(acf(emp, nlags=1, fft=True)[1]),
            "lag_6": float(acf(emp, nlags=6, fft=True)[6]),
            "lag_12": float(acf(emp, nlags=12, fft=True)[12]),
            "lag_24": float(acf(emp, nlags=24, fft=True)[24]),
        },
        "adf": {
            "level_stat": float(adfuller(emp, autolag="AIC")[0]),
            "level_pvalue": float(adfuller(emp, autolag="AIC")[1]),
            "diff1_stat": float(adfuller(emp.diff().dropna(), autolag="AIC")[0]),
            "diff1_pvalue": float(adfuller(emp.diff().dropna(), autolag="AIC")[1]),
            "diff12_stat": float(adfuller(emp.diff(12).dropna(), autolag="AIC")[0]),
            "diff12_pvalue": float(adfuller(emp.diff(12).dropna(), autolag="AIC")[1]),
        },
        "largest_negative_monthly_changes": [
            {
                "date": str(idx.date()),
                "change": float(value),
            }
            for idx, value in emp.diff().nsmallest(5).items()
        ],
    }


def fit_added_models(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    emp_train = train[TARGET].astype(float)

    ets = ExponentialSmoothing(
        emp_train,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)

    arima = ARIMA(emp_train, order=(2, 2, 2)).fit()

    var_fit = VAR(train[PANEL_COLUMNS].astype(float)).fit(3)
    var_forecast = var_fit.forecast(train[PANEL_COLUMNS].astype(float).values[-3:], steps=len(test))

    return {
        "Holt-Winters": np.asarray(ets.forecast(len(test))),
        "ARIMA(2,2,2)": np.asarray(arima.forecast(len(test))),
        "VAR(3)": np.asarray(var_forecast[:, PANEL_COLUMNS.index(TARGET)]),
    }


def build_metrics_rows(
    test: pd.DataFrame,
    added_predictions: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    actual = test[TARGET].astype(float).to_numpy()
    predictions_for_csv: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for model_name, meta in NOTEBOOK_MODELS.items():
        pred = meta["prediction"]
        if pred is None:
            rows.append(
                {
                    "model": model_name,
                    "group": "existing_repo",
                    "family": meta["family"],
                    "source": meta["source"],
                    "mae": meta["mae"],
                    "mse": meta["mse"],
                    "has_forecast_path": False,
                    "note": meta.get("note", ""),
                }
            )
            continue

        pred_array = np.asarray(pred, dtype=float)
        predictions_for_csv[model_name] = pred_array
        rows.append(
            {
                "model": model_name,
                "group": "existing_repo",
                "family": meta["family"],
                "source": meta["source"],
                "mae": float(mean_absolute_error(actual, pred_array)),
                "mse": float(mean_squared_error(actual, pred_array)),
                "has_forecast_path": True,
                "note": "",
            }
        )

    for model_name, pred_array in added_predictions.items():
        predictions_for_csv[model_name] = pred_array
        rows.append(
            {
                "model": model_name,
                "group": "added_models",
                "family": "classical_time_series",
                "source": "scripts/build_report_assets.py",
                "mae": float(mean_absolute_error(actual, pred_array)),
                "mse": float(mean_squared_error(actual, pred_array)),
                "has_forecast_path": True,
                "note": "",
            }
        )

    return rows, predictions_for_csv


def make_predictions_frame(test: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame({"date": test.index, "Actual": test[TARGET].astype(float).to_numpy()})
    for model_name, pred_array in predictions.items():
        frame[model_name] = pred_array
    return frame


def plot_historical_context(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    y_max = df[TARGET].max()
    y_min = df[TARGET].min()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df[TARGET], color="steelblue", linewidth=1.2, label=TARGET)
    ax.axvspan(train.index[0], train.index[-1], alpha=0.08, color="green", label="Training period")
    ax.axvspan(test.index[0], test.index[-1], alpha=0.30, color="orange", label="Test window")

    for date, label in EVENTS.items():
        ts = pd.Timestamp(date)
        ax.axvline(ts, color="crimson", linestyle="--", linewidth=1, alpha=0.75)
        ax.text(
            ts,
            y_min + (y_max - y_min) * 0.05,
            label,
            rotation=90,
            fontsize=8,
            color="crimson",
            va="bottom",
        )

    ax.set_title("Monthly Employment with Train/Test Split and Major Shocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Employment (thousands)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "historical_context.png", dpi=200)
    plt.close(fig)


def plot_mae_ranking(metrics_df: pd.DataFrame) -> None:
    ordered = metrics_df.sort_values("mae", ascending=True)
    colors = {
        "added_models": "#2a9d8f",
        "existing_repo": "#577590",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        ordered["model"],
        ordered["mae"],
        color=[colors[group] for group in ordered["group"]],
    )
    ax.invert_yaxis()
    ax.set_title("Held-Out Test MAE by Model")
    ax.set_xlabel("MAE (thousands of workers)")
    ax.grid(axis="x", alpha=0.25)

    for idx, (_, row) in enumerate(ordered.iterrows()):
        ax.text(row["mae"] + 20, idx, f"{row['mae']:.2f}", va="center", fontsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["added_models"], label="Added models"),
        plt.Rectangle((0, 0), 1, 1, color=colors["existing_repo"], label="Existing repo models"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "all_model_mae.png", dpi=200)
    plt.close(fig)


def plot_forecasts(
    test: pd.DataFrame,
    predictions_df: pd.DataFrame,
    columns: list[str],
    title: str,
    filename: str,
) -> None:
    palette = [
        "#000000",
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, predictions_df["Actual"], color=palette[0], marker="o", linewidth=2, label="Actual")

    for color, model_name in zip(palette[1:], columns, strict=False):
        linestyle = "--" if model_name in {"Chronos", "ARIMA(2,2,2)"} else "-"
        ax.plot(
            test.index,
            predictions_df[model_name],
            color=color,
            linewidth=1.6,
            linestyle=linestyle,
            label=model_name,
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Employment (thousands)")
    ax.grid(alpha=0.3)
    ax.legend(ncols=2, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)


def write_report_readme(summary: dict[str, object], metrics_df: pd.DataFrame) -> None:
    top_three = metrics_df.sort_values("mae").head(3)
    lines = [
        "# Report Assets",
        "",
        "This folder consolidates saved notebook outputs from the repo and three added classical models:",
        "",
        "- `Holt-Winters`",
        "- `ARIMA(2,2,2)`",
        "- `VAR(3)`",
        "",
        "The benchmark target is `EMPLOY` from `data/processed/monthly_labor_market.csv`.",
        "",
        f"- Full sample: {summary['start']} to {summary['end']} ({summary['rows']} monthly observations)",
        f"- Train split: {summary['train_start']} to {summary['train_end']} ({summary['train_rows']} rows)",
        f"- Test split: {summary['test_start']} to {summary['test_end']} ({summary['test_rows']} rows)",
        "",
        "## Best MAE Models",
        "",
    ]

    for _, row in top_three.iterrows():
        lines.append(f"- `{row['model']}`: MAE `{row['mae']:.2f}`, MSE `{row['mse']:.2f}`")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `model_metrics.csv`: consolidated MAE/MSE leaderboard",
            "- `model_predictions.csv`: actual values and available test-window forecasts",
            "- `dataset_summary.json`: train/test split and basic time-series diagnostics",
            "- `figures/`: exported PNGs for the report",
        ]
    )

    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split : split + 12].copy()

    summary = compute_dataset_summary(df, split)
    added_predictions = fit_added_models(train, test)
    metric_rows, predictions = build_metrics_rows(test, added_predictions)

    metrics_df = pd.DataFrame(metric_rows).sort_values("mae", ascending=True)
    predictions_df = make_predictions_frame(test, predictions)

    metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "model_predictions.csv", index=False)
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_historical_context(df, train, test)
    plot_mae_ranking(metrics_df)
    plot_forecasts(
        test,
        predictions_df,
        ["Holt-Winters", "VAR(3)", "ARIMA(2,2,2)"],
        "Added Models vs Actual Employment",
        "added_model_comparison.png",
    )
    plot_forecasts(
        test,
        predictions_df,
        ["Holt-Winters", "VAR(3)", "ARIMA(2,2,2)", "N-BEATS"],
        "Top Forecasting Models on the 2013-2014 Test Window",
        "top_model_comparison.png",
    )
    plot_forecasts(
        test,
        predictions_df,
        ["Chronos", "N-BEATS", "TimesFM", "TTM", "DeepAR"],
        "Existing Notebook Models vs Actual Employment",
        "existing_model_comparison.png",
    )

    write_report_readme(summary, metrics_df)

    print(f"Wrote report assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
