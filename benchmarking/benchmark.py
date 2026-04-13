from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarking.config import BenchmarkConfig
from benchmarking.data import MatchedPanels, build_matched_panels, train_standardize
from benchmarking.model import InformerForecaster


@dataclass(slots=True)
class BenchmarkResult:
    per_target: pd.DataFrame
    summary: pd.DataFrame
    comparison: pd.DataFrame
    output_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_windows(
    values: np.ndarray,
    context_length: int,
    prediction_length: int,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for end in range(context_length, train_end - prediction_length + 1):
        xs.append(values[end - context_length : end])
        ys.append(values[end : end + prediction_length])
    if not xs:
        raise ValueError("Not enough data to create training windows")
    return np.stack(xs), np.stack(ys)


def _select_origins(length: int, config: BenchmarkConfig) -> list[int]:
    first_origin = max(config.min_train_size, config.context_length + config.prediction_length + 12)
    last_origin = length - config.prediction_length
    if last_origin <= first_origin:
        raise ValueError("Series too short for requested rolling evaluation setup")
    origins = np.linspace(first_origin, last_origin, num=config.rolling_splits, dtype=int)
    return sorted(set(int(origin) for origin in origins))


def _build_model(config: BenchmarkConfig, num_features: int) -> InformerForecaster:
    return InformerForecaster(
        num_features=num_features,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        factor=config.attention_factor,
    )


def _fit_single_origin(
    train_values: np.ndarray,
    context: np.ndarray,
    config: BenchmarkConfig,
) -> np.ndarray:
    scaled_train, scaled_context, _, _ = train_standardize(train_values, context)
    features = train_values.shape[1]
    model = _build_model(config, features).to(config.device)
    inputs, targets = _make_windows(
        scaled_train,
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        train_end=scaled_train.shape[0],
    )
    split = max(1, int(len(inputs) * 0.85))
    train_ds = TensorDataset(
        torch.tensor(inputs[:split], dtype=torch.float32),
        torch.tensor(targets[:split], dtype=torch.float32),
    )
    valid_ds = TensorDataset(
        torch.tensor(inputs[split:], dtype=torch.float32),
        torch.tensor(targets[split:], dtype=torch.float32),
    )
    if len(valid_ds) == 0:
        valid_ds = train_ds
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _ in range(config.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(config.device)
                yb = yb.to(config.device)
                valid_losses.append(criterion(model(xb), yb).item())
        mean_loss = float(np.mean(valid_losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        forecast = model(
            torch.tensor(scaled_context[None, ...], dtype=torch.float32, device=config.device)
        ).cpu().numpy()[0]
    return forecast


def _evaluate_dataset(
    frame: pd.DataFrame,
    dataset_name: str,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    origins = _select_origins(len(frame), config)
    records: list[dict[str, object]] = []

    for origin in origins:
        train_values = values[:origin]
        context = train_values[-config.context_length :]
        forecast_scaled = _fit_single_origin(train_values, context, config)
        _, actual_scaled, mean, std = train_standardize(
            train_values,
            values[origin : origin + config.prediction_length],
        )
        forecast = forecast_scaled * std + mean
        actual = actual_scaled * std + mean
        forecast_index = frame.index[origin : origin + config.prediction_length]

        for horizon in config.eval_horizons:
            step = horizon - 1
            for col_idx, variable in enumerate(frame.columns):
                pred = float(forecast[step, col_idx])
                obs = float(actual[step, col_idx])
                err = pred - obs
                records.append(
                    {
                        "dataset": dataset_name,
                        "origin": str(frame.index[origin - 1]),
                        "target_date": str(forecast_index[step]),
                        "horizon": horizon,
                        "target": variable,
                        "prediction": pred,
                        "actual": obs,
                        "absolute_error": abs(err),
                        "squared_error": err * err,
                    }
                )

    return pd.DataFrame.from_records(records)


def _summarize_results(per_prediction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_target = (
        per_prediction.groupby(["dataset", "horizon", "target"], as_index=False)
        .agg(
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            num_forecasts=("absolute_error", "size"),
        )
        .sort_values(["dataset", "horizon", "target"])
    )
    summary = (
        per_target.groupby(["dataset", "horizon"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            targets=("target", "size"),
        )
        .sort_values(["dataset", "horizon"])
    )
    return per_target, summary


def _build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    comparison = summary.pivot(index="horizon", columns="dataset", values=["mae_mean", "rmse_mean"])
    comparison.columns = [f"{metric}_{dataset}" for metric, dataset in comparison.columns]
    comparison = comparison.reset_index()
    comparison["mae_mean_delta_collected_minus_fred_md"] = (
        comparison["mae_mean_collected"] - comparison["mae_mean_fred_md"]
    )
    comparison["rmse_mean_delta_collected_minus_fred_md"] = (
        comparison["rmse_mean_collected"] - comparison["rmse_mean_fred_md"]
    )
    return comparison


def _write_outputs(
    config: BenchmarkConfig,
    panels: MatchedPanels,
    per_prediction: pd.DataFrame,
    per_target: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> Path:
    output_dir = config.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_prediction.to_csv(output_dir / "per_prediction_metrics.csv", index=False)
    per_target.to_csv(output_dir / "per_target_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    comparison.to_csv(output_dir / "comparison_metrics.csv", index=False)
    metadata = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "variables": panels.variables,
        "sample_start": str(panels.collected.index.min()),
        "sample_end": str(panels.collected.index.max()),
        "num_observations": int(len(panels.collected)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    def _markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows_"
        header = "| " + " | ".join(frame.columns.astype(str)) + " |"
        divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *rows])

    lines = [
        "# Informer Benchmark Summary",
        "",
        f"- FRED-MD vintage: `{config.fred_md_vintage}`",
        f"- Matched variables: `{', '.join(panels.variables)}`",
        f"- Common sample: `{panels.collected.index.min()}` to `{panels.collected.index.max()}`",
        f"- Rolling splits: `{config.rolling_splits}`",
        "",
        "## Summary Metrics",
        "",
        _markdown_table(summary),
        "",
        "## Collected vs FRED-MD",
        "",
        _markdown_table(comparison),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))
    return output_dir


def run_benchmark(dataset: str, config: BenchmarkConfig) -> BenchmarkResult:
    set_seed(config.seed)
    panels = build_matched_panels(config)
    per_prediction_frames = []

    if dataset in {"collected", "both"}:
        per_prediction_frames.append(_evaluate_dataset(panels.collected, "collected", config))
    if dataset in {"fred_md", "both"}:
        per_prediction_frames.append(_evaluate_dataset(panels.fred_md, "fred_md", config))
    per_prediction = pd.concat(per_prediction_frames, ignore_index=True)
    per_target, summary = _summarize_results(per_prediction)
    comparison = (
        _build_comparison(summary)
        if dataset == "both"
        else pd.DataFrame(columns=["horizon"])
    )
    output_dir = _write_outputs(
        config=config,
        panels=panels,
        per_prediction=per_prediction,
        per_target=per_target,
        summary=summary,
        comparison=comparison,
    )
    return BenchmarkResult(
        per_target=per_target,
        summary=summary,
        comparison=comparison,
        output_dir=output_dir,
    )
