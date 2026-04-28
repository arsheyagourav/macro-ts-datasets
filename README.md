# macro-ts-datasets

Macro time series datasets organized by category and frequency for forecasting and analysis.

## Processed datasets

- `data/processed/monthly_labor_market.csv`
- `data/processed/monthly_industrial_production_capacity_utilization.csv`
- `data/processed/quarterly_income.csv`
- `data/processed/quarterly_spending.csv`

## Report assets

To rebuild the consolidated EMPLOY report bundle, run:

```bash
MPLCONFIGDIR=/tmp/mpl /opt/miniconda3/bin/python scripts/build_report_assets.py
```

This exports:

- `results/report_assets/model_metrics.csv`
- `results/report_assets/model_predictions.csv`
- `results/report_assets/dataset_summary.json`
- `results/report_assets/figures/*.png`

## Informer benchmark

This repo now includes an Informer-style rolling forecast benchmark that compares:

- the repo's collected monthly data
- the official `FRED-MD` monthly database

The benchmark is pinned to the St. Louis Fed monthly vintage `2026-03-md.csv` and uses the overlap variables that can be matched cleanly across both datasets:

- `LFC` -> `CLF16OV`
- `EMPLOY` -> `CE16OV`
- `RUC` -> `UNRATE`
- `H` -> `AWHMAN`
- `IPT` -> `INDPRO`
- `IPM` -> `IPMANSICS`
- `CUM` -> `CUMFNS`

### Run it

From the repo root:

```bash
python scripts/run_informer_benchmark.py --dataset both
```

Useful options:

```bash
python scripts/run_informer_benchmark.py \
  --dataset both \
  --fred-md-vintage 2026-03-md.csv \
  --context-length 36 \
  --prediction-length 6 \
  --horizons 1 3 6 \
  --rolling-splits 6 \
  --seed 42
```

The script will download the pinned `FRED-MD` vintage into `data/raw/fred_md/` if it is missing.

### Outputs

By default, results are written under:

```text
results/informer_benchmark/<fred_md_vintage_stem>/
```

Each run writes:

- `per_prediction_metrics.csv`
- `per_target_metrics.csv`
- `summary_metrics.csv`
- `comparison_metrics.csv`
- `run_metadata.json`
- `README.md`

### Verified artifacts

The implementation was validated with:

- unit tests in `tests/test_informer_benchmark.py`
- a smoke run in `results/informer_benchmark/smoke/`
- a fuller benchmark run in `results/informer_benchmark/default_run/`
