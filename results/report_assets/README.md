# Report Assets

This folder consolidates saved notebook outputs from the repo and three added classical models:

- `Holt-Winters`
- `ARIMA(2,2,2)`
- `VAR(3)`

The benchmark target is `EMPLOY` from `data/processed/monthly_labor_market.csv`.

- Full sample: 1964-01-01 to 2026-02-01 (746 monthly observations)
- Train split: 1964-01-01 to 2013-08-01 (596 rows)
- Test split: 2013-09-01 to 2014-08-01 (12 rows)

## Best MAE Models

- `Holt-Winters`: MAE `92.71`, MSE `12335.93`
- `VAR(3)`: MAE `96.83`, MSE `13285.07`
- `ARIMA(2,2,2)`: MAE `103.21`, MSE `16708.52`

## Files

- `model_metrics.csv`: consolidated MAE/MSE leaderboard
- `model_predictions.csv`: actual values and available test-window forecasts
- `dataset_summary.json`: train/test split and basic time-series diagnostics
- `figures/`: exported PNGs for the report
