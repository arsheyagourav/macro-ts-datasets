# Informer Benchmark Summary

- FRED-MD vintage: `2026-03-md.csv`
- Matched variables: `CUM, EMPLOY, H, IPM, IPT, LFC, RUC`
- Common sample: `1964-01` to `2026-01`
- Rolling splits: `4`

## Summary Metrics

| dataset | horizon | mae_mean | rmse_mean | targets |
| --- | --- | --- | --- | --- |
| collected | 1 | 3329.8153643742617 | 3628.405281555368 | 7 |
| collected | 3 | 3346.2115915709605 | 3676.200027833682 | 7 |
| collected | 6 | 3313.0460677706283 | 3779.3662411280866 | 7 |
| fred_md | 1 | 3810.795903521305 | 4243.062556195588 | 7 |
| fred_md | 3 | 3421.2708624782563 | 4001.1128601397113 | 7 |
| fred_md | 6 | 3152.88610764865 | 3861.262236964769 | 7 |

## Collected vs FRED-MD

| horizon | mae_mean_collected | mae_mean_fred_md | rmse_mean_collected | rmse_mean_fred_md | mae_mean_delta_collected_minus_fred_md | rmse_mean_delta_collected_minus_fred_md |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3329.8153643742617 | 3810.795903521305 | 3628.405281555368 | 4243.062556195588 | -480.98053914704315 | -614.6572746402194 |
| 3 | 3346.2115915709605 | 3421.2708624782563 | 3676.200027833682 | 4001.1128601397113 | -75.05927090729574 | -324.9128323060295 |
| 6 | 3313.0460677706283 | 3152.88610764865 | 3779.3662411280866 | 3861.262236964769 | 160.1599601219782 | -81.89599583668223 |
