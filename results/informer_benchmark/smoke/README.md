# Informer Benchmark Summary

- FRED-MD vintage: `2026-03-md.csv`
- Matched variables: `CUM, EMPLOY, H, IPM, IPT, LFC, RUC`
- Common sample: `2018-02` to `2026-01`
- Rolling splits: `2`

## Summary Metrics

| dataset | horizon | mae_mean | rmse_mean | targets |
| --- | --- | --- | --- | --- |
| collected | 1 | 1699.8655519755455 | 1714.8464666719988 | 7 |
| collected | 3 | 1859.7191465309832 | 1885.0738908260926 | 7 |
| collected | 6 | 1772.2761046242847 | 1795.5536626430883 | 7 |
| fred_md | 1 | 1466.2497276857932 | 1492.9712172164473 | 7 |
| fred_md | 3 | 1634.8603403868167 | 1677.437104212086 | 7 |
| fred_md | 6 | 1669.8003489283105 | 1678.8798808035058 | 7 |

## Collected vs FRED-MD

| horizon | mae_mean_collected | mae_mean_fred_md | rmse_mean_collected | rmse_mean_fred_md | mae_mean_delta_collected_minus_fred_md | rmse_mean_delta_collected_minus_fred_md |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1699.8655519755455 | 1466.2497276857932 | 1714.8464666719988 | 1492.9712172164473 | 233.61582428975225 | 221.8752494555515 |
| 3 | 1859.7191465309832 | 1634.8603403868167 | 1885.0738908260926 | 1677.437104212086 | 224.85880614416646 | 207.6367866140065 |
| 6 | 1772.2761046242847 | 1669.8003489283105 | 1795.5536626430883 | 1678.8798808035058 | 102.47575569597416 | 116.67378183958249 |
