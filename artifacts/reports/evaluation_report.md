# Evaluation Report — Retail Demand Forecasting

*Run:* `20260824T190154Z-ae08a3` · *Generated:* 2026-08-24T19:07:05.302129+00:00 · *Data hash:* `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`

*Configuration:* active-product window k = 6 · seed 42 · train targets
2010-03..2011-05 · validation targets
2011-01..2011-05 · hold-out targets
2011-06..2011-11 · never scored: 2011-12 ·
champion gates: max |bias| 10.0 %, meaningful improvement
2.00 wMAPE points, tie band 1.00
points · default z 1.645 (offered: 1.28, 1.645, 2.05) · σ = 1.4826 x
MAD, fallback product -> abc_group -> global.

*Provenance:* every number below was computed by the pipeline and is traceable to a table under
`artifacts/reports/evaluation_tables/`, `artifacts/forecasts/` or `artifacts/reports/`. Narrative
may be enriched by the LLM agent in LLM mode; the same guard checks both versions (PRD §38).

## Comparison of all candidates

| Model | wMAPE | Bias | MAE | RMSE | n rows | Coverage | Negative share (raw) | Δ wMAPE vs B2 | Relative improvement | ≥ threshold vs B2 | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_last_month | 55.7 % | -8.5 % | 85.1 | 250.3 | 19,968 | 100.0 % | 0.0 % | -1.0 % | -1.8 % | False |  |
| B2_ma3 | 54.7 % | -17.4 % | 83.6 | 251.1 | 19,968 | 100.0 % | 0.0 % | 0.0 % | 0.0 % | False |  |
| B3_seasonal_naive | 89.6 % | 35.0 % | 118.6 | 359.8 | 16,529 | 82.8 % | 0.0 % | -34.8 % | -63.7 % | False | B3 (seasonal naive) has no observed month t-12 for every product; metrics computed only on rows where a seasonal forecast exists (see coverage_share). |
| M1_linear | 56.0 % | -18.1 % | 85.6 | 257.8 | 19,968 | 100.0 % | 13.9 % | -1.3 % | -2.4 % | False |  |
| M2_gbm_poisson | 52.5 % | 0.5 % | 80.2 | 236.9 | 19,968 | 100.0 % | 0.0 % | 2.2 % | 4.1 % | False |  |
| M3_gbm_squared | 54.3 % | 1.3 % | 82.9 | 230.7 | 19,968 | 100.0 % | 0.0 % | 0.5 % | 0.8 % | False |  |
| M4_gbm_absolute | 50.2 % | -25.8 % | 76.6 | 257.1 | 19,968 | 100.0 % | 0.2 % | 4.6 % | 8.4 % | False |  |

B3 (seasonal naive) has no observed month t-12 for every product; its metrics are computed only on
rows where a seasonal forecast exists — see the coverage column above.

## By hold-out month and by ABC group

ABC computed on the training window through 2011-05.

### By hold-out month

| Model | Month | wMAPE | Bias | MAE | RMSE | n rows |
|---|---|---|---|---|---|---|
| B1_last_month | 2011-06 | 61.0 % | 9.0 % | 67.2 | 207.1 | 3,284 |
| B1_last_month | 2011-07 | 54.9 % | 0.3 % | 64.6 | 170.3 | 3,289 |
| B1_last_month | 2011-08 | 59.3 % | -3.1 % | 73.8 | 223.0 | 3,310 |
| B1_last_month | 2011-09 | 57.0 % | -23.4 % | 94.1 | 239.8 | 3,324 |
| B1_last_month | 2011-10 | 55.2 % | -4.9 % | 97.9 | 275.5 | 3,373 |
| B1_last_month | 2011-11 | 51.0 % | -16.4 % | 111.6 | 344.7 | 3,388 |
| B2_ma3 | 2011-06 | 58.8 % | -0.7 % | 64.9 | 183.8 | 3,284 |
| B2_ma3 | 2011-07 | 53.6 % | -6.0 % | 63.0 | 167.4 | 3,289 |
| B2_ma3 | 2011-08 | 54.4 % | -4.4 % | 67.7 | 185.1 | 3,310 |
| B2_ma3 | 2011-09 | 53.9 % | -26.6 % | 89.1 | 225.6 | 3,324 |
| B2_ma3 | 2011-10 | 57.3 % | -22.6 % | 101.6 | 284.1 | 3,373 |
| B2_ma3 | 2011-11 | 52.1 % | -27.7 % | 114.0 | 384.6 | 3,388 |
| B3_seasonal_naive | 2011-06 | 94.9 % | 42.8 % | 94.5 | 266.1 | 2,734 |
| B3_seasonal_naive | 2011-07 | 82.5 % | 16.0 % | 84.2 | 205.6 | 2,693 |
| B3_seasonal_naive | 2011-08 | 108.2 % | 47.1 % | 114.6 | 375.4 | 2,722 |
| B3_seasonal_naive | 2011-09 | 92.3 % | 34.6 % | 132.0 | 429.4 | 2,787 |
| B3_seasonal_naive | 2011-10 | 88.9 % | 37.3 % | 134.6 | 363.8 | 2,817 |
| B3_seasonal_naive | 2011-11 | 78.9 % | 32.7 % | 150.1 | 450.1 | 2,776 |
| M1_linear | 2011-06 | 59.1 % | -5.5 % | 65.1 | 163.7 | 3,284 |
| M1_linear | 2011-07 | 53.0 % | -11.7 % | 62.3 | 160.1 | 3,289 |
| M1_linear | 2011-08 | 56.7 % | -6.1 % | 70.6 | 187.0 | 3,310 |
| M1_linear | 2011-09 | 55.9 % | -26.3 % | 92.3 | 229.6 | 3,324 |
| M1_linear | 2011-10 | 57.7 % | -21.7 % | 102.2 | 289.4 | 3,373 |
| M1_linear | 2011-11 | 54.6 % | -25.4 % | 119.4 | 414.1 | 3,388 |
| M2_gbm_poisson | 2011-06 | 56.6 % | 9.1 % | 62.4 | 174.0 | 3,284 |
| M2_gbm_poisson | 2011-07 | 48.2 % | -2.0 % | 56.7 | 149.5 | 3,289 |
| M2_gbm_poisson | 2011-08 | 56.6 % | 8.7 % | 70.4 | 185.8 | 3,310 |
| M2_gbm_poisson | 2011-09 | 52.1 % | -12.0 % | 86.0 | 217.7 | 3,324 |
| M2_gbm_poisson | 2011-10 | 56.9 % | 6.5 % | 100.8 | 269.6 | 3,373 |
| M2_gbm_poisson | 2011-11 | 47.3 % | -2.4 % | 103.6 | 356.6 | 3,388 |
| M3_gbm_squared | 2011-06 | 57.3 % | 7.9 % | 63.2 | 167.4 | 3,284 |
| M3_gbm_squared | 2011-07 | 50.2 % | 3.1 % | 59.0 | 147.3 | 3,289 |
| M3_gbm_squared | 2011-08 | 58.7 % | 10.8 % | 73.1 | 189.9 | 3,310 |
| M3_gbm_squared | 2011-09 | 54.7 % | -10.5 % | 90.3 | 219.5 | 3,324 |
| M3_gbm_squared | 2011-10 | 57.7 % | 4.9 % | 102.2 | 265.9 | 3,373 |
| M3_gbm_squared | 2011-11 | 49.5 % | -2.2 % | 108.2 | 335.4 | 3,388 |
| M4_gbm_absolute | 2011-06 | 50.5 % | -20.5 % | 55.6 | 154.8 | 3,284 |
| M4_gbm_absolute | 2011-07 | 46.7 % | -22.0 % | 54.9 | 158.6 | 3,289 |
| M4_gbm_absolute | 2011-08 | 48.6 % | -23.2 % | 60.6 | 190.0 | 3,310 |
| M4_gbm_absolute | 2011-09 | 52.7 % | -35.7 % | 87.1 | 233.9 | 3,324 |
| M4_gbm_absolute | 2011-10 | 51.9 % | -20.8 % | 92.0 | 281.6 | 3,373 |
| M4_gbm_absolute | 2011-11 | 49.3 % | -28.4 % | 108.0 | 417.1 | 3,388 |

### By ABC group (training-window ABC)

| Model | ABC class | wMAPE | Bias | MAE | RMSE | n rows |
|---|---|---|---|---|---|---|
| B1_last_month | A | 50.5 % | -6.6 % | 168.0 | 382.6 | 5,271 |
| B1_last_month | B | 61.2 % | -12.4 % | 66.7 | 214.1 | 5,658 |
| B1_last_month | C | 64.0 % | -9.7 % | 48.3 | 155.9 | 9,039 |
| B2_ma3 | A | 46.9 % | -11.2 % | 155.7 | 356.7 | 5,271 |
| B2_ma3 | B | 63.6 % | -20.3 % | 69.2 | 248.5 | 5,658 |
| B2_ma3 | C | 67.0 % | -30.7 % | 50.5 | 162.6 | 9,039 |
| B3_seasonal_naive | A | 75.8 % | 38.9 % | 250.1 | 546.6 | 4,771 |
| B3_seasonal_naive | B | 123.3 % | 42.0 % | 107.7 | 342.1 | 4,825 |
| B3_seasonal_naive | C | 128.2 % | -11.6 % | 35.8 | 146.9 | 6,933 |
| M1_linear | A | 46.2 % | -15.8 % | 153.6 | 366.5 | 5,271 |
| M1_linear | B | 62.8 % | -26.6 % | 68.4 | 262.6 | 5,658 |
| M1_linear | C | 75.1 % | -16.5 % | 56.7 | 159.0 | 9,039 |
| M2_gbm_poisson | A | 45.8 % | 0.3 % | 152.1 | 349.8 | 5,271 |
| M2_gbm_poisson | B | 59.3 % | -4.2 % | 64.5 | 232.6 | 5,658 |
| M2_gbm_poisson | C | 63.7 % | 5.4 % | 48.1 | 137.0 | 9,039 |
| M3_gbm_squared | A | 46.0 % | 0.4 % | 152.8 | 341.2 | 5,271 |
| M3_gbm_squared | B | 61.8 % | -4.0 % | 67.3 | 219.4 | 5,658 |
| M3_gbm_squared | C | 68.7 % | 8.5 % | 51.8 | 139.7 | 9,039 |
| M4_gbm_absolute | A | 45.0 % | -22.3 % | 149.5 | 381.4 | 5,271 |
| M4_gbm_absolute | B | 57.2 % | -32.0 % | 62.2 | 253.3 | 5,658 |
| M4_gbm_absolute | C | 57.1 % | -29.2 % | 43.1 | 145.1 | 9,039 |

## Champion decision trace

**Champion:** M2_gbm_poisson (ml). **Best gate-1-passing
baseline:** B1_last_month.
**Improvement over the best baseline:** 3.20 wMAPE points —
**meaningful:** True (threshold:
2.00 points).

Post-hoc bias correction is out of scope for this MVP and is not applied anywhere in this pipeline
(PRD §20).

| Model | wMAPE | Bias | Gate 1 (bias) | Months \|bias\| over threshold | Gate 2 rank | Gate 3 decision | Excluded |
|---|---|---|---|---|---|---|---|
| B1_last_month | 55.7 % | -8.5 % | True | none | 3 | — | — |
| B2_ma3 | 54.7 % | -17.4 % | False | 2011-09, 2011-11 | — | — | — |
| B3_seasonal_naive | 89.6 % | 35.0 % | False | 2011-06, 2011-08, 2011-09, 2011-10, 2011-11 | — | — | reference_only model, partial hold-out coverage (share=0.827774) |
| M1_linear | 56.0 % | -18.1 % | False | 2011-09, 2011-11 | — | — | — |
| M2_gbm_poisson | 52.5 % | 0.5 % | True | none | 1 | — | — |
| M3_gbm_squared | 54.3 % | 1.3 % | True | none | 2 | — | — |
| M4_gbm_absolute | 50.2 % | -25.8 % | False | 2011-09, 2011-11 | — | — | — |

## Back-test consistency

The hold-out comparison above uses each fitted model's single *fixed* fit (through
2011-05), scored once against the whole hold-out. Baselines need no fitting, so
their hold-out rows are the hold-out slice of the rolling back-test. This section instead
re-scores every candidate at **every** rolling back-test origin, to show how stable each model's
wMAPE and bias are across origins rather than at the single hold-out fit.

| Model | Mean wMAPE | Std wMAPE | Min wMAPE | Max wMAPE | Mean bias | Months \|bias\| over threshold |
|---|---|---|---|---|---|---|
| B1_last_month | 66.8 % | 21.5 % | 47.8 % | 133.8 % | 3.9 % | 4 |
| B2_ma3 | 70.7 % | 24.5 % | 52.1 % | 130.4 % | 4.7 % | 6 |
| B3_seasonal_naive | 100.3 % | 14.4 % | 78.9 % | 122.8 % | 42.6 % | 11 |
| M1_linear | 70.9 % | 32.3 % | 53.0 % | 193.6 % | 3.5 % | 2 |
| M2_gbm_poisson | 66.9 % | 27.1 % | 46.8 % | 158.7 % | 9.1 % | 3 |
| M3_gbm_squared | 70.4 % | 29.6 % | 49.1 % | 168.3 % | 13.0 % | 5 |
| M4_gbm_absolute | 58.2 % | 15.8 % | 46.5 % | 112.9 % | -19.2 % | 10 |

## Inventory KPIs per policy

Two policies, simulated for the champion and B2 (the main baseline) on the identical hold-out
rows: `forecast_only` (target = forecast, no safety stock) and `forecast_plus_ss` (target =
forecast + z x robust σ). Output is always a **Recommended Target Inventory**, never an order
quantity — the dataset carries no on-hand or on-order data (PRD §7).

| Model | Policy | Fill rate | Stockout units | Excess units | Stockout SKU-month rate | Excess per unit shortage |
|---|---|---|---|---|---|---|
| M2_gbm_poisson | forecast_only | 73.3 % | 585,231 | 600,835 | 25.3 % | 1.03 |
| M2_gbm_poisson | forecast_plus_ss | 89.9 % | 221,351 | 2,287,053 | 7.3 % | 10.33 |
| B2_ma3 | forecast_only | 65.5 % | 754,376 | 446,805 | 39.1 % | 0.59 |
| B2_ma3 | forecast_plus_ss | 88.3 % | 256,271 | 2,210,581 | 11.3 % | 8.63 |

### z sensitivity (M2_gbm_poisson, forecast_plus_ss)

| z | Fill rate | Excess units | Stockout units |
|---|---|---|---|
| 1.28 | 88.0 % | 1,872,581 | 261,771 |
| 1.645 | 89.9 % | 2,287,053 | 221,351 |
| 2.05 | 91.4 % | 2,758,696 | 188,286 |

Excess is dominated by a few very large orders: for M2_gbm_poisson under
`forecast_plus_ss`, the `top_1pct_share` of SKU-months by excess account for
17.6 % of total excess units, and the `top_5pct_share` account for
42.2 % of 2,287,053 total excess units
(`excess_concentration.csv`).

σ source, champion, most recent hold-out month 2011-11
(3388 active products, `sigma_summary.csv`): product-level
86.0 %, ABC-group level 14.0 %, global level
0.0 %; zero-MAD share 0.0 %.

## Quarterly aggregation

| Model | Scope | Quarter | wMAPE | Bias | n rows |
|---|---|---|---|---|---|
| B2_ma3 | overall | all | 44.3 % | 3.0 % | 16,306 |
| B2_ma3 | quarter | 2010-Q3 | 43.9 % | -9.3 % | 3,350 |
| B2_ma3 | quarter | 2010-Q4 | 41.4 % | 0.5 % | 3,310 |
| B2_ma3 | quarter | 2011-Q1 | 69.3 % | 42.9 % | 3,329 |
| B2_ma3 | quarter | 2011-Q2 | 36.9 % | 4.4 % | 3,127 |
| B2_ma3 | quarter | 2011-Q3 | 35.7 % | -11.7 % | 3,190 |
| M2_gbm_poisson | overall | all | 40.5 % | 7.2 % | 16,306 |
| M2_gbm_poisson | quarter | 2010-Q3 | 46.8 % | -12.1 % | 3,350 |
| M2_gbm_poisson | quarter | 2010-Q4 | 41.9 % | 27.4 % | 3,310 |
| M2_gbm_poisson | quarter | 2011-Q1 | 49.5 % | 17.9 % | 3,329 |
| M2_gbm_poisson | quarter | 2011-Q2 | 33.5 % | 3.5 % | 3,127 |
| M2_gbm_poisson | quarter | 2011-Q3 | 31.5 % | -4.5 % | 3,190 |

This project trains and evaluates a single **one-step-ahead monthly model**: at any forecast
origin, it predicts only the single month that immediately follows. Every quarterly figure in
`quarterly_forecast.csv` is a **sum of three such one-step-ahead forecasts**, each produced at its
own origin (the month before the one it predicts) by the rolling back-test — never a separate
quarterly model, and never a single forecast covering all three months at once.

As a direct consequence, this module
**cannot forecast all three months of a quarter at the start of the quarter**.
The second and third months of any quarter appear here only once the preceding month's data
becomes available, one month at a time.

A genuine start-of-quarter, multi-month forecast is produced by a *separate* module,
`pipeline.multi_horizon`: it runs the same one-step-ahead champion **recursively**, feeding each
month's forecast back into the panel as that month's units before predicting the next, and measures
a distinct sigma for every horizon from a back-test run at that same horizon. Its output is
`multi_horizon_plan.csv` and `period_plan.csv`, and its numbers are *not* interchangeable with the
ones in this file: a later horizon compounds the error of every horizon before it, which is exactly
why it carries its own, wider safety stock. `quarterly_forecast.csv` remains what it says it is —
sums of genuine one-step-ahead forecasts, each made one month before its target.

The rolling estimate for the current partial quarter is not an exception to this: it combines the
already-observed actual sales of the quarter's completed months with the single genuine
one-step-ahead forecast for the next month, and is therefore never treated as a scored, complete
quarter (`complete = False`, no `actual_sum`).

## Reproducibility

Seed: 42. Two runs on the same input data and configuration produce byte-identical
artifacts (CLAUDE.md §2 rule 16) — this run's data hash and configuration are recorded in the
header above.

**Versions:**
* `python 3.11.15`
* `pandas 2.2.3`
* `numpy 1.26.4`
* `sklearn 1.5.2`
* `crewai 0.86.0`
* `streamlit 1.39.0`

**Artifacts registered by this run:**
* `artifacts/models/M1_linear.joblib` — 2,198 bytes
* `artifacts/models/M2_gbm_poisson.joblib` — 744,464 bytes
* `artifacts/models/M3_gbm_squared.joblib` — 386,096 bytes
* `artifacts/models/M4_gbm_absolute.joblib` — 205,280 bytes
* `artifacts/reports/evaluation_tables/abc_train.csv` — 203,688 bytes
* `artifacts/reports/evaluation_tables/backtest_by_origin.csv` — 5,922 bytes
* `artifacts/reports/evaluation_tables/backtest_consistency.csv` — 12,170 bytes
* `artifacts/forecasts/backtest_predictions.csv` — 26,612,474 bytes
* `artifacts/forecasts/baseline_predictions.csv` — 9,435,181 bytes
* `artifacts/models/candidates_meta.json` — 1,879 bytes
* `artifacts/reports/champion_decision.json` — 3,722 bytes
* `data/processed/clean_data.csv` — 7,796,333 bytes
* `data/processed/clean_transactions.parquet` — 9,861,840 bytes
* `artifacts/reports/eda_tables/E01_cleaning_waterfall.csv` — 1,120 bytes
* `artifacts/reports/data_quality_findings.json` — 28,152 bytes
* `artifacts/contracts/dataset_contract.json` — 6,187 bytes
* `artifacts/reports/eda_report.html` — 2,324,154 bytes
* `artifacts/reports/eda_tables/index.json` — 3,835 bytes
* `artifacts/reports/evaluation_tables/evaluation_summary.json` — 3,070 bytes
* `artifacts/reports/evaluation_tables/excess_concentration.csv` — 1,010 bytes
* `artifacts/reports/feature_validation.json` — 1,397 bytes
* `data/processed/features.csv` — 7,457,869 bytes
* `artifacts/reports/evaluation_tables/holdout_metrics_by_abc.csv` — 1,629 bytes
* `artifacts/reports/evaluation_tables/holdout_metrics_by_month.csv` — 2,841 bytes
* `artifacts/reports/evaluation_tables/holdout_metrics_overall.csv` — 1,004 bytes
* `artifacts/forecasts/holdout_predictions.csv` — 5,297,918 bytes
* `artifacts/reports/evaluation_tables/holdout_rows_all_models.csv` — 1,890,035 bytes
* `artifacts/forecasts/holdout_simulation_rows.csv` — 28,690,897 bytes
* `artifacts/reports/evaluation_tables/improvement_vs_b2.csv` — 459 bytes
* `artifacts/reports/insights.md` — 4,640 bytes
* `artifacts/forecasts/inventory_kpis.csv` — 45,927 bytes
* `artifacts/forecasts/inventory_plan.csv` — 802,776 bytes
* `artifacts/forecasts/latest_forecast.csv` — 354,381 bytes
* `artifacts/models/model.joblib` — 744,608 bytes
* `artifacts/models/model_meta.json` — 1,327 bytes
* `artifacts/forecasts/multi_horizon_plan.csv` — 1,589,912 bytes
* `artifacts/forecasts/period_plan.csv` — 6,185,137 bytes
* `artifacts/forecasts/quarterly_forecast.csv` — 4,029,305 bytes
* `artifacts/reports/evaluation_tables/quarterly_limitation.md` — 1,872 bytes
* `artifacts/reports/evaluation_tables/quarterly_metrics.csv` — 1,412 bytes
* `data/processed/returns_lines.parquet` — 132,403 bytes
* `artifacts/reports/evaluation_tables/sigma_summary.csv` — 3,149 bytes
* `artifacts/forecasts/sigma_table.csv` — 7,924,292 bytes
