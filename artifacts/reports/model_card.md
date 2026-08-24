# Model Card — Retail Demand Forecasting

*Run:* `20260824T181001Z-27103c` · *Generated:* 2026-08-24T18:16:10.224440+00:00 · *Data hash:* `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`

*Provenance:* every number below was computed by the pipeline and is traceable to a table under
`artifacts/reports/evaluation_tables/`, `artifacts/forecasts/`, `artifacts/models/` or
`artifacts/contracts/`. Narrative may be enriched by the LLM agent in LLM mode; the same guard
checks both versions (PRD §38).

## 1. Model purpose

This model forecasts **SKU x month gross demand**: the number of units of one product expected to
sell in the following calendar month, for every product **active** in the 6 months
before the target month (PRD §14). Its forecast feeds a separate, deterministic inventory-policy
layer (PRD §24) that converts it into a **Recommended Target Inventory** using safety stock derived
from out-of-sample forecast errors; the model itself never predicts inventory. The dataset carries
no on-hand or on-order data, so this output is never an order quantity (PRD §7).

## 2. Training data summary

The panel used for training and evaluation covers **100,717** product-month rows across
**4,723** products, `2009-12` to `2011-11` complete, plus
the partial month `2011-12` (shown, never scored).

* **Training window:** `2010-03` to `2011-05` — 52,214 rows fed to
  each hold-out candidate model (`candidates_meta.json`).
* **Hold-out window:** `2011-06` to `2011-11` — months the model never saw
  during training, kept aside to test it honestly.
* **Operational refit:** `model.joblib` is the champion configuration refit through
  `2011-11` (`model_meta.json`), producing the `2011-12`
  forecast.
* **Active-product window:** k = 6 months (PRD §14).

**Cleaning summary** (`data_quality_findings.json` waterfall): 1,067,371 raw sales-line rows in,
1,003,338 rows out after cleaning and de-duplication.

**Source:** `UCI Online Retail II (CC BY 4.0); Kaggle mirror mashlyn/online-retail-ii-uci`. **Citation:** `Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. CC BY 4.0`.

## 3. Metrics

wMAPE and Bias are always reported together (PRD §23) — never one without the other.

| Model | wMAPE | Bias | MAE | RMSE | n rows | Coverage | Δ wMAPE vs B2 |
|---|---|---|---|---|---|---|---|
| B1_last_month | 55.7 % | -8.5 % | 85.1 | 250.3 | 19,968 | 100.0 % | -1.0 % |
| B2_ma3 | 54.7 % | -17.4 % | 83.6 | 251.1 | 19,968 | 100.0 % | 0.0 % |
| B3_seasonal_naive | 89.6 % | 35.0 % | 118.6 | 359.8 | 16,529 | 82.8 % | -34.8 % |
| M1_linear | 55.9 % | -18.2 % | 85.3 | 257.4 | 19,968 | 100.0 % | -1.1 % |
| M2_gbm_poisson | 52.6 % | 0.6 % | 80.3 | 237.4 | 19,968 | 100.0 % | 2.2 % |
| M3_gbm_squared | 54.4 % | 1.6 % | 83.1 | 231.4 | 19,968 | 100.0 % | 0.3 % |
| M4_gbm_absolute | 49.9 % | -25.2 % | 76.2 | 255.8 | 19,968 | 100.0 % | 4.8 % |

**Champion:** M2_gbm_poisson (ml), selected by the PRD §20 gates
(bias, accuracy, inventory tie-break, meaningful improvement), executed by code — never picked by
hand. Best gate-1-passing baseline: B1_last_month.
Improvement over that baseline: 3.12 wMAPE points (meaningful:
True).

### By hold-out month

| Model | Month | wMAPE | Bias |
|---|---|---|---|
| B1_last_month | 2011-06 | 61.0 % | 9.0 % |
| B1_last_month | 2011-07 | 54.9 % | 0.3 % |
| B1_last_month | 2011-08 | 59.3 % | -3.1 % |
| B1_last_month | 2011-09 | 57.0 % | -23.4 % |
| B1_last_month | 2011-10 | 55.2 % | -4.9 % |
| B1_last_month | 2011-11 | 51.0 % | -16.4 % |
| B2_ma3 | 2011-06 | 58.8 % | -0.7 % |
| B2_ma3 | 2011-07 | 53.6 % | -6.0 % |
| B2_ma3 | 2011-08 | 54.4 % | -4.4 % |
| B2_ma3 | 2011-09 | 53.9 % | -26.6 % |
| B2_ma3 | 2011-10 | 57.3 % | -22.6 % |
| B2_ma3 | 2011-11 | 52.1 % | -27.7 % |
| B3_seasonal_naive | 2011-06 | 94.9 % | 42.8 % |
| B3_seasonal_naive | 2011-07 | 82.5 % | 16.0 % |
| B3_seasonal_naive | 2011-08 | 108.2 % | 47.1 % |
| B3_seasonal_naive | 2011-09 | 92.3 % | 34.6 % |
| B3_seasonal_naive | 2011-10 | 88.9 % | 37.3 % |
| B3_seasonal_naive | 2011-11 | 78.9 % | 32.7 % |
| M1_linear | 2011-06 | 59.0 % | -5.5 % |
| M1_linear | 2011-07 | 52.8 % | -11.8 % |
| M1_linear | 2011-08 | 56.4 % | -6.2 % |
| M1_linear | 2011-09 | 55.7 % | -26.4 % |
| M1_linear | 2011-10 | 57.6 % | -21.7 % |
| M1_linear | 2011-11 | 54.4 % | -25.3 % |
| M2_gbm_poisson | 2011-06 | 56.7 % | 9.3 % |
| M2_gbm_poisson | 2011-07 | 48.1 % | -2.4 % |
| M2_gbm_poisson | 2011-08 | 57.2 % | 10.3 % |
| M2_gbm_poisson | 2011-09 | 52.0 % | -11.6 % |
| M2_gbm_poisson | 2011-10 | 56.9 % | 6.4 % |
| M2_gbm_poisson | 2011-11 | 47.2 % | -3.0 % |
| M3_gbm_squared | 2011-06 | 57.6 % | 8.4 % |
| M3_gbm_squared | 2011-07 | 50.3 % | 3.6 % |
| M3_gbm_squared | 2011-08 | 58.7 % | 11.1 % |
| M3_gbm_squared | 2011-09 | 54.8 % | -10.4 % |
| M3_gbm_squared | 2011-10 | 57.8 % | 5.0 % |
| M3_gbm_squared | 2011-11 | 49.6 % | -1.9 % |
| M4_gbm_absolute | 2011-06 | 50.2 % | -19.7 % |
| M4_gbm_absolute | 2011-07 | 46.6 % | -21.9 % |
| M4_gbm_absolute | 2011-08 | 48.4 % | -22.3 % |
| M4_gbm_absolute | 2011-09 | 52.5 % | -35.0 % |
| M4_gbm_absolute | 2011-10 | 51.9 % | -20.3 % |
| M4_gbm_absolute | 2011-11 | 48.9 % | -27.8 % |

### By ABC group (training-window ABC)

| Model | ABC class | wMAPE | Bias |
|---|---|---|---|
| B1_last_month | A | 50.5 % | -6.6 % |
| B1_last_month | B | 61.2 % | -12.4 % |
| B1_last_month | C | 64.0 % | -9.7 % |
| B2_ma3 | A | 46.9 % | -11.2 % |
| B2_ma3 | B | 63.6 % | -20.3 % |
| B2_ma3 | C | 67.0 % | -30.7 % |
| B3_seasonal_naive | A | 75.8 % | 38.9 % |
| B3_seasonal_naive | B | 123.3 % | 42.0 % |
| B3_seasonal_naive | C | 128.2 % | -11.6 % |
| M1_linear | A | 46.2 % | -15.5 % |
| M1_linear | B | 62.6 % | -26.9 % |
| M1_linear | C | 74.7 % | -17.0 % |
| M2_gbm_poisson | A | 45.8 % | 0.4 % |
| M2_gbm_poisson | B | 59.5 % | -4.2 % |
| M2_gbm_poisson | C | 63.7 % | 5.5 % |
| M3_gbm_squared | A | 46.2 % | 0.7 % |
| M3_gbm_squared | B | 61.8 % | -3.9 % |
| M3_gbm_squared | C | 68.9 % | 8.8 % |
| M4_gbm_absolute | A | 44.8 % | -21.2 % |
| M4_gbm_absolute | B | 57.0 % | -31.9 % |
| M4_gbm_absolute | C | 56.8 % | -29.4 % |

## 4. Limitations

* **Twenty-four months of history only.** The dataset runs `2009-12` to
  `2011-11` complete; there is no earlier history to validate longer seasonal cycles.
* **December 2011 is partial** (`2011-12`) — shown as a partial actual, never scored,
  and it is the target of the latest operational forecast only.
* **No on-hand or on-order inventory data.** The output is a **Recommended Target Inventory**, never
  an order quantity (PRD §7); an actual replenishment quantity cannot be computed from this dataset.
* **Cold start.** A product with no observed month at or before the forecast origin gets no model
  forecast and no invented history (status `Insufficient History / New Product`).
* **Extreme orders.** A handful of very large wholesale lines exist in the data; safety stock uses a
  robust spread measure (median absolute deviation) rather than a plain standard deviation so a few
  outliers cannot inflate the recommendation for every similar product (PRD §26).
* **Gross demand, not net of returns.** The target is units sold on positive sales lines;
  cancellations and adjustments are never subtracted, because inventory was needed to fulfil the
  original order (PRD §9).
* **Left-censoring.** Products already selling in the first observed month may be older than the
  data shows; `product_age_months` is a lower bound, not a true age (PRD §47).
* **Service-level disclaimer:** `z = 1.645 does not guarantee a 95% fill rate; the achieved fill rate is measured in the back-test (PRD §25)`
* **Quarterly figures are sums of one-step-ahead monthly forecasts, not a genuine three-month-ahead
  forecast** — see the methodology note directly below.
* **σ fallback.** A product needs at least 6 out-of-sample
  residuals to use its own σ; otherwise the estimate falls back through
  product -> abc_group -> global (PRD §27).
* **Post-hoc bias correction is out of scope** for this MVP — it proved unstable and is explicitly
  excluded (PRD §20).

### Quarterly methodology note

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

## 5. Ethical considerations

* **No decisions about individuals.** This model forecasts aggregate product demand; it never
  scores, ranks or makes a decision about a person.
* **Customer ID is never a feature.** Rows without a `Customer ID` are kept (the sale still
  happened and still needed stock), and the identifier itself is diagnostic-only, never a model
  input.
* **Anonymised data.** The underlying dataset carries no names, addresses or other direct
  identifiers — only a numeric customer id, country and transaction detail.
* **Residual risks:**
  * *Blind reliance* — a planner treating the forecast or the recommended inventory as certain
    rather than an estimate with a measured error and fill rate.
  * *Overstock from poor uncertainty estimates* — σ estimated from too few residuals (product,
    then ABC-group, then global fallback) can misstate the safety stock for a thin-history product.
  * *Stockouts from a biased model* — a model whose bias gate would fail if re-checked out of band
    (e.g. after a demand shift the pipeline has not yet re-evaluated).
  * *Misinterpretation of safety stock* — `z = 1.645 does not guarantee a 95% fill rate; the achieved fill rate is measured in the back-test (PRD §25)`
* **Scope of the recommendation.** Every number here relies only on the historical sales recorded
  in this dataset; it carries no external signal (promotions, competitor activity, macroeconomic
  data) and should not be read as one.
* **Licensing and attribution.** Source: `UCI Online Retail II (CC BY 4.0); Kaggle mirror mashlyn/online-retail-ii-uci`. Citation: `Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. CC BY 4.0`. This
  attribution is carried into every derived artifact this pipeline produces.

## Configuration

* Active-product window: k = 6 months.
* Lead time: 1 month(s).
* Safety-stock z: 1.645 (offered: 1.28, 1.645, 2.05).
* σ: 1.4826 x MAD, minimum 6 residuals, fallback
  product -> abc_group -> global.
* Champion gates: max |bias| 10.0 %, monthly bias report threshold
  25.0 %, tie band 1.00 points,
  meaningful improvement 2.00 points, similar-fill-rate
  tolerance 1.0 %.
* Temporal split: train 2010-03..2011-05, validation
  2011-01..2011-05, hold-out
  2011-06..2011-11, never scored: 2011-12.

## Version

* Run: `20260824T181001Z-27103c`. Data hash: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`. Seed: 42.
* Library versions:
  * `python 3.11.15`
  * `pandas 2.2.3`
  * `numpy 1.26.4`
  * `sklearn 1.5.2`
  * `crewai 0.86.0`
  * `streamlit 1.39.0`
* Artifacts registered by this run:
  * `artifacts/models/M1_linear.joblib` — 2,198 bytes
  * `artifacts/models/M2_gbm_poisson.joblib` — 744,464 bytes
  * `artifacts/models/M3_gbm_squared.joblib` — 386,096 bytes
  * `artifacts/models/M4_gbm_absolute.joblib` — 205,280 bytes
  * `artifacts/reports/evaluation_tables/abc_train.csv` — 203,688 bytes
  * `artifacts/reports/evaluation_tables/backtest_by_origin.csv` — 5,922 bytes
  * `artifacts/reports/evaluation_tables/backtest_consistency.csv` — 12,170 bytes
  * `artifacts/forecasts/backtest_predictions.csv` — 26,612,607 bytes
  * `artifacts/forecasts/baseline_predictions.csv` — 9,435,181 bytes
  * `artifacts/models/candidates_meta.json` — 1,879 bytes
  * `artifacts/reports/champion_decision.json` — 3,724 bytes
  * `data/processed/clean_data.csv` — 7,796,333 bytes
  * `data/processed/clean_transactions.parquet` — 9,861,840 bytes
  * `artifacts/reports/eda_tables/E01_cleaning_waterfall.csv` — 1,120 bytes
  * `artifacts/reports/data_quality_findings.json` — 28,152 bytes
  * `artifacts/contracts/dataset_contract.json` — 6,187 bytes
  * `artifacts/reports/eda_report.html` — 2,324,154 bytes
  * `artifacts/reports/eda_tables/index.json` — 3,835 bytes
  * `artifacts/reports/evaluation_report.md` — 16,386 bytes
  * `artifacts/reports/evaluation_tables/evaluation_summary.json` — 3,070 bytes
  * `artifacts/reports/evaluation_tables/excess_concentration.csv` — 1,010 bytes
  * `artifacts/reports/feature_validation.json` — 1,397 bytes
  * `data/processed/features.csv` — 7,457,869 bytes
  * `artifacts/reports/evaluation_tables/holdout_metrics_by_abc.csv` — 1,629 bytes
  * `artifacts/reports/evaluation_tables/holdout_metrics_by_month.csv` — 2,841 bytes
  * `artifacts/reports/evaluation_tables/holdout_metrics_overall.csv` — 1,004 bytes
  * `artifacts/forecasts/holdout_predictions.csv` — 5,299,096 bytes
  * `artifacts/reports/evaluation_tables/holdout_rows_all_models.csv` — 1,890,529 bytes
  * `artifacts/forecasts/holdout_simulation_rows.csv` — 28,692,000 bytes
  * `artifacts/reports/evaluation_tables/improvement_vs_b2.csv` — 459 bytes
  * `artifacts/reports/insights.md` — 4,640 bytes
  * `artifacts/forecasts/inventory_kpis.csv` — 45,927 bytes
  * `artifacts/forecasts/inventory_plan.csv` — 802,744 bytes
  * `artifacts/forecasts/latest_forecast.csv` — 354,373 bytes
  * `artifacts/models/model.joblib` — 744,608 bytes
  * `artifacts/models/model_meta.json` — 1,328 bytes
  * `artifacts/forecasts/multi_horizon_plan.csv` — 1,589,839 bytes
  * `artifacts/forecasts/period_plan.csv` — 6,185,025 bytes
  * `artifacts/forecasts/quarterly_forecast.csv` — 4,029,292 bytes
  * `artifacts/reports/evaluation_tables/quarterly_limitation.md` — 1,872 bytes
  * `artifacts/reports/evaluation_tables/quarterly_metrics.csv` — 1,412 bytes
  * `data/processed/returns_lines.parquet` — 132,403 bytes
  * `artifacts/reports/evaluation_tables/sigma_summary.csv` — 3,149 bytes
  * `artifacts/forecasts/sigma_table.csv` — 7,924,303 bytes
