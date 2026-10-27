# Data for the SARIMAX post

Everything the notebook reads. Times are **fixed Eastern Standard Time (UTC−5) all year**, timezone-naive, as in the
[demand post](../../ontario-load-weather-eda/data/README.md): no daylight-saving gaps or repeats.

## `ontario_hourly.parquet`

One row per hour, 2019-01-01 to 2025-12-31.

| Column | Unit | Description |
|---|---|---|
| `datetime` | – | Start of the hour (IESO "hour ending 1" is `00:00`). |
| `demand_mw` | MW | IESO **Ontario Demand** (grid-delivered energy; excludes exports and embedded generation). |
| `temp_c` | °C | Air temperature, Ontario demand-weighted average of five station groups. |
| `dew_c` | °C | Dew-point temperature, same weighting. |
| `is_holiday` | 0/1 | Ontario statutory holidays plus the Civic Holiday (first Monday of August). |
| `is_xmas_week` | 0/1 | December 24 to 31 and January 1. |

A few hours have no weather reading (`NaN`); the notebook interpolates gaps of up to three hours. The weather weighting
is described in the demand post's data README.

## Saved backtest results

These come from the day-ahead walk-forward backtests of the MetaGrid project (origin 00:00 EST, horizons 1 to 24 hours,
parameters re-estimated monthly).

| File | Contents |
|---|---|
| `backtests.parquet` | Hourly forecasts and actuals (`target_time`, `horizon`, `forecast`, `actual`) of seven runs on the 2025 test year (`period = "test 2025"`) and three on January to September 2026 (`"holdout 2026"`): the same hour yesterday, SARIMA on the demand alone, and four SARIMAX specifications up to the final one. |
| `cross_validation.parquet` | Mean absolute error by month (`model`, `month`, `n`, `mae`) of 15 SARIMAX specifications in the rolling-origin cross-validation 2022 to 2024 (36 monthly folds). |
| `order_selection.parquet` | AIC, BIC and log-likelihood of 13 SARIMA orders fitted to the residuals of the regression on 2022 and 2023. |
| `ols_window_grid.parquet` | Out-of-sample error and bias of the regression alone on 2024, by training window and by whether a yearly cycle and a trend are included. |
| `sarima_windows.parquet` | Monthly error on 2024 of the final specification when the SARIMA parameters are estimated on 730, 365, 180 or 90 days, with the run time of each backtest. |

## `live_week.parquet`

Hourly forecasts for 19 October to 1 November 2026 from the final model, trained only on data before each week's
Monday (`week` = `test` or `forecast`). Columns: `actual` (demand published so far), `naive_day`, `naive_week`,
`day_ahead_obs`, `day_ahead_nwp` (updated every midnight with the day's actuals), `week_ahead_obs`, `week_ahead_nwp`
(one forecast on Monday, no updates), `regression_only_nwp`, and `temp_obs` / `temp_nwp` (demand-weighted observed and
forecast temperature, °C). The `_nwp` columns use Open-Meteo forecasts (archived ones for the test week), the `_obs`
columns the observed weather; only the test week has the `_obs` and day-ahead versions.

## Sources and terms

- **Demand:** Independent Electricity System Operator (IESO), Public Reports (`PUB_Demand`),
  <https://reports-public.ieso.ca/public/>. Used under the IESO's terms of use for public reports.
- **Forecast weather (`live_week.parquet`):** [Open-Meteo](https://open-meteo.com/) forecast and historical-forecast APIs
  (CC BY 4.0), temperature and dew point at five points, weighted like the observed weather.
- **Weather:** Environment and Climate Change Canada, historical hourly climate data,
  <https://climate.weather.gc.ca/>. Contains information licensed under the
  [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
