# Forecasting Ontario's electricity demand with a regression and SARIMA errors

From a plain regression on weather and calendar to a day-ahead forecast that halves the error of
"same hour yesterday": the regression and its significance tests, the residuals, a yearly cycle and a trend,
stationarity and differencing, ACF and PACF, AIC and BIC, one backtest month by hand, the choice of the
specification by cross-validation, and a final test on 2025 and on a 2026 holdout.

- **Article:** [archibaldchain.github.io/articles/sarimax-ontario-load.html](https://archibaldchain.github.io/articles/sarimax-ontario-load.html)
- **Notebook:** [`sarimax-ontario-load.ipynb`](sarimax-ontario-load.ipynb), or [open it in Google Colab](https://colab.research.google.com/github/ArchibaldChain/ArchibaldChain.github.io/blob/master/notebooks/sarimax-ontario-load/sarimax-ontario-load.ipynb)
- **Data:** the parquet files in [`data/`](data/), described in [`data/README.md`](data/README.md)

The modelling steps run live in the notebook (the regression, the tests, the ACF and PACF, one SARIMA fit and a
month of day-ahead forecasts, about a minute in all). The long backtests behind the cross-validation and the final
test, several hundred model fits, are saved in `data/` and loaded. The notebook reads only those files, so it runs
anywhere with the packages in [`../requirements.txt`](../requirements.txt).
