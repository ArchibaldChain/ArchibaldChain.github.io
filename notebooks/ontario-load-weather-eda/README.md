# What drives Ontario's electricity demand?

Exploratory analysis of 20+ years of hourly Ontario electricity demand against the weather:
the long-run trend, daily and weekly rhythms, the temperature U-curve, heating vs cooling
sensitivity, humidity, calendar effects, the 2020 lockdown, and autocorrelation, ending in
a list of features for a forecasting model.

- **Article:** [archibaldchain.github.io/articles/ontario-load-weather-eda.html](https://archibaldchain.github.io/articles/ontario-load-weather-eda.html)
- **Notebook:** [`ontario-load-weather-eda.ipynb`](ontario-load-weather-eda.ipynb), or [open it in Google Colab](https://colab.research.google.com/github/ArchibaldChain/ArchibaldChain.github.io/blob/master/notebooks/ontario-load-weather-eda/ontario-load-weather-eda.ipynb)
- **Data:** [`data/ontario_hourly.parquet`](data/ontario_hourly.parquet), described in [`data/README.md`](data/README.md)

The notebook reads only the parquet file in `data/`, so it runs anywhere with the packages
in [`../requirements.txt`](../requirements.txt).
