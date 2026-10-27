---
title: "Forecasting Ontario's electricity demand with a regression and SARIMA errors"
date: "2026-10-27"
description: "From a plain regression to a day-ahead forecast that halves the error of \"same hour yesterday\", one decision at a time: what each step bought, and what the tests said."
lang: "en"
github: "https://github.com/ArchibaldChain/ArchibaldChain.github.io/blob/master/notebooks/sarimax-ontario-load/sarimax-ontario-load.ipynb"
colab: "https://colab.research.google.com/github/ArchibaldChain/ArchibaldChain.github.io/blob/master/notebooks/sarimax-ontario-load/sarimax-ontario-load.ipynb"
---

In the [previous post](https://archibaldchain.github.io/articles/ontario-load-weather-eda.html) I took Ontario's hourly
demand apart: the daily and weekly rhythm, the U-shaped response to temperature, the effect of humidity, the holidays.
This post puts those pieces back together into a forecasting model, and it is as much about the *process* as about the
result: what to put in the regression, whether to difference, which order to pick, how long a window to learn from, and
how to evaluate the model without fooling yourself.

The model is a classic. A **regression** on weather and calendar captures what demand *should* be; a **SARIMA** model
captures how the leftover persists from hour to hour and from day to day. The textbook name is a regression with SARIMA
errors, or SARIMAX. It is deliberately a baseline: simple and interpretable, so that fancier models later have something
honest to beat.

### This week's forecast, as a first look

The model of this post, a regression on weather and the calendar with SARIMA errors, is built step by step below. First,
here it is on a week that had only just begun when I ran it (27 October 2026). The week of 19 to 25 October is a small
test sample, and the week of 26 October to 1 November is a real forecast. Two kinds of forecast appear, and the post
is about the first:

- **Day-ahead forecast.** Every midnight, forecast the next 24 hours. When the day is over, its actual demand is fed
  into the model, and the next midnight's forecast starts from there. This is the task of every backtest below.
- **Week-ahead forecast.** One forecast on Monday at 00:00 for all 168 hours of the week, with no actual demand fed in
  along the way.

For each week the model is trained on everything before its Monday, and nothing from the week itself is used (the demand
file ends at midnight on Monday the 26th). Nothing is retuned. Two things are harder than in the backtests, on purpose:

1. **The weather is a forecast.** A week that has not happened has no observed temperature, so both weeks use numerical
   weather forecasts (Open-Meteo, weighted over the same five regions). For the test week these are the archived
   forecasts: against the stations they were off by 0.8 °C on average and 0.6 °C too cold. I downloaded the forecast week's
   weather on the 27th, so Monday's values are closer to what happened than a true week-ahead forecast would have been.
2. **A week-ahead forecast gets no daily update.** The daily update is a Kalman filter step (`extend` in statsmodels) that
   takes the day's actual demand into the SARIMA part of the model. The filter itself is not lost: it turns the history
   into a starting state, and from there the week-ahead forecast runs on the model alone. What is lost is the daily
   correction. To see what that costs, the test week is forecast both ways.

![Forecasting Ontario's electricity demand with a regression and SARIMA errors](/articles/sarimax-ontario-load/fig-01.png)

| test week, 19 to 25 October | MAE (MW) | MAPE (%) |
|---|---:|---:|
| Same hour last week | 566 | 3.71 |
| Same hour yesterday | 513 | 3.43 |
| Day-ahead forecast, observed weather (the backtests' setting) | 249 | 1.68 |
| Day-ahead forecast, forecast weather | 290 | 1.96 |
| Week-ahead forecast, observed weather | 282 | 1.91 |
| Week-ahead forecast, forecast weather | 347 | 2.36 |

In the test week the day-ahead forecast, with forecast weather, misses by about **290 MW** (a MAPE of 2.0%), against 513 MW for repeating yesterday and 566 MW for repeating last week: **43% less error than "same hour
yesterday"**, close to the 49% of the 2025 test year. Forecast weather cost about 40 MW compared with the observed
weather the backtests use (249 MW), which is a first idea of how far those backtests sit from a deployed model.
Taking away the daily update costs a little more: **347 MW** for the week-ahead forecast, still 39% below
"same hour last week". Day by day the week-ahead forecast's error does not rise steadily with the horizon (by day it was 460, 495, 373, 173,
342, 248 and 341 MW from Monday to Sunday): Monday and Tuesday were the hardest days for every variant, Thursday the
easiest. One week is a small sample, but it suggests the daily update is worth tens of MW, not hundreds.

The forecast week, in the shaded part of the figure: weekdays average 15.4 to 15.9 GW with the peak on Tuesday evening
at about 17.7 GW, falling to 14.7 GW on Saturday and 14.3 GW on Sunday, 1 November. I did not tune anything on the test
week, and the forecast cannot be judged until the demand arrives; the same pipeline run a week from now will say how it did.

## Key takeaways

- **A day-ahead error of 390 MW (2.3%), 49% below "same hour yesterday".** On the 2025 test year the model's mean absolute
  error (MAE) was 389.5 MW, a mean absolute percentage error (MAPE) of 2.33%, against 758 MW (4.58%) for repeating
  yesterday. On January to September 2026, data that arrived after every choice had been made, the saving is the same:
  417 MW (2.46%) against 838 MW (4.95%), 50%.
- **The two halves need each other.** SARIMA on the demand history alone reaches 617 MW; with a weather-and-calendar
  regression underneath, the very first version reached 467 MW.
- **A yearly cycle, a trend and a longer training window were worth about 26 MW** in the cross-validation, because demand
  has a seasonal shape that temperature does not explain and it drifts upward.
- **They also removed the need to difference.** With them the residuals pass the stationarity tests, and in the
  cross-validation every model without a plain difference beat every model with one, the best by 7.0 ± 1.4 MW.
- **Estimating the SARIMA part on 90 days is as good as on two years:** within 1 MW, in a quarter of the time.
- **Where it fails:** holidays and the days next to them, and the middle of the day. And these numbers use the
  *observed* weather, so they are an upper bound for a model that has to use weather forecasts: in the
  live check at the top, forecast weather cost about 17% more error.

## The task, and how to keep it honest

**What is being forecast.** Every day at midnight, the model has to predict the next 24 hours of Ontario demand using only
what was known up to that moment: the *day-ahead* task. The error is measured by the **mean absolute error (MAE)** in MW and its percentage version, the **mean absolute percentage error (MAPE)**, both averaged over every forecast hour; models are chosen on the MAE, and the two tell the same story. The yardstick is the simplest
thing that works, *the same hour yesterday*.

**How to evaluate without fooling yourself.**

- **Walk forward.** Parameters are re-estimated on the first of each month from data before that month, then frozen.
  Inside the month the model only watches the actual demand arrive, day by day. Nothing from the future leaks in.
- **Choose on validation years, test once.** Orders, windows and extra terms were chosen with a rolling-origin
  cross-validation on 2022 to 2024: every month is one fold, 36 in all. 2025 is the test year, looked at once at the end.
- **A holdout that arrived later.** January to September 2026 was added after all the choices were frozen.
- **Weather is observed, not forecast.** The model is given the temperature that actually happened on the forecast day. (The live check
  at the top is the exception: it uses forecast weather.)

The long backtests (several hundred model fits) are saved locally.

## Step 1: a regression on weather and the calendar

Start with what demand *should* be, given the weather and the calendar. Ordinary least squares writes demand as a
sum of terms, and everything I found in the previous post becomes a column:

- **Temperature as two hinges.** Demand rises on both sides of a comfortable range, so heating degrees are
  `max(11 − T, 0)` and cooling degrees `max(T − 16, 0)`, each with its own slope.
- **Humidity makes heat cost more:** cooling degrees times the dew point.
- **The shape of the day:** waves with periods of 24 and 12 hours.
- **Day types that change that shape:** Saturday, Sunday, holidays and Christmas week.

That is 23 columns. One choice deserves a word: a day type is not a flat shift. A Sunday is lighter because offices, shops
and factories are closed, which is a *daytime* effect, so each day type gets its own copy of the daily waves. A flat
"Sunday" dummy would lower demand by the same amount at 3 a.m. as at 3 p.m.

The regression is fitted on a fixed two-year window, 1 January 2022 to 31 December 2023 (730 days, 17,520 hours), and explains **89% of the variance** (R² = 0.886). Every block of
terms is overwhelmingly significant: the weather (joint χ² = 4,593), the daily profile (23,341), Saturday (1,110), Sunday
(1,630) and holidays (178). Holidays are the weakest block, which makes sense: a year has only about ten of them, so
their hour-by-hour shape is estimated from very few days.

One detail matters more than it looks. The standard errors are the robust (HAC) ones. As the next step shows, the errors
of this regression are strongly autocorrelated, and the usual standard errors, which assume independent errors, would make
every term look several times more significant than it is.

> **A note on HAC.** HAC stands for heteroskedasticity and autocorrelation consistent (here the Newey-West version). It estimates how uncertain a coefficient is while counting the correlation between errors up to 48 hours apart, with weights that fade as the gap grows. The coefficients are the same as in ordinary least squares; only the standard errors, and so the p-values, change.

|  | coefficient (MW) | HAC se | t | p-value |  | plain OLS se |
|---|---:|---:|---:|---:|---:|---:|
| `const` | 14,410 | 44.4 | 324.4 | <0.0001 | \*\*\* | 10.1 |
| `heat_deg` | 177.9 | 3.1 | 57.2 | <0.0001 | \*\*\* | 0.9 |
| `cool_deg` | -60.6 | 27.6 | -2.2 | 0.0283 | \* | 7 |
| `cool_x_dew` | 30.9 | 1.6 | 19 | <0.0001 | \*\*\* | 0.4 |
| `is_xmas_week` | 85.4 | 132.9 | 0.6 | 0.5205 |  | 38.6 |
| `sin24` | -1,144.5 | 21.1 | -54.3 | <0.0001 | \*\*\* | 9.9 |
| `cos24` | -1,259 | 24.4 | -51.6 | <0.0001 | \*\*\* | 10 |
| `sin12` | -668.2 | 9.4 | -71.1 | <0.0001 | \*\*\* | 9.7 |
| `cos12` | -769.6 | 12.1 | -63.5 | <0.0001 | \*\*\* | 9.7 |
| `is_saturday` | -1,027.4 | 55.5 | -18.5 | <0.0001 | \*\*\* | 16.5 |
| `is_saturday_x_sin24` | 21.3 | 43.6 | 0.5 | 0.6249 |  | 23.3 |
| `is_saturday_x_cos24` | 422 | 46.1 | 9.1 | <0.0001 | \*\*\* | 23.3 |
| `is_saturday_x_sin12` | 230.3 | 23.1 | 10 | <0.0001 | \*\*\* | 23.3 |
| `is_saturday_x_cos12` | 341.8 | 23 | 14.9 | <0.0001 | \*\*\* | 23.3 |
| `is_sunday` | -1,217.5 | 56.3 | -21.6 | <0.0001 | \*\*\* | 16.5 |
| `is_sunday_x_sin24` | -371.9 | 40.2 | -9.2 | <0.0001 | \*\*\* | 23.3 |
| `is_sunday_x_cos24` | 542 | 45.4 | 11.9 | <0.0001 | \*\*\* | 23.3 |
| `is_sunday_x_sin12` | 220 | 23.9 | 9.2 | <0.0001 | \*\*\* | 23.3 |
| `is_sunday_x_cos12` | 278.6 | 23.7 | 11.8 | <0.0001 | \*\*\* | 23.3 |
| `is_holiday` | -1,367 | 136 | -10.1 | <0.0001 | \*\*\* | 34.9 |
| `is_holiday_x_sin24` | -182.2 | 80.5 | -2.3 | 0.0237 | \* | 47.2 |
| `is_holiday_x_cos24` | 400.7 | 110 | 3.6 | 0.0003 | \*\*\* | 47.2 |
| `is_holiday_x_sin12` | 229.3 | 50.7 | 4.5 | <0.0001 | \*\*\* | 47.2 |
| `is_holiday_x_cos12` | 320.4 | 48.1 | 6.7 | <0.0001 | \*\*\* | 47.2 |

All 24 coefficients, in MW per unit of the column (`***` p < 0.001, `**` p < 0.01, `*` p < 0.05). Two terms are not
significant: Christmas week (p = 0.52, nine days a year, and the holiday terms already cover most of it) and the
Saturday × sin24 wave (p = 0.62). Everything else is, but read the weather rows together. The cooling-degree slope on its
own is *negative* (−61 MW per degree) and barely significant, because it is only the slope at a dew point of zero; with the
humidity term the slope is −61 + 31 × dew point, which is positive once the dew point passes about 2 °C, and in summer it
always does. The last column shows what the correction is worth: for the cooling slope the plain standard error is 7.0
against a robust 27.6, so ignoring the autocorrelation would have called it a 8.6-sigma effect instead of 2.2.

**What's the training window?** This step uses one fixed window, the two years before the 2024 validation year, and it exists to explain
and test the regression, not to forecast. The backtests behind the later steps work differently: on the first of every
month the regression is refitted on the *previous 730 days* (the same length as here, but rolling forward), the SARIMA
part is estimated on the last 90 days only (steps 7 and 8), and everything is then frozen for the month. So the coefficients of
a given backtest month differ slightly from this table.

## Step 2: what the regression leaves behind

![Step 2: what the regression leaves behind](/articles/sarimax-ontario-load/fig-02.png)

An R² near 0.9 can hide a lot. The residuals are not noise: they **drift for days at a time**, and their autocorrelation is
0.92 at one hour, 0.59 at 24 hours and still 0.44 at a week. The Durbin-Watson statistic is 0.15, where 2 would mean no
autocorrelation at all.

What a regression on weather and calendar cannot know is *momentum*. If demand ran 1 GW above the prediction in this hour,
it probably will in the next hour, and tomorrow at the same time. Modelling that is the job of the SARIMA part, and
it is why the regression on its own makes a poor forecaster.

## Step 3: a yearly cycle and a trend

I could see two things missing in those residuals, and both are about the slow part of demand:

- **A trend.** Ontario's demand has been growing again after a decade of decline, so a regression fitted on the past
  keeps under-forecasting the present.
- **A yearly cycle** that temperature does not explain (daylight, the school year, summer vacations). I model it with
  two sine and cosine pairs of the day of the year, enough for a shape with two peaks per year.

![Step 3: a yearly cycle and a trend](/articles/sarimax-ontario-load/fig-03.png)

The fitted cycle (left) is double-peaked: about +0.9 GW in January, −0.9 GW in May, a second bump in August. Together with
a trend of **+204 MW per year** it lifts the in-sample R² from 0.886 to 0.922 and shrinks the residual standard deviation
from 754 to 624 MW; the autocorrelation at 24 hours falls from 0.59 to 0.50 and at a week from 0.44 to 0.28. Both blocks
are significant (χ² = 584 and 30).

The other two panels show what this is worth **out of sample**: the plain regression alone, fitted monthly and used to
forecast 2024. Without the extra terms a longer training window makes things *worse* (MAE 599 MW with 365 days, 734 MW
with 1,461 days; the bias goes from −158 to −541 MW), because the regression learns an old level while demand keeps rising.
The yearly cycle alone helps at 365 days (517 MW) but has the same problem with longer windows. With the trend as well, a
longer window helps instead: 494 MW with 730 days, with a bias of only −10 MW. A trend inside a single year cannot be told
apart from the season, so the two come together with a **730-day window**.

## Step 4: stationary or not? How much to difference?

SARIMA models want a stationary series: a stable mean and spread. If the series drifts, the usual cure is to
**difference** it, which means subtracting the previous value (`d`) or the value one season ago (`D`, here 24 hours).
Differencing is not free: it throws away information and, done unnecessarily, adds noise.

Two tests look at this from opposite sides. **ADF** has "there is a unit root" as its null; **KPSS** has "the series is
stationary" as its null. If they agree you know; if they disagree the series is stationary around something that moves.

For the plain regression the two tests **disagree**: ADF rejects the unit root, KPSS rejects stationarity (a statistic of
2.35 against a 5% critical value of 0.463). That is the signature of a level that drifts, and it is exactly what the trend
and the yearly cycle were for. With them, ADF still rejects the unit root and KPSS no longer rejects stationarity (0.17), so
**the plain difference is not needed: d = 0**.

A seasonal difference is another matter. The autocorrelation at 24 hours is still 0.50, so I keep one (`D = 1`: subtract the
same hour of the previous day) and look at what is left.

## Step 5: reading the ACF and the PACF

![Step 5: reading the ACF and the PACF](/articles/sarimax-ontario-load/fig-04.png)

Reading these plots is a craft with two rules of thumb: an **AR** term shows as a PACF that cuts off and an ACF that fades;
an **MA** term is the other way around.

- **The left column, no difference at all:** the daily part already has the look of the middle column (an ACF fading from 0.89, a PACF with its big bars at lags 1 to 3), but at the seasonal lags the ACF stays high and fades slowly: 0.50 at 24 hours, 0.35 at 48 and 0.29 at 72. That is the reason for the seasonal difference (`D = 1`); the middle column shows what is left after it.
- **The non-seasonal part, middle column:** the PACF is large at lags 1 and 2 (0.90 and −0.27) and within ±0.03 up to lag 12,
  while the ACF fades geometrically (0.90, 0.76, 0.63, then 0.35 at 6 hours and 0.13 at 12). That is an **AR(2)**.
- **The seasonal part:** the ACF is −0.35 at lag 24, but only −0.09 and −0.06 at lags 48 and 72, so it cuts off after one
  seasonal lag, while the PACF has a spike beside each seasonal lag (0.35 at 25) that fades slowly. That is a
  **seasonal MA(1)**.
- **The right column** shows what an extra plain difference does: a positive spike at lag 1 and a slowly fading negative
  tail, the mark of over-differencing, which a model could only undo with an MA coefficient of −1.

So the starting point is SARIMA(2,0,0)(0,1,1) with a 24-hour season. With 17,520 hours the 95% band is only ±0.015, so
nearly every bar is "significant": what counts is the size and the pattern.

## Step 6: choosing the order

![Step 6: choosing the order](/articles/sarimax-ontario-load/fig-05.png)

With a starting point, let the data compare neighbours by AIC and BIC (fit quality minus a price per parameter). They are
only comparable between models that difference the same way, so the grey `d = 1` bars are just for reference.

- **The seasonal MA is indispensable** (dropping it costs +4,990 BIC), and so is the second AR term
  (+1,998 with a single AR term, +277 with two).
- **A seasonal AR term pays for itself:** (2,0,1) goes from +250 to +53 when it is added.
- **Beyond that the gains are small:** the best, (3,0,1)(1,1,1), is only 48 to 69 BIC points ahead of its neighbours.

One row is a warning: (2,0,2)(1,1,1) comes out *worse* than (2,0,1)(1,1,1), the model it contains (log-likelihood
−118,060 against −118,017). A maximum-likelihood fit cannot do that, so the optimiser stopped at a local optimum. The
information criteria give a shortlist, not a verdict, and the decision is made by forecasting error.

## Step 7: one backtest month, by hand

Before the cross-validation, here is exactly what one month of a backtest does. On the first of the month:

1. fit the **regression** on the previous 730 days;
2. estimate the **SARIMA parameters** on the residuals of the last 90 days only;
3. filter the last 30 days of residuals through the fitted model to set its state.

Then, for every day of the month: forecast the next 24 hours at midnight (the SARIMA forecast of the residual plus the
regression), and once the day is over, show the model its actual values and move on. The parameters stay fixed inside the
month.

![Step 7: one backtest month, by hand](/articles/sarimax-ontario-load/fig-06.png)

The fitted parameters (above) tell the story of the error process. The two AR terms (1.09 and −0.21) carry the
hour-to-hour momentum. The seasonal MA of **−0.93** means the model corrects with an exponentially fading average of the
errors at the same hour on the previous days, the weights fading by a factor of about 0.93 per day, rather than
trusting yesterday alone.

For July 2025 this hand-made run has an MAE of **463 MW** against 1,107 MW for the same hour yesterday, and it is the
better of the two on 28 of the 31 days. It also reproduces the saved backtest used in the rest of the post to within
0.16 MW; what is left is the optimiser's last digits.

## Step 8: 90 days or 730 days to estimate the SARIMA parameters?

One more choice before the cross-validation: how much history to estimate the SARIMA parameters on. The regression uses 730
days, because its yearly cycle needs more than a year. The SARIMA part only describes how the leftover errors persist from
hour to hour and from day to day, and it may not need two years. A shorter window is also much cheaper, and the fit is
repeated every month of every backtest.

So I re-ran the 2024 validation year with the SARIMA parameters estimated on the last 730, 365, 180 and 90 days of the
residuals (the regression, the order and everything else unchanged):

![Step 8: 90 days or 730 days to estimate the SARIMA parameters?](/articles/sarimax-ontario-load/fig-07.png)

| days for the SARIMA parameters | time (s) | MAE (MW) |
|---|---:|---:|
| 730 | 345 | 384.7 |
| 365 | 202 | 385.8 |
| 180 | 134 | 386.8 |
| 90 | 80 | 385.5 |

All four are within about 2 MW of each other, which is inside the noise of one validation year, and 730 days is not even the
best of the shorter ones by much (384.7 MW against 385.5 MW for 90 days). But the 90-day fit makes a year of monthly
refits about four times faster (1.3 against 5.7 minutes). That is the window I use from here on; the regression keeps its 730.

## Step 9: deciding by cross-validation

Now the decision. The cross-validation turns every month of 2022 to 2024 into a fold: 36 times, a month is forecast by a
model fitted on data before it. I compared 15 specifications that differ in the order, in whether there is a plain
difference, and in how many yearly harmonics the regression has. The figure shows each one's MAE and, in
markers, the three years separately.

![Step 9: deciding by cross-validation](/articles/sarimax-ontario-load/fig-08.png)

- **No plain difference wins.** Every `d = 0` model (380.7 to 384.6 MW) beats every `d = 1` model, and the best,
  (2,0,2)(1,1,1), is **7.0 ± 1.4 MW** ahead of the (2,1,1) I started with (paired differences by month, mean ± standard
  error). This agrees with the stationarity tests of step 4.
- **Two yearly harmonics are enough.** One is 20.3 ± 2.8 MW worse, three or four gain nothing (+0.3 ± 1.3 and
  −1.1 ± 1.3 MW). Dropping the yearly cycle, the trend and the long window costs 26 MW (414.1 against 387.7).
- **Among the `d = 0` models the differences are inside the noise** (380.7, 381.9, 381.9, 382.0 MW), so I took the nominal
  winner, (2,0,2)(1,1,1), and not the one the information criteria liked.

The final specification, frozen before looking at 2025: a regression on weather, calendar, two yearly harmonics and a
trend over the previous 730 days, with **SARIMA(2,0,2)(1,1,1)₂₄** on its residuals, parameters estimated on the last 90
days and refreshed every month.

## The final exam: 2025, and a holdout nobody has tuned on

Now the part that matters. The frozen specification was run once on 2025, and then on January to September 2026.

![The final exam: 2025, and a holdout nobody has tuned on](/articles/sarimax-ontario-load/fig-09.png)

- **2025: 389.5 MW** (MAPE 2.33%) against 758.0 MW (4.58%) for the same hour yesterday, a saving of **48.6%**.
  The left panel is the whole journey, and every step took a bite: SARIMA alone 617 MW, the regression underneath it
  467, a second AR term 428, the yearly cycle and trend 395, and the cross-validated `d = 0` order 390.
- **2026 holdout: 417.2 MW** (2.46%) against 838.5 MW (4.95%), a saving of **50.2%**. The absolute error is higher than in 2025, but so is
  the baseline's, and the saving is the same, so the model did not degrade on data it had never seen.
- **The gain over the specification before the cross-validation is real but modest:** the previous (2,1,1) scored 398.7 MW
  on 2025 and 419.0 MW on the holdout, so 9.2 MW and 1.8 MW.
- **The gap holds in every month** (right panel), summer and winter, 2025 and 2026.

![The final exam: 2025, and a holdout nobody has tuned on](/articles/sarimax-ontario-load/fig-10.png)

To see what these errors look like, here is the hardest week of the test year: the week of the 2025 summer peak, a heat
wave. "Same hour yesterday" is a day late whenever the weather moves: it under-forecasts while the heat builds and
over-forecasts when it breaks, and its error that week is **1,771 MW**. The model, a regression plus SARIMA errors,
follows the shape and misses by **544 MW**.

## Where it still fails

![Where it still fails](/articles/sarimax-ontario-load/fig-11.png)

- **By weekday, Mondays are the hardest (414 MW) and Fridays the easiest (360 MW).** A Monday follows a weekend, and the same
  hour yesterday is a poor guide to it. In percentages the weekend is as hard (MAPE 2.49% on Saturday, 2.48% on Monday, 2.15% on
  Friday), because weekend demand is lower, so the same MW error is a bigger share.
- **By hour, the middle of the day is the hardest.** Because every forecast is made at midnight, the horizon is also the
  hour of the day: the error is 189 MW in the first hour, when the model still remembers last night, climbs to about
  570 MW around noon and 2 p.m. (MAPE 1.29% in the first hour, 3.37% around noon), when, I'd guess, air conditioning and the office day set the level, and is 348 MW at the end of the
  day.
- **Four of the eight worst days sit next to a holiday.** January 2 (the Thursday after New Year's Day) and Monday June 30
  (the day before Canada Day) were over-forecast by about 1.1 GW (MAPE 7.1% and 6.6%), because many people took the day off; Boxing Day and the
  Sunday after Christmas were under-forecast by about 0.9 GW. A holiday flag cannot know about *bridge days*. The other
  four (June 25, July 7, July 24 and a Saturday in March) I cannot explain from weather and calendar alone.

## The recipe, and what comes next

| Step | The question | What I found | The decision |
|---|---|---|---|
| 1. Regression | Do weather and calendar explain demand? | R² 0.89; every block significant (robust tests) | keep it |
| 2. Residuals | Are the errors noise? | autocorrelation 0.92 at 1 h, Durbin-Watson 0.15 | model them with SARIMA |
| 3. Yearly cycle, trend | What slow part is left? | ±1 GW cycle, +204 MW per year; sd 754 → 624 MW | add both, learn from 730 days |
| 4. Stationarity | Difference? | tests agree only after step 3 | `d = 0`, `D = 1` |
| 5. ACF and PACF | Which terms? | AR(2), seasonal MA(1) | start at (2,0,0)(0,1,1) |
| 6. AIC and BIC | Which order? | a seasonal AR also pays, then small gains | a shortlist, not a verdict |
| 7. One backtest month | How is the forecast made? | refit monthly, roll the filter daily | no leakage |
| 8. Cross-validation | Which specification? | `d = 0` by 7.0 ± 1.4 MW; two harmonics | (2,0,2)(1,1,1), 90-day SARIMA window |

**What this baseline cannot do.** It uses the weather that actually happened. A real forecast has to use *forecast* weather,
which is worse, so the next step is to repeat the evaluation with archived weather forecasts. It also knows holidays
only as a yes or a no, so bridge days and the days around holidays are the clearest place to improve. And a point forecast
says nothing about how wrong it might be, which is what the probabilistic work after this needs.

A first gradient-boosting model on the same backtest, same observed weather, reaches about 300 MW. A regression with SARIMA
errors will not win on accuracy; its job is to be a transparent baseline, one whose every term can be read, tested and
explained.

All the code is in the notebook linked at the top of this page, and the backtest results it reads are in its `data` folder.
