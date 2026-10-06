---
title: "What drives Ontario's electricity demand?"
date: "2026-09-29"
description: "Twenty years of hourly grid data, twenty years of weather, and what they tell a forecaster."
lang: "en"
github: "https://github.com/ArchibaldChain/ArchibaldChain.github.io/blob/master/notebooks/ontario-load-weather-eda/ontario-load-weather-eda.ipynb"
colab: "https://colab.research.google.com/github/ArchibaldChain/ArchibaldChain.github.io/blob/master/notebooks/ontario-load-weather-eda/ontario-load-weather-eda.ipynb"
---

Every hour of every day, Ontario's grid operator (the IESO) has to match supply to a
number it can't observe in advance: how much power the province will draw. Get it wrong
on the low side and expensive peaking plants scramble; get it wrong on the high side and
you've paid for generation nobody used. Before I build a forecasting model for that
number, I want to understand it.

This post is the exploratory half of that project. I'll start with the rhythm of a single
week, zoom out to two decades, then bring in the weather, piece by piece, until what's
left over is the part weather can't explain. Every chart ends up in the same place: a
list of things a forecasting model will need to know.

> **The data.** Hourly *Ontario Demand* from the IESO, May 2002 to September 2026, about
> 214,000 hours. This is the power delivered from the transmission grid to Ontario
> customers; it excludes exports, and it excludes generation embedded in local
> distribution networks such as rooftop solar, which quietly *reduces* this number.
> Weather is hourly observations from Environment and Climate Change Canada at seven
> stations in five regions (Toronto ×2, London ×2, Ottawa, Sudbury, Thunder Bay), combined into one
> Ontario temperature weighted by each region's share of demand. All times are Eastern
> Standard Time year-round, the clock both sources publish on.
> All the analysis below works at the province level: total Ontario Demand against this
> single weighted temperature.

## Key takeaways

- **Demand fell for a decade, then turned.** Ontario used about 16% less grid electricity in
  2017 than in 2005; by 2025 it was back up 10%.
- **Heat costs more than cold.** Demand bottoms out around 13–14 °C. On workdays each degree
  above 16 °C adds about **550 MW**, each degree below 11 °C only about **200 MW**.
- **Humidity matters.** At the same temperature, humid summer afternoons draw 1.5–2 GW more
  than dry ones.
- **The grid is getting more sensitive to heat.** The cooling slope has risen about 19% since
  2006.
- **Winter peaks keep time; summer peaks follow the heat.** January's busiest hour is 17:00 or
  18:00 every day; July's wanders from 13:00 to 18:00.
- **The calendar is worth a gigawatt.** Weekends run about 1 GW lighter, and a holiday takes out
  as much load as a Sunday.
- **Spring 2020 broke the pattern.** During the lockdown, demand ran 12% below what the weather
  and calendar predicted.
- **Demand remembers.** The best single predictor of an hour is the same hour yesterday
  (r = 0.90).

## A winter week and a summer week

Start small. Here are two weeks from 2025: the one containing the year's winter peak
(a January cold snap) and the one containing its summer peak (the late-June heat wave).

![A winter week and a summer week](/articles/ontario-load-weather-eda/fig-01.png)

The two shapes are different animals. A winter day has **two humps**: one as people wake
up and turn things on, a bigger one in the late afternoon and evening when lights, cooking
and heating stack on top of businesses that haven't closed yet. A summer heat-wave day has
**one tall peak** in the late afternoon, when buildings have soaked up the day's heat and
air conditioners are working hardest.

Notice the scale of the swing, too. In that summer week demand moved by about **12 GW**
between the quietest night and the peak hour, almost twice the winter week's range. The
summer peak, **24.9 GW** at 18:00 EST on June 24 (7 p.m. on the clock), was the highest
hour Ontario has seen since the early 2010s.

Also visible: the weekend. Saturday and Sunday sit lower in both weeks even though the
weather didn't take the weekend off.

Is that week typical? Overlaying every day of the month shows how much the shape
holds up from one day to the next.

![A winter week and a summer week](/articles/ontario-load-weather-eda/fig-02.png)

January's evening peak runs like a clock: on 25 of 31 days the busiest hour started
at 17:00 EST, and on the other six at 18:00. Cold days lift the whole curve rather than
reshaping it, so the spread between days is about the same at 3 a.m. as at the peak.
July is looser. The busiest hour wandered between 13:00 and 18:00 EST depending on when
the heat built up, and the average day swung **7 GW** from its overnight low to its
afternoon high, against **4.5 GW** in January.

**For the forecaster:** the shape of a winter day comes almost entirely from the
calendar; the shape of a summer day needs the hour-by-hour weather.

## Twenty years of demand

![Twenty years of demand](/articles/ontario-load-weather-eda/fig-03.png)

Zoom out and the story becomes one of decline and recovery. Ontario drew **157 TWh** in
2005. Twelve years later, in 2017, it drew **132 TWh**, about **16% less**, despite a
growing population. Canada's energy regulator
[traced the decline](https://www.cer-rec.gc.ca/en/data-analysis/energy-markets/market-snapshots/2018/market-snapshot-why-is-ontarios-electricity-demand-declining.html)
to a mix of conservation programs such as the IESO's Save on Energy, more efficient lighting
and appliances, an economy shifting from energy-intensive industry like pulp and paper toward
services, the 2009 recession, and a wave of solar panels installed after 2009. That last one
matters for how you read this chart:
[*Ontario Demand*](https://www.ieso.ca/power-data/demand-overview/real-time-demand-reports)
counts only energy supplied from the IESO-administered market, while generation connected to
local distribution networks
[offsets demand on the transmission grid](https://www.ieso.ca/en/Power-Data/Supply-Overview/Distribution-Connected-Generation)
before the IESO sees it. So some of the "decline" is load that moved off the grid's books
rather than load that disappeared.

Since 2017 the trend has turned. 2025 came in at **145.6 TWh**, up **10%** from the low,
and the peak hour has climbed back toward 25 GW. The IESO's planning outlooks attribute the
growth to
[electric-vehicle supply-chain manufacturing and data centres](https://ieso.ca/en/Powering-Tomorrow/2025/Seven-Graphs-and-a-Map-2025-Annual-Planning-Outlook),
[greenhouse expansion in the southwest](https://www.ieso.ca/en/Corporate-IESO/Media/News-Releases/2019/10/New-Greenhouse-Study),
and [electrification and population growth](https://ieso.ca/Sector-Participants/Planning-and-Forecasting/Annual-Planning-Outlook/2026-APO-Summary),
and the latest outlook expects demand to grow about 65% by 2050.

Could some of this twenty-year swing simply be the weather, a run of mild years followed by
harsher ones? That question deserves its own analysis, and it's the subject of a follow-up
post.

**For the forecaster:** the level of demand drifts over the years for reasons a weather model
never sees: the economy, efficiency, behind-the-meter generation and electrification. A
day-ahead model picks these up through lagged demand, since yesterday's load already reflects
today's economy; what it must avoid is learning its level from an era that no longer exists.
A forecast years ahead has no such shortcut and has to model those drivers explicitly.

## The daily rhythm changes with the seasons

![The daily rhythm changes with the seasons](/articles/ontario-load-weather-eda/fig-04.png)

Averaging over five years smooths the winter morning hump into a shoulder, but the
seasonal difference is clear. **July** climbs steadily all day and peaks at 16:00–17:00
EST (5–6 p.m. on the clock). **January** jumps early, holds a plateau through the working
day, and peaks at 17:00 EST, when darkness arrives before offices empty. The shoulder
months, **April** and **October**, are the quiet ones: no heating, no cooling, just the
baseline of a province going about its day.

The overnight trough is remarkably stable: 12–15 GW at 2–3 a.m. in every season. That
floor is Ontario's industrial and always-on load, and it's worth remembering later when we
talk about what the weather can't explain.

![The daily rhythm changes with the seasons](/articles/ontario-load-weather-eda/fig-05.png)

The week has its own shape. Monday through Friday are nearly identical; Saturday and
Sunday are about **1 GW (6%) lighter** on average, and the gap is widest during working
hours. Weekend mornings also start later: the dark band that marks the weekday morning
ramp arrives an hour or two late on Saturday and later still on Sunday.

## Enter the weather: the U-curve

Now the main character. If you plot every hour's demand against the temperature at that
hour, twenty years of data collapse into a single, very recognisable shape.

![Enter the weather: the U-curve](/articles/ontario-load-weather-eda/fig-06.png)

This is the **U-curve**, and it's the single most important picture in load forecasting.
Demand is lowest at around **13–14 °C**, a temperature where nobody needs heating or
cooling. Move away from it in either direction and demand climbs.

But the U is lopsided. From the bottom of the curve out to 30 °C, median demand rises by
more than **8 GW**. Going the other way, down to −18 °C, adds only about **4 GW**. Ontario
mostly heats with natural gas, so cold weather reaches the electricity grid indirectly,
through furnace fans, block heaters, longer lighting hours, and the minority of homes with
electric heat. Heat, on the other hand, runs straight into air conditioners, which are
almost all electric.

The vertical spread matters as much as the median line. At any given temperature there's
a band of 5–6 GW between busy and quiet hours. That spread is the time of day, the day of
the week, and the long-term trend, all folded together. We'll pull those apart next.

## How many megawatts per degree?

Hourly data mixes the temperature effect with the daily rhythm. Averaging to **daily**
values removes most of that, and makes the relationship clean enough to fit with straight
lines: a heating slope below one temperature, a cooling slope above another, and a flat
zone in between. The two breakpoints are chosen to fit recent weekdays best.

![How many megawatts per degree?](/articles/ontario-load-weather-eda/fig-07.png)

Three straight lines explain **80%** of the day-to-day variation in workday demand, using
nothing but the daily mean temperature. The best breakpoints are **11 °C** and **16 °C**:
between them, the weather barely matters.

The slopes are the headline numbers:

| | Workday demand change |
|---|---|
| Each °C colder than 11 °C | **+197 MW** |
| Each °C warmer than 16 °C | **+548 MW** |

So each degree of heat adds nearly **three times** as much load as a degree of cold. A
summer day averaging 26 °C (a hot one, since the average includes the night) puts about
5.5 GW on the grid above a mild day. Weekends follow the same shape, shifted down by
roughly a gigawatt, which is the calendar effect from earlier showing up again.

## It's not the heat, it's the humidity

Anyone who has lived through a Toronto summer knows 28 °C and dry is not the same as
28 °C and sticky. The grid seems to know it too. Here are summer weekday afternoons,
grouped by dew point (a measure of how much moisture is in the air).

![It's not the heat, it's the humidity](/articles/ontario-load-weather-eda/fig-08.png)

At every temperature, the humid line sits above the dry one, by roughly **1.5–2 GW**.
Part of that is physics: air conditioners spend energy removing moisture, not just heat,
and people run them harder when it's muggy. Part of it is correlation: humid days tend to
come in multi-day heat waves, when buildings have had no chance to cool off overnight.
Either way the lesson for a model is the same: **temperature alone understates summer
demand**, and dew point (or a humidex-style index) earns its place as a feature.

## Is the grid getting more sensitive to the weather?

If the cooling slope is ~550 MW per degree today, was it always? Fitting the same
three-line model to each year's workdays separately:

![Is the grid getting more sensitive to the weather?](/articles/ontario-load-weather-eda/fig-09.png)

Cooling sensitivity has risen from about **460 MW/°C** (2006–2010 average) to about
**550 MW/°C** (2021–2025), roughly **+19%**, with 2025 the highest year on record at
over 600 MW/°C. More air conditioning in more homes, larger homes, and a growing population
all push the same way. 2020 stands out as well: with people working from home, the
province's air conditioners were running in houses instead of offices, and the slope
jumped.

Heating sensitivity is flatter but has also crept up recently, to above 200 MW/°C in 2024
and 2025. That's what you'd expect as heat pumps start to replace gas furnaces. It's early,
but it's the signal to watch: electrified heating would make the cold side of the U much
steeper.

**For the forecaster:** the weather response isn't fixed. A model that learns "one degree
is worth X megawatts" from 2010 data will underestimate today's heat-wave peaks. Either
weight recent data more heavily, or let the temperature effect interact with a trend.

## What the weather can't explain: the calendar

To separate the calendar from the weather, I fit one regression on 2017–2019 daily data (before the pandemic) with the heating and cooling terms, a humidity interaction, a linear trend, and indicators for each weekday, statutory holidays, and the week between Christmas and New Year. Weather and calendar together explain **90%** of the day-to-day variation in 2017–2019 (in-sample R² = 0.90), and the calendar coefficients read directly as megawatts:

![What the weather can't explain: the calendar](/articles/ontario-load-weather-eda/fig-10.png)

Relative to a Monday with identical weather, Tuesday through Thursday are slightly busier
(about +0.1 GW) and Friday a little lighter. Saturday removes **1.1 GW** and Sunday
**1.3 GW**. A statutory holiday removes **1.4 GW**, so a holiday Monday behaves like a
Sunday. The holiday week at the end of December removes another **0.8 GW** on top of any
weekend or holiday, as offices and factories close between the holidays.

After weather and calendar, the typical daily error is about **470 MW**, around 3% of
average demand. That's a decent first benchmark for the forecasting models to beat, with
one caveat: it uses the *actual* weather, which a real forecast won't have.

## A natural experiment: spring 2020

A model that knows the weather and the calendar can answer a counterfactual question: what
*should* demand have been? Spring 2020 is the most dramatic test there is.

The model from the previous section was fit on 2017–2019 only, so every day on the chart
below, from January 2020 on, is out of sample. It is fed the weather that actually
happened, which makes this a counterfactual ("what would demand have been, given that
weather?") rather than a forecast.

![A natural experiment: spring 2020](/articles/ontario-load-weather-eda/fig-11.png)

Through the first ten weeks of 2020, out of sample and before the pandemic arrived, the model tracks reality within half a percent.
Then Ontario declared a state of emergency on March 17, and demand fell away from the
dashed line. From late March to the end of May it averaged **12% below** what the weather
and calendar predicted, bottoming out about **15.5% below** in early May. Offices,
schools, restaurants and many factories closed; the extra household use didn't come close
to making up for them.

The gap mostly closed over the summer (a hot summer with everyone home kept air
conditioners busy), and by the second half of 2020 demand was within about 2% of
expectations. A smaller shortfall persists into 2021, which is some mix of lasting
changes in how people work and a three-year-old model drifting out of date.

**For the forecaster:** 2020 is not like the other years. It should be flagged or
down-weighted in training, and it's a reminder that no amount of weather data will
predict a policy shock.

## Demand remembers

The last property matters more for forecasting than for understanding: how much does
demand at one hour tell you about demand at another?

![Demand remembers](/articles/ontario-load-weather-eda/fig-12.png)

Demand is strongly autocorrelated. The previous hour explains almost everything
(r = 0.97), which is why very-short-term forecasts are easy. More useful for a day-ahead
forecast are the daily peaks: the **same hour yesterday** has a correlation of **0.90**.
The peaks slowly decay as you go further back, except for a small bump at **7 and 14
days**: the same hour on the same weekday. A forecasting model should be given exactly
these as inputs: demand 24 hours ago, 48 hours ago, and one week ago.

## What this means for a forecast

Pulling it all together, here's what Ontario's demand is made of, and what a model will
need to see:

| What we found | Size | Feature for the model |
|---|---|---|
| Demand bottoms out at 11–16 °C and rises both ways | the U-curve | heating & cooling degrees (hinged temperature) |
| Heat costs ~550 MW/°C, cold ~200 MW/°C | ~3× asymmetry | separate heating and cooling terms, not a single temperature |
| Humidity adds load at the same temperature | +1.5–2 GW on muggy afternoons | dew point, or cooling × dew point |
| Heat sensitivity has grown ~19% since 2006 | +7 MW/°C per year | trend term, or weight recent years more |
| Weekends and holidays are ~1.1–1.4 GW lighter | ~7–9% | day of week, holiday flag, Christmas week |
| The shape of the day depends on the season | two humps vs one peak | hour × season (or hour × temperature) interactions |
| Demand echoes itself | r = 0.90 at 24 h | lagged demand at 24 h, 48 h, 168 h |
| 2020 broke the rules | −12% for ten weeks | flag or down-weight the pandemic period |
| Demand fell until 2017, now rising | −16% then +10% | don't train only on old data |

A simple daily model built from these pieces already gets within about 3% on a typical
day, but only because it was handed the *real* weather. In practice a forecast only has a
weather forecast, which is itself wrong in ways that matter most on exactly the hottest
and coldest days. That's the next post: building the model, and checking how much of its
accuracy survives when it has to use forecast weather instead of the truth.

---

*Data: IESO Public Reports (Ontario Demand); Environment and Climate Change Canada
hourly climate observations (Open Government Licence – Canada).*

*Analysis and drafting done with the help of [Claude Code](https://claude.com/claude-code);
questions, judgement calls and editing are mine. Every number in the text comes from the
notebook.*
