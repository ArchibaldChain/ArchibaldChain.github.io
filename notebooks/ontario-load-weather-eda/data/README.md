# `ontario_hourly.parquet`

One row per hour from May 2002 onward.

| Column | Unit | Description |
|---|---|---|
| `datetime` | – | Start of the hour, **fixed Eastern Standard Time (UTC−5) all year**, timezone-naive. Both sources publish on this clock, so there are no daylight-saving gaps or repeats. IESO's "hour ending 1" is `00:00` here. |
| `demand_mw` | MW | IESO **Ontario Demand**: energy delivered from the IESO-controlled grid to Ontario customers. Excludes exports and generation embedded in distribution networks (e.g. rooftop solar). |
| `temp_c` | °C | Air temperature, Ontario demand-weighted average (see below). |
| `dew_c` | °C | Dew-point temperature, same weighting. |
| `wind_kph` | km/h | Wind speed, same weighting. |

Weather columns are empty before 2006.

## How the weather is weighted

Hourly observations from seven Environment and Climate Change Canada stations are grouped
into five regions. Each region is weighted by the share of Ontario demand in the IESO zones
it stands in for (shares computed from IESO zonal demand):

| Region | Stations | IESO zones | Weight |
|---|---|---|---|
| Toronto | Toronto City; Toronto Pearson Int'l | Toronto, Essa | 0.42 |
| Southwest | London CS; London Int'l | Southwest, West, Niagara, Bruce | 0.34 |
| Ottawa | Ottawa Macdonald-Cartier Int'l | Ottawa, East | 0.13 |
| Northeast | Sudbury A (from March 2013) | Northeast | 0.08 |
| Northwest | Thunder Bay CS | Northwest | 0.03 |

Within a region the stations are averaged. When a region has no reading for an hour (for
example Sudbury before 2013), the remaining regions' weights are rescaled to sum to one.

## Sources and terms

- **Demand:** Independent Electricity System Operator (IESO), Public Reports
  (`PUB_Demand`, `PUB_DemandZonal`), <https://reports-public.ieso.ca/public/>.
  Used under the IESO's terms of use for public reports.
- **Weather:** Environment and Climate Change Canada, historical hourly climate data,
  <https://climate.weather.gc.ca/>. Contains information licensed under the
  [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).

This file is a derived snapshot, not an official product of either source.
