# GTD Country-Month Panel Diagnostics

- Selected panel period: 2000-01-01 to 2020-12-01
- Raw GTD rows loaded: 209,706
- Rows dropped because `imonth` was missing or 0: 20
- Rows dropped because `imonth` was outside 1-12: 0
- Rows dropped because a valid month date could not be created: 0
- Cleaned event rows: 209,686
- Number of countries: 204
- Number of months: 252
- Expected panel rows = countries x months: 51,408
- Actual panel rows: 51,408
- Total attacks in cleaned data within selected period: 139,872
- Total attacks in panel after aggregation: 139,872
- Country-months with at least one strict attack: 9,119
- Share of country-months with at least one strict attack: 0.177

Note: missing `nkill` and `nwound` values are set to zero only while aggregating fatalities and injuries. This is a modeling choice to revisit later.

`attack_month_t` equals 1 when `n_attacks_strict > 0` in the current country-month. It is not an onset variable.

## Missing Expected GTD Variables

- None

## Country Metadata Warnings

- No country codes with multiple names or regions detected.

## Top 20 Countries by `n_attacks_strict`

| country | country_txt | n_attacks_strict |
| --- | --- | --- |
| 95 | Iraq | 23827 |
| 4 | Afghanistan | 15351 |
| 153 | Pakistan | 11836 |
| 92 | India | 9788 |
| 147 | Nigeria | 4841 |
| 160 | Philippines | 4738 |
| 228 | Yemen | 4196 |
| 205 | Thailand | 3286 |
| 182 | Somalia | 3261 |
| 45 | Colombia | 2382 |
| 200 | Syria | 2071 |
| 113 | Libya | 1935 |
| 60 | Egypt | 1683 |
| 167 | Russia | 1661 |
| 141 | Nepal | 1445 |
| 209 | Turkey | 1441 |
| 229 | Democratic Republic of the Congo | 1266 |
| 603 | United Kingdom | 1259 |
| 155 | West Bank and Gaza Strip | 1206 |
| 97 | Israel | 1168 |

## Attacks by Year

| year | n_attacks_all | n_attacks_strict |
| --- | --- | --- |
| 2000 | 1823 | 1461 |
| 2001 | 1912 | 1689 |
| 2002 | 1330 | 1150 |
| 2003 | 1280 | 1092 |
| 2004 | 1164 | 980 |
| 2005 | 2017 | 1634 |
| 2006 | 2757 | 2375 |
| 2007 | 3250 | 2844 |
| 2008 | 4801 | 4343 |
| 2009 | 4723 | 4564 |
| 2010 | 4827 | 4634 |
| 2011 | 5075 | 4662 |
| 2012 | 8525 | 6972 |
| 2013 | 12047 | 10034 |
| 2014 | 16960 | 13547 |
| 2015 | 15138 | 12385 |
| 2016 | 14051 | 11526 |
| 2017 | 11364 | 8994 |
| 2018 | 9853 | 7771 |
| 2019 | 8537 | 6769 |
| 2020 | 8438 | 6324 |
