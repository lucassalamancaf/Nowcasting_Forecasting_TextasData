# GTD Initial Data Description

This report describes the processed country-month panel. `attack_month_t` equals 1 when `n_attacks_strict > 0` in the same month. Forecasting outcomes are built from strict GTD attacks.

## Panel Balance

- Country-month rows: 51,408
- Countries: 204
- Months: 252
- Balanced panel: True
- Duplicate country-month cells: 0

## Attack-Month Indicator

- Country-months with `attack_month_t == 1`: 9,119
- Country-months with `attack_month_t == 0`: 42,289
- Share with at least one strict attack: 0.177
- Countries with zero strict attacks in the selected period: 37

## Binary Outcome Class Balance

| outcome | definition | positive_rows | negative_rows | total_rows | positive_rate | negative_rate | extra_positive_vs_strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strict_attack_month | n_attacks_strict > 0 | 9119 | 42289 | 51408 | 0.177 | 0.823 | 0 |
| all_gtd_attack_month | n_attacks_all > 0 | 9713 | 41695 | 51408 | 0.189 | 0.811 | 594 |

## Forecasting Outcomes

`hard_onset_next_3months_t` uses a 60-month clean spell.

| outcome | definition | positive_rows | negative_rows | valid_rows | missing_rows | positive_rate | negative_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incidence_next_month_t | strict attack occurs in t+1 | 9080 | 42124 | 51204 | 204 | 0.177 | 0.823 |
| onset_next_month_t | no strict attack in t; strict attack occurs in t+1 | 2587 | 39538 | 42125 | 9283 | 0.061 | 0.939 |
| onset_next_3months_t | no strict attack in t; strict attack occurs in t+1, t+2, or t+3 | 5701 | 36106 | 41807 | 9601 | 0.136 | 0.864 |
| hard_onset_next_3months_t | 60 prior attack-free months through t; strict attack occurs in t+1, t+2, or t+3 | 308 | 13953 | 14261 | 37147 | 0.022 | 0.978 |

## Three-Month Onset Rates by Region

| outcome | region_txt | valid_rows | positive_rows | negative_rows | positive_rate |
| --- | --- | --- | --- | --- | --- |
| hard_onset_next_3months_t | Central Asia | 207 | 12 | 195 | 0.058 |
| hard_onset_next_3months_t | Eastern Europe | 1989 | 57 | 1932 | 0.029 |
| hard_onset_next_3months_t | East Asia | 600 | 17 | 583 | 0.028 |
| hard_onset_next_3months_t | Middle East & North Africa | 816 | 22 | 794 | 0.027 |
| hard_onset_next_3months_t | Western Europe | 1251 | 30 | 1221 | 0.024 |
| hard_onset_next_3months_t | South America | 758 | 18 | 740 | 0.024 |
| hard_onset_next_3months_t | South Asia | 261 | 6 | 255 | 0.023 |
| hard_onset_next_3months_t | Sub-Saharan Africa | 3541 | 80 | 3461 | 0.023 |
| hard_onset_next_3months_t | Southeast Asia | 703 | 14 | 689 | 0.02 |
| hard_onset_next_3months_t | Central America & Caribbean | 2834 | 39 | 2795 | 0.014 |
| hard_onset_next_3months_t | Australasia & Oceania | 1301 | 13 | 1288 | 0.01 |
| onset_next_3months_t | North America | 430 | 230 | 200 | 0.535 |
| onset_next_3months_t | South Asia | 1001 | 264 | 737 | 0.264 |
| onset_next_3months_t | Middle East & North Africa | 3700 | 778 | 2922 | 0.21 |
| onset_next_3months_t | Western Europe | 4664 | 879 | 3785 | 0.188 |
| onset_next_3months_t | South America | 2941 | 525 | 2416 | 0.179 |
| onset_next_3months_t | Central Asia | 1846 | 292 | 1554 | 0.158 |
| onset_next_3months_t | Southeast Asia | 1929 | 302 | 1627 | 0.157 |
| onset_next_3months_t | Sub-Saharan Africa | 10161 | 1406 | 8755 | 0.138 |
| onset_next_3months_t | East Asia | 1633 | 170 | 1463 | 0.104 |
| onset_next_3months_t | Eastern Europe | 5922 | 577 | 5345 | 0.097 |
| onset_next_3months_t | Australasia & Oceania | 2423 | 111 | 2312 | 0.046 |
| onset_next_3months_t | Central America & Caribbean | 5157 | 167 | 4990 | 0.032 |

## Top Regions by Strict Attacks

| region_txt | countries | country_months | attack_months | attack_month_share | n_attacks_strict | fatalities_all | injuries_all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| South Asia | 9 | 2268 | 1252 | 0.552 | 40295 | 105570 | 135531 |
| Middle East & North Africa | 23 | 5796 | 2055 | 0.355 | 39895 | 129472 | 201229 |
| Sub-Saharan Africa | 49 | 12348 | 2079 | 0.168 | 15337 | 78617 | 45175 |
| Southeast Asia | 11 | 2772 | 820 | 0.296 | 9147 | 9750 | 19777 |
| Western Europe | 23 | 5796 | 1076 | 0.186 | 3635 | 1017 | 6202 |
| Eastern Europe | 26 | 6552 | 560 | 0.085 | 2970 | 5954 | 9909 |
| South America | 14 | 3528 | 556 | 0.158 | 2970 | 3698 | 5626 |
| North America | 3 | 756 | 321 | 0.425 | 854 | 3715 | 24346 |
| Central Asia | 8 | 2016 | 148 | 0.073 | 214 | 287 | 469 |
| East Asia | 7 | 1764 | 111 | 0.063 | 210 | 861 | 1296 |
| Central America & Caribbean | 21 | 5292 | 73 | 0.014 | 115 | 122 | 167 |
| Australasia & Oceania | 10 | 2520 | 68 | 0.027 | 108 | 74 | 103 |

## Top 20 Countries by Strict Attacks

| country_txt | region_txt | attack_months | attack_month_share | n_attacks_strict | fatalities_all | injuries_all | max_monthly_attacks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Iraq | Middle East & North Africa | 224 | 0.889 | 23827 | 80997 | 138162 | 417 |
| Afghanistan | South Asia | 240 | 0.952 | 15351 | 67120 | 66436 | 234 |
| Pakistan | South Asia | 248 | 0.984 | 11836 | 21980 | 38339 | 263 |
| India | South Asia | 252 | 1.0 | 9788 | 11016 | 19818 | 144 |
| Nigeria | Sub-Saharan Africa | 192 | 0.762 | 4841 | 28650 | 12196 | 85 |
| Philippines | Southeast Asia | 236 | 0.937 | 4738 | 5534 | 9173 | 81 |
| Yemen | Middle East & North Africa | 180 | 0.714 | 4196 | 12770 | 13364 | 83 |
| Thailand | Southeast Asia | 216 | 0.857 | 3286 | 2525 | 7393 | 73 |
| Somalia | Sub-Saharan Africa | 187 | 0.742 | 3261 | 12836 | 10774 | 88 |
| Colombia | South America | 235 | 0.933 | 2382 | 3329 | 4962 | 76 |
| Syria | Middle East & North Africa | 119 | 0.472 | 2071 | 18338 | 17038 | 64 |
| Libya | Middle East & North Africa | 103 | 0.409 | 1935 | 2917 | 3840 | 77 |
| Egypt | Middle East & North Africa | 121 | 0.48 | 1683 | 3516 | 4484 | 87 |
| Russia | Eastern Europe | 216 | 0.857 | 1661 | 3476 | 6307 | 44 |
| Nepal | South Asia | 197 | 0.782 | 1445 | 1914 | 2099 | 97 |
| Turkey | Middle East & North Africa | 200 | 0.794 | 1441 | 2647 | 7449 | 93 |
| Democratic Republic of the Congo | Sub-Saharan Africa | 158 | 0.627 | 1266 | 6978 | 1995 | 41 |
| United Kingdom | Western Europe | 206 | 0.817 | 1259 | 156 | 1456 | 29 |
| West Bank and Gaza Strip | Middle East & North Africa | 222 | 0.881 | 1206 | 1043 | 1975 | 52 |
| Israel | Middle East & North Africa | 206 | 0.817 | 1168 | 975 | 4662 | 135 |

## Yearly Summary

| year | n_attacks_strict | n_attacks_all | active_country_months | incidence_next_month | onset_next_3months | hard_onset_next_3months | fatalities_all | injuries_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2000 | 1461 | 1823 | 398 | 400 | 352 | 0 | 4394 | 5797 |
| 2001 | 1689 | 1912 | 379 | 363 | 281 | 0 | 7729 | 28187 |
| 2002 | 1150 | 1330 | 299 | 308 | 297 | 0 | 4797 | 7081 |
| 2003 | 1092 | 1280 | 290 | 275 | 217 | 0 | 3317 | 7384 |
| 2004 | 980 | 1164 | 234 | 242 | 179 | 1 | 5716 | 11976 |
| 2005 | 1634 | 2017 | 292 | 299 | 219 | 11 | 6343 | 12961 |
| 2006 | 2375 | 2757 | 291 | 281 | 214 | 11 | 9316 | 15470 |
| 2007 | 2844 | 3250 | 320 | 329 | 264 | 44 | 12825 | 22531 |
| 2008 | 4343 | 4801 | 421 | 423 | 316 | 44 | 9244 | 19168 |
| 2009 | 4564 | 4723 | 382 | 375 | 253 | 11 | 9277 | 19147 |
| 2010 | 4634 | 4827 | 347 | 348 | 249 | 9 | 7829 | 15953 |
| 2011 | 4662 | 5075 | 341 | 352 | 263 | 21 | 8246 | 14662 |
| 2012 | 6972 | 8525 | 451 | 460 | 312 | 13 | 15498 | 25451 |
| 2013 | 10034 | 12047 | 516 | 516 | 296 | 26 | 22279 | 37694 |
| 2014 | 13547 | 16960 | 577 | 575 | 278 | 23 | 44648 | 41143 |
| 2015 | 12385 | 15138 | 615 | 616 | 277 | 16 | 38993 | 44204 |
| 2016 | 11526 | 14051 | 628 | 631 | 321 | 22 | 35239 | 40576 |
| 2017 | 8994 | 11364 | 635 | 632 | 294 | 26 | 26897 | 25594 |
| 2018 | 7771 | 9853 | 609 | 609 | 310 | 10 | 23291 | 20609 |
| 2019 | 6769 | 8537 | 541 | 538 | 276 | 6 | 20412 | 18776 |
| 2020 | 6324 | 8438 | 553 | 508 | 233 | 14 | 22847 | 15466 |

## Figures Written

- `outputs/figures/active_countries_by_month.png`
- `outputs/figures/class_balance_attack_months.png`
- `outputs/figures/forecast_outcome_class_balance.png`
- `outputs/figures/three_month_onset_rates_by_region.png`
- `outputs/figures/region_attacks_by_month.png`
- `outputs/figures/region_total_attacks_strict.png`
- `outputs/figures/region_attack_month_share.png`
- `outputs/figures/positive_country_month_attack_distribution.png`
- `outputs/figures/country_zoom_attacks_by_month.png`
- `outputs/figures/top30_country_attack_month_heatmap.png`
