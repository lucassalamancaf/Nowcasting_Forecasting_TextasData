# GTD Country-Month Panel

This project prepares the first data-engineering step for the terrorism
nowcasting/forecasting homework. The script converts event-level Global
Terrorism Database (GTD) records into a clean country-month panel.

It does not create onset variables or forecasting models.

## Raw Data

GTD is distributed by START and may require accepting terms of use. Because of
that, this project expects the raw file to be placed manually in:

```text
data/raw/
```

Accepted formats are:

- `.xlsx`
- `.xls`
- `.csv`
- `.zip` containing one of the above

The script searches `data/raw/` and uses the most GTD-like file it finds.

## Run

From this `HW1_terrorism` directory:

```bash
python -m pip install -r requirements.txt
```

```bash
python src/01_build_gtd_country_month_panel.py --start 2010-01 --end 2020-12
```

The default period is already `2010-01` through `2020-12`, so this also works:

```bash
python src/01_build_gtd_country_month_panel.py
```

To create initial descriptive tables and plots after the panel has been built:

```bash
python src/02_describe_gtd_country_month_panel.py
```

To compare the rolling forecasts with and without the text-topic features:

```bash
python src/07_compare_text_ablation.py
```

After the ablation predictions have been created once, rerun only the metrics
and plots with:

```bash
python src/07_compare_text_ablation.py --reuse-predictions
```

## Outputs

The panel-building script creates:

```text
data/interim/gtd_events_clean.parquet
data/interim/gtd_events_clean_sample.csv
data/processed/gtd_country_month_panel.parquet
data/processed/gtd_country_month_panel.csv
outputs/diagnostics/gtd_panel_diagnostics.md
outputs/figures/global_attacks_by_month.png
outputs/figures/top20_countries_attacks_strict.png
```

The descriptive script creates:

```text
data/processed/gtd_country_month_panel_with_outcomes.parquet
data/processed/gtd_country_month_panel_with_outcomes.csv
outputs/diagnostics/gtd_initial_data_description.md
outputs/diagnostics/region_summary.csv
outputs/diagnostics/country_summary.csv
outputs/diagnostics/yearly_summary.csv
outputs/diagnostics/monthly_global.csv
outputs/diagnostics/panel_balance_by_country.csv
outputs/diagnostics/outcome_class_balance.csv
outputs/diagnostics/forecasting_outcome_class_balance.csv
outputs/diagnostics/forecasting_outcome_region_summary.csv
outputs/figures/active_countries_by_month.png
outputs/figures/class_balance_attack_months.png
outputs/figures/forecast_outcome_class_balance.png
outputs/figures/three_month_onset_rates_by_region.png
outputs/figures/region_attacks_by_month.png
outputs/figures/region_total_attacks_strict.png
outputs/figures/region_attack_month_share.png
outputs/figures/positive_country_month_attack_distribution.png
outputs/figures/country_zoom_attacks_by_month.png
outputs/figures/top30_country_attack_month_heatmap.png
```

The text-ablation script creates:

```text
data/processed/gtd_text_ablation_preds_inc.csv
data/processed/gtd_text_ablation_preds_ons.csv
outputs/diagnostics/text_ablation_performance_summary.csv
outputs/diagnostics/text_ablation_key_numbers.txt
outputs/figures/text_ablation_roc_pr_curves.png
```

## Panel Columns

The final panel includes one row per GTD country code and month in the selected
period:

- `country`, `country_txt`, `region`, `region_txt`: GTD country and region
  metadata.
- `month`: month-start date, for example `2010-01-01`.
- `n_attacks_all`: count of all GTD events.
- `n_attacks_strict`: count of events where `doubtterr != 1`; missing
  `doubtterr` values are treated as not doubtful.
- `n_success_all`, `n_success_strict`: successful attacks.
- `n_suicide_all`, `n_suicide_strict`: suicide attacks.
- `fatalities_all`, `fatalities_strict`: sums of `nkill`.
- `injuries_all`, `injuries_strict`: sums of `nwound`.
- `n_fatal_attacks_all`, `n_fatal_attacks_strict`: attacks with `nkill > 0`.
- `attack_month_t`: equals 1 when `n_attacks_strict > 0` in the current
  country-month, and 0 otherwise. This is not an onset variable.

The outcome panel additionally includes:

- `incidence_next_month_t`: equals 1 if a strict attack occurs in `t+1`.
- `onset_next_month_t`: equals 1 if there is no strict attack in `t` and a
  strict attack occurs in `t+1`; current attack months are set to missing.
- `onset_next_3months_t`: equals 1 if there is no strict attack in `t` and a
  strict attack occurs in `t+1`, `t+2`, or `t+3`; current attack months are set
  to missing.
- `hard_onset_next_3months_t`: equals 1 if the previous 60 months through `t`
  have no strict attacks and a strict attack occurs in `t+1`, `t+2`, or `t+3`.
  Rows without 60 months of history or outside the hard-onset risk set are set
  to missing.

## Notes

- GTD is event-level data. This script converts it into country-month format.
- Months with no attacks are explicitly included as zeros.
- Rows with unknown month, `imonth == 0`, are dropped at this stage and counted
  in diagnostics.
- Missing fatalities and injuries are set to zero only for aggregation. This is
  a modeling choice that should be revisited before final analysis.
- If a country code appears with multiple country names or regions, the
  diagnostics report flags it. The final panel uses modal metadata for the
  country code.
- Onset labels are intentionally not created in this step.
