# HW1: Terrorism Onset Forecasting
Forecasting and Nowcasting with Text as Data — BSE Term 3 2026
Group: Corneel Moons, Eleanor Pogrund, Lucas Salamanca

## Data sources
- GTD: place globalterrorismdb_0522dist.xlsx in data/raw/
- topics.csv: from conflictforecast.org, place in data/raw/
- ucdp.csv: from course materials, place in data/raw/
- V-Dem-CY-Core-v16.csv: from v-dem.net, place in data/raw/

## How to run (from HW1_terrorism/ directory)
1. python src/01_build_gtd_country_month_panel.py
2. python src/02_describe_gtd_country_month_panel.py
3. python src/03_merge_text_features.py
4. python src/03b_merge_macro_features.py
5. python src/04_engineer_features.py
6. python src/05_rolling_forecast.py
7. Open and run src/06_evaluate.ipynb

## Dependencies
panelsplit, scikit-learn, pandas, numpy, matplotlib, wbgapi, pyarrow
Install with: uv sync