"""Feature engineering for the GTD terrorism onset model.

Adapted directly from Ben Seimon's prediction_solution.py.
Reads data/processed/gtd_panel_with_all_features.parquet and produces
data/processed/gtd_features.parquet with:
  - Rolling mean of attack history at windows [1, 3, 12, 36] months
  - months_since_last_attack (since variable)
  - ongoing indicator
  - 15 LDA topic stocks from conflictforecast.org (text features)
  - World Bank macro variables (gdp growth, unemployment, govt expenditure, inflation)
  - V-Dem democracy index
  - UCDP ongoing conflict dummy

NAs in macro/political variables are filled with column median.
Index: (iso3, period) where period is integer YYYYMM, matching Ben's convention.
"""

from __future__ import annotations
from functools import reduce
from pathlib import Path
import pandas as pd
import numpy as np

PANEL_PATH = Path("data/processed/gtd_panel_with_all_features.parquet")
OUTPUT_DIR = Path("data/processed")

TOPIC_COLS  = [f"stock_topic_{i}" for i in range(15)]
MACRO_COLS  = [
    "gdp_pc_growth", "unemployment_rate",
    "govt_expenditure_gdp", "inflation_cpi",
    "democracy_index", "ongoing_conflict",
]
WINDOWS    = [1, 3, 12, 36]
THRESHOLD  = 0


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    df = df.dropna(subset=["stock_topic_0"]).copy()
    df["period"] = df["month"].dt.year * 100 + df["month"].dt.month
    print(f"Loaded panel: {len(df):,} rows, {df['iso3'].nunique()} countries")
    print(f"Period range: {df['period'].min()} - {df['period'].max()}")
    return df


def make_rolling_mean(df: pd.DataFrame, y_col: str, windows: list[int]) -> pd.DataFrame:
    """Rolling mean of y_col over past N months — shift(1) so purely backward-looking."""
    out = df[["iso3", "period", y_col]].copy()
    for w in windows:
        out[f"{y_col}_rm{w}"] = (
            df.groupby("iso3")[y_col]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    return out.drop(columns=[y_col])


def make_since(df: pd.DataFrame, y_col: str, threshold: int) -> pd.DataFrame:
    """Months since last attack above threshold — captures peace spell duration."""
    out = df[["iso3", "period"]].copy()

    def months_since(series: pd.Series) -> pd.Series:
        result, count = [], 0
        for val in series:
            if val > threshold:
                count = 0
            else:
                count += 1
            result.append(count)
        return pd.Series(result, index=series.index)

    out[f"{y_col}_since_{threshold}"] = (
        df.groupby("iso3")[y_col].transform(months_since)
    )
    return out


def make_ongoing(df: pd.DataFrame, y_col: str, threshold: int) -> pd.DataFrame:
    """Binary: country currently above attack threshold."""
    out = df[["iso3", "period"]].copy()
    out[f"{y_col}_ongoing_{threshold}"] = (df[y_col] > threshold).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    y_col = "n_attacks_strict"

    rolling = make_rolling_mean(df, y_col, WINDOWS)
    since   = make_since(df, y_col, THRESHOLD)
    ongoing = make_ongoing(df, y_col, THRESHOLD)
    topics  = df[["iso3", "period"] + TOPIC_COLS]

    frames = [rolling, since, ongoing, topics]
    X = reduce(
        lambda l, r: l.merge(r, on=["iso3", "period"], how="inner"),
        frames
    )
    X = X.set_index(["iso3", "period"]).sort_index()

    # Add macro/political features — fill NAs with column median
    macro = df.set_index(["iso3", "period"])[MACRO_COLS].copy()
    for col in MACRO_COLS:
        n_miss = macro[col].isna().sum()
        if n_miss > 0:
            fill_val = macro[col].median()
            macro[col] = macro[col].fillna(fill_val)
            print(f"  {col}: filled {n_miss:,} NAs with median ({fill_val:.3f})")
    X = X.merge(macro, left_index=True, right_index=True, how="left")

    print(f"\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    na_count = X.isna().sum().sum()
    if na_count > 0:
        print(f"Warning: {na_count} remaining NAs — filling with 0.")
        X = X.fillna(0)
    else:
        print("No NAs in feature matrix ✓")

    print(f"\nFeature columns ({len(X.columns)}):")
    for col in X.columns:
        print(f"  {col}")

    return X


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df.set_index(["iso3", "period"]).sort_index()

    inc = base[["incidence_next_month_t"]].copy()
    inc.columns = ["inc_target"]

    ons = base[["onset_next_3months_t"]].copy()
    ons.columns = ["ons_target"]

    print(f"\nIncidence target: {inc['inc_target'].notna().sum():,} valid, "
          f"{int(inc['inc_target'].sum()):,} positive "
          f"({inc['inc_target'].mean():.1%} rate)")
    print(f"Onset target:     {ons['ons_target'].notna().sum():,} valid, "
          f"{int(ons['ons_target'].sum()):,} positive "
          f"({ons['ons_target'].mean():.1%} rate)")

    return inc, ons


def main() -> None:
    print("=== Step 4: Feature engineering ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_panel(PANEL_PATH)
    X  = build_features(df)
    inc_df, ons_df = build_targets(df)

    X.to_parquet(OUTPUT_DIR / "gtd_features.parquet")
    X.to_csv(OUTPUT_DIR / "gtd_features.csv")
    inc_df.to_parquet(OUTPUT_DIR / "gtd_target_inc.parquet")
    ons_df.to_parquet(OUTPUT_DIR / "gtd_target_ons.parquet")

    print(f"\nSaved: data/processed/gtd_features.parquet")
    print(f"Saved: data/processed/gtd_target_inc.parquet")
    print(f"Saved: data/processed/gtd_target_ons.parquet")
    print("\nDone. Next step: src/05_rolling_forecast.py")


if __name__ == "__main__":
    main()
