"""Merge macro-economic, political, and conflict features onto the GTD panel.

Reads data/processed/gtd_panel_with_topics.parquet (output of step 03).

Adds:
  1. World Bank macro variables (annual -> forward-filled to monthly):
     - GDP per capita growth
     - Unemployment rate
     - Government expenditure % of GDP
     - Inflation (CPI)

  2. V-Dem political rights score (annual -> forward-filled to monthly):
     - v2x_polyarchy: Electoral Democracy Index (0-1)

  3. UCDP ongoing conflict dummy (monthly, already in ucdp.csv):
     - ongoing_conflict: 1 if fatalities_UCDP > 0 in that country-month

All annual series are forward-filled to monthly frequency within each country.
This is standard practice — we assume the annual value holds until the next
observation arrives. This is documented explicitly in the methodology section.

Output: data/processed/gtd_panel_with_all_features.parquet / .csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import wbgapi as wb

# ── Paths ─────────────────────────────────────────────────────────────────────
PANEL_PATH  = Path("data/processed/gtd_panel_with_topics.parquet")
UCDP_PATH   = Path("data/raw/ucdp.csv")
VDEM_PATH   = Path("data/raw/V-Dem-CY-Core-v16.csv")
OUTPUT_DIR  = Path("data/processed")
DIAG_DIR    = Path("outputs/diagnostics")

# ── World Bank indicator codes ────────────────────────────────────────────────
WB_INDICATORS = {
    "NY.GDP.PCAP.KD.ZG": "gdp_pc_growth",
    "SL.UEM.TOTL.ZS":    "unemployment_rate",
    "GC.XPN.TOTL.GD.ZS": "govt_expenditure_gdp",
    "FP.CPI.TOTL.ZG":    "inflation_cpi",
}

VDEM_COL = "v2x_polyarchy"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    print(f"Panel loaded: {len(df):,} rows, {df['iso3'].nunique()} countries")
    return df


def annual_to_monthly(
    annual: pd.DataFrame,
    id_col: str,
    year_col: str,
    value_cols: list[str],
    months: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Expand annual country-year data to monthly frequency by forward-filling.

    For each country, we assign the annual value to January of that year,
    then forward-fill through the remaining months. This is the standard
    approach for merging annual economic data onto monthly panels.
    """
    # Create Jan-01 date for each year
    annual = annual.copy()
    annual["month"] = pd.to_datetime(
        annual[year_col].astype(int).astype(str) + "-01-01"
    )

    # Build complete monthly spine per country
    countries = annual[id_col].unique()
    spine = pd.MultiIndex.from_product(
        [countries, months], names=[id_col, "month"]
    ).to_frame(index=False)

    # Merge annual onto spine (annual value goes into its January row)
    merged = spine.merge(
        annual[[id_col, "month"] + value_cols],
        on=[id_col, "month"],
        how="left"
    )

    # Forward-fill within each country
    merged = merged.sort_values([id_col, "month"])
    merged[value_cols] = (
        merged.groupby(id_col)[value_cols]
        .transform(lambda x: x.ffill())
    )
    return merged


# ── 1. World Bank ─────────────────────────────────────────────────────────────

def fetch_world_bank(
    indicators: dict[str, str],
    start_year: int = 2009,
    end_year: int = 2020,
) -> pd.DataFrame:
    """Pull World Bank indicators and return long-format country-year DataFrame."""
    print("\nFetching World Bank data...")
    frames = []
    for code, name in indicators.items():
        try:
            raw = wb.data.DataFrame(
                code,
                time=range(start_year, end_year + 1),
            )
            # wbgapi returns wide format: index=economy, columns=YR2010 etc.
            raw = raw.reset_index()
            raw = raw.melt(
                id_vars=["economy"],
                var_name="year_str",
                value_name=name,
            )
            raw["year"] = raw["year_str"].str.replace("YR", "").astype(int)
            raw = raw.rename(columns={"economy": "iso3"})[["iso3", "year", name]]
            frames.append(raw)
            print(f"  {name}: {raw[name].notna().sum():,} non-missing observations")
        except Exception as e:
            print(f"  Warning: could not fetch {code} ({name}): {e}")

    if not frames:
        raise RuntimeError("No World Bank data fetched. Check internet connection.")

    from functools import reduce
    wb_df = reduce(
        lambda l, r: l.merge(r, on=["iso3", "year"], how="outer"),
        frames
    )
    print(f"World Bank panel: {len(wb_df):,} country-year rows")
    return wb_df


# ── 2. V-Dem ──────────────────────────────────────────────────────────────────

def load_vdem(path: Path) -> pd.DataFrame:
    """Load V-Dem Core CSV and extract polyarchy index."""
    print("\nLoading V-Dem data...")

    # V-Dem CSV is large — read only needed columns
    # Key columns: country_text_id (ISO3-like), year, v2x_polyarchy
    try:
        vdem = pd.read_csv(
            path,
            usecols=["country_text_id", "year", VDEM_COL],
            low_memory=False,
        )
    except ValueError:
        # If usecols fails, load all and filter
        vdem = pd.read_csv(path, low_memory=False)
        vdem = vdem[["country_text_id", "year", VDEM_COL]]

    vdem = vdem.rename(columns={
        "country_text_id": "iso3",
        VDEM_COL: "democracy_index",
    })
    vdem = vdem[vdem["year"].between(2009, 2020)].copy()
    vdem["year"] = vdem["year"].astype(int)

    print(f"V-Dem rows (2009-2020): {len(vdem):,}")
    print(f"  Countries: {vdem['iso3'].nunique()}")
    print(f"  Non-missing democracy_index: {vdem['democracy_index'].notna().sum():,}")
    return vdem


# ── 3. UCDP ───────────────────────────────────────────────────────────────────

def load_ucdp(path: Path) -> pd.DataFrame:
    """Load UCDP and create monthly ongoing conflict dummy."""
    print("\nLoading UCDP data...")
    ucdp = pd.read_csv(path, index_col=0)

    # Convert period YYYYMM to month date
    ucdp["month"] = pd.to_datetime(
        ucdp["period"].astype(str), format="%Y%m"
    )
    ucdp["ongoing_conflict"] = (ucdp["fatalities_UCDP"] > 0).astype(int)
    ucdp = ucdp[["isocode", "month", "ongoing_conflict"]].rename(
        columns={"isocode": "iso3"}
    )
    print(f"UCDP rows: {len(ucdp):,}, countries: {ucdp['iso3'].nunique()}")
    return ucdp


# ── Main merge ────────────────────────────────────────────────────────────────

def merge_all(panel: pd.DataFrame) -> pd.DataFrame:
    months = pd.date_range("2010-01-01", "2020-12-01", freq="MS")

    # ── World Bank ────────────────────────────────────────────────────────────
    wb_df = fetch_world_bank(WB_INDICATORS)
    wb_monthly = annual_to_monthly(
        annual=wb_df,
        id_col="iso3",
        year_col="year",
        value_cols=list(WB_INDICATORS.values()),
        months=months,
    )

    panel = panel.merge(wb_monthly, on=["iso3", "month"], how="left")
    for col in WB_INDICATORS.values():
        n_miss = panel[col].isna().sum()
        pct = 100 * n_miss / len(panel)
        print(f"  {col}: {n_miss:,} missing ({pct:.1f}%)")

    # ── V-Dem ─────────────────────────────────────────────────────────────────
    vdem = load_vdem(VDEM_PATH)
    vdem_monthly = annual_to_monthly(
        annual=vdem,
        id_col="iso3",
        year_col="year",
        value_cols=["democracy_index"],
        months=months,
    )
    panel = panel.merge(vdem_monthly, on=["iso3", "month"], how="left")
    n_miss = panel["democracy_index"].isna().sum()
    print(f"  democracy_index: {n_miss:,} missing ({100*n_miss/len(panel):.1f}%)")

    # ── UCDP ──────────────────────────────────────────────────────────────────
    ucdp = load_ucdp(UCDP_PATH)
    panel = panel.merge(ucdp, on=["iso3", "month"], how="left")
    panel["ongoing_conflict"] = panel["ongoing_conflict"].fillna(0).astype(int)
    print(f"  ongoing_conflict: {panel['ongoing_conflict'].sum():,} conflict country-months")

    return panel


def save_outputs(panel: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    out_parquet = OUTPUT_DIR / "gtd_panel_with_all_features.parquet"
    out_csv     = OUTPUT_DIR / "gtd_panel_with_all_features.csv"
    panel.to_parquet(out_parquet, index=False)
    panel.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_csv}")
    print(f"\nFinal panel shape: {panel.shape}")
    print("\nAll columns:")
    for col in panel.columns:
        print(f"  {col}")

    # Coverage summary
    feature_cols = list(WB_INDICATORS.values()) + ["democracy_index", "ongoing_conflict"]
    coverage = pd.DataFrame([{
        "feature": col,
        "non_missing": panel[col].notna().sum(),
        "pct_coverage": f"{100*panel[col].notna().mean():.1f}%"
    } for col in feature_cols])
    coverage.to_csv(DIAG_DIR / "macro_feature_coverage.csv", index=False)
    print("\nCoverage summary saved to outputs/diagnostics/macro_feature_coverage.csv")


def main() -> None:
    print("=== Step 3b: Merge macro + political + conflict features ===\n")

    panel = load_panel(PANEL_PATH)
    panel = merge_all(panel)
    save_outputs(panel)

    print("\nDone. Next step: src/04_engineer_features.py")
    print("Update PANEL_PATH in 04 to: data/processed/gtd_panel_with_all_features.parquet")


if __name__ == "__main__":
    main()
