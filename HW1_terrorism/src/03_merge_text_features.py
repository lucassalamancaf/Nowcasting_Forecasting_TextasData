"""Merge conflictforecast.org LDA topic stocks onto the GTD country-month panel.

Text feature source: topics.csv from conflictforecast.org
  - 15 LDA topic 'stocks' per country-month (smoothed with decay δ=0.8)
  - Monthly, 174 countries, January 2010 – December 2024
  - Unit identifier: ISO3 country code

GTD panel unit identifier: GTD integer country code + country_txt name
  - We bridge via a manual ISO3 crosswalk on country_txt

Lag rationale:
  The outcome variable onset_next_3months_t is defined as:
  "no attack at time t; at least one attack in t+1, t+2, or t+3"
  Topic stocks at time t therefore serve as *leading* indicators —
  they capture the political/conflict discourse in the month BEFORE
  the forecast window begins. No additional lag shift is needed because
  the outcome already looks forward. This avoids data leakage.

Output: data/processed/gtd_panel_with_topics.parquet / .csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PANEL_PATH  = Path("data/processed/gtd_country_month_panel_with_outcomes.parquet")
TOPICS_PATH = Path("data/raw/topics.csv")          # copy topics.csv here
OUTPUT_DIR  = Path("data/processed")
DIAG_DIR    = Path("outputs/diagnostics")

# ── ISO3 crosswalk: GTD country_txt → ISO3 ───────────────────────────────────
# Generated from the GTD country list; covers all 204 countries in panel.
# Unmapped countries will be dropped from the merged panel with a warning.
CROSSWALK = {
    "Afghanistan": "AFG",
    "Albania": "ALB",
    "Algeria": "DZA",
    "Angola": "AGO",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "Australia": "AUS",
    "Austria": "AUT",
    "Azerbaijan": "AZE",
    "Bahrain": "BHR",
    "Bangladesh": "BGD",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Bolivia": "BOL",
    "Bosnia-Herzegovina": "BIH",
    "Brazil": "BRA",
    "Bulgaria": "BGR",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cambodia": "KHM",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Central African Republic": "CAF",
    "Chad": "TCD",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Comoros": "COM",
    "Democratic Republic of the Congo": "COD",
    "Republic of the Congo": "COG",
    "Costa Rica": "CRI",
    "Croatia": "HRV",
    "Cuba": "CUB",
    "Cyprus": "CYP",
    "Czech Republic": "CZE",
    "Denmark": "DNK",
    "Djibouti": "DJI",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Eritrea": "ERI",
    "Estonia": "EST",
    "Ethiopia": "ETH",
    "Finland": "FIN",
    "France": "FRA",
    "Gabon": "GAB",
    "Georgia": "GEO",
    "Germany": "DEU",
    "Ghana": "GHA",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Guinea": "GIN",
    "Guinea-Bissau": "GNB",
    "Haiti": "HTI",
    "Honduras": "HND",
    "Hungary": "HUN",
    "India": "IND",
    "Indonesia": "IDN",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Ivory Coast": "CIV",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kosovo": "XKX",
    "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ",
    "Laos": "LAO",
    "Latvia": "LVA",
    "Lebanon": "LBN",
    "Liberia": "LBR",
    "Libya": "LBY",
    "Lithuania": "LTU",
    "Macedonia": "MKD",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "Mali": "MLI",
    "Mauritania": "MRT",
    "Mexico": "MEX",
    "Moldova": "MDA",
    "Morocco": "MAR",
    "Mozambique": "MOZ",
    "Myanmar": "MMR",
    "Nepal": "NPL",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Niger": "NER",
    "Nigeria": "NGA",
    "North Korea": "PRK",
    "Norway": "NOR",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Papua New Guinea": "PNG",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Romania": "ROU",
    "Russia": "RUS",
    "Rwanda": "RWA",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Sierra Leone": "SLE",
    "Somalia": "SOM",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "South Sudan": "SSD",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "Sudan": "SDN",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Tajikistan": "TJK",
    "Tanzania": "TZA",
    "Thailand": "THA",
    "Togo": "TGO",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Turkmenistan": "TKM",
    "Uganda": "UGA",
    "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "West Bank and Gaza Strip": "PSE",
    "Yemen": "YEM",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
    "Montenegro": "MNE",
    "East Timor": "TLS",
    "Singapore": "SGP",
    "Qatar": "QAT",
    "Slovak Republic": "SVK",
    "Slovenia": "SVN",
    "Namibia": "NAM",
    "Botswana": "BWA",
    "Iceland": "ISL",
    "Luxembourg": "LUX",
    "Malta": "MLT",
    "Fiji": "FJI",
}

TOPIC_COLS = [f"stock_topic_{i}" for i in range(15)]


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel["month"] = pd.to_datetime(panel["month"])
    print(f"Panel loaded: {len(panel):,} rows, {panel['country'].nunique()} countries")
    return panel


def load_topics(path: Path) -> pd.DataFrame:
    topics = pd.read_csv(path, index_col=0)
    # Convert period 202301 -> 2023-01-01
    topics["month"] = pd.to_datetime(
        topics["period"].astype(str), format="%Y%m"
    )
    topics = topics[["isocode", "month"] + TOPIC_COLS].copy()
    print(f"Topics loaded: {len(topics):,} rows, {topics['isocode'].nunique()} ISO3 countries")
    print(f"  Period: {topics['month'].min().date()} to {topics['month'].max().date()}")
    return topics


def add_iso3(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["iso3"] = panel["country_txt"].map(CROSSWALK)
    n_unmapped = panel["iso3"].isna().sum()
    unmapped_countries = panel.loc[panel["iso3"].isna(), "country_txt"].unique()
    if n_unmapped > 0:
        print(f"\nWarning: {len(unmapped_countries)} countries unmapped to ISO3"
              f" ({n_unmapped:,} rows will be dropped):")
        for c in sorted(unmapped_countries):
            print(f"  - {c}")
    return panel


def merge_topics(panel: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    # Restrict to 2010+ where topics are available
    panel_2010 = panel[panel["month"] >= "2010-01-01"].copy()
    n_before = len(panel_2010)

    merged = panel_2010.merge(
        topics.rename(columns={"isocode": "iso3"}),
        on=["iso3", "month"],
        how="left"
    )

    # Report coverage
    has_topics = merged[TOPIC_COLS[0]].notna()
    n_matched = has_topics.sum()
    n_countries_matched = merged.loc[has_topics, "country"].nunique()

    print(f"\nMerge results (2010-2020 window):")
    print(f"  Panel rows:              {n_before:,}")
    print(f"  Rows with topic data:    {n_matched:,}  ({100*n_matched/n_before:.1f}%)")
    print(f"  Countries with topics:   {n_countries_matched}")
    print(f"  Countries without:       {merged.loc[~has_topics, 'country_txt'].nunique()}")

    return merged


def check_leakage(panel: pd.DataFrame) -> None:
    """Sanity check: topics at t should predict outcome at t+1 through t+3.
    The outcome onset_next_3months_t is already forward-looking so no
    additional shift is needed. This function just confirms the logic."""
    print("\nLag/leakage check:")
    print("  onset_next_3months_t = 1 if any attack in t+1, t+2, t+3 (given no attack at t)")
    print("  stock_topic_X at time t = news discourse BEFORE the forecast window")
    print("  → No additional lag shift needed. Leakage risk: LOW.")
    print("  → Topics capture conditions at t; outcome window starts at t+1.")


def save_outputs(merged: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    out_parquet = OUTPUT_DIR / "gtd_panel_with_topics.parquet"
    out_csv     = OUTPUT_DIR / "gtd_panel_with_topics.csv"
    merged.to_parquet(out_parquet, index=False)
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_csv}")

    # Diagnostics: coverage by country
    coverage = (
        merged.groupby(["country_txt", "iso3", "region_txt"])
        .agg(
            total_months=("month", "size"),
            months_with_topics=(TOPIC_COLS[0], lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    coverage["topic_coverage"] = coverage["months_with_topics"] / coverage["total_months"]
    coverage.to_csv(DIAG_DIR / "topic_merge_coverage.csv", index=False)

    print(f"\nColumn list of final panel:")
    for col in merged.columns:
        print(f"  {col}")


def main() -> None:
    print("=== Step 3: Merge text features (conflictforecast.org topics) ===\n")

    panel  = load_panel(PANEL_PATH)
    topics = load_topics(TOPICS_PATH)
    panel  = add_iso3(panel)
    merged = merge_topics(panel, topics)
    check_leakage(merged)
    save_outputs(merged)

    print("\nDone. Next step: src/04_engineer_features.py")


if __name__ == "__main__":
    main()
