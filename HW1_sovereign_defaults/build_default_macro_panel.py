"""
Build the first-stage sovereign default and macro-financial country-year panel.

Install dependencies if needed:
    python -m pip install pandas numpy openpyxl wbgapi country_converter

Outputs are written to the repository-level output/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import country_converter as coco
import numpy as np
import pandas as pd
import wbgapi as wb


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

DEFAULT_DB_URL = (
    "https://www.bankofcanada.ca/wp-content/uploads/2025/10/"
    "BoC-BoE-Database-2025.xlsx"
)
DEFAULT_DB_PATH = DATA_DIR / "BoC-BoE-Database-2025.xlsx"

START_YEAR = 1960
END_YEAR = 2024

WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "DT.DOD.DECT.GN.ZS": "external_debt_gni",
    "DT.TDS.DECT.EX.ZS": "debt_service_exports",
    "FI.RES.TOTL.CD": "reserves_usd",
    "PA.NUS.FCRF": "exchange_rate",
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "GC.DOD.TOTL.GD.ZS": "central_gov_debt_gdp",
}
MACRO_VARIABLES = list(WB_INDICATORS.values())

MANUAL_COUNTRY_ISO3 = {
    "Bahamas": "BHS",
    "Bosnia & Herzegovina": "BIH",
    "Brunei": "BRN",
    "Cabo Verde": "CPV",
    "Congo": "COG",
    "Congo, Democratic Republic of": "COD",
    "Congo, Republic of": "COG",
    "Cote d'Ivoire": "CIV",
    "Côte d’Ivoire": "CIV",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Egypt": "EGY",
    "Eswatini": "SWZ",
    "Gambia": "GMB",
    "Iran": "IRN",
    "Ivory Coast": "CIV",
    "Korea, North": "PRK",
    "Korea, Rep.": "KOR",
    "Korea, South": "KOR",
    "Kyrgyz Republic": "KGZ",
    "Laos": "LAO",
    "Macedonia": "MKD",
    "Micronesia": "FSM",
    "Netherlands Antilles": "ANT",
    "North Korea": "PRK",
    "Republic of Congo": "COG",
    "Russia": "RUS",
    "Russian Federation": "RUS",
    "São Tomé and Príncipe": "STP",
    "Sao Tome and Principe": "STP",
    "Slovak Republic": "SVK",
    "Slovakia": "SVK",
    "South Korea": "KOR",
    "St. Kitts and Nevis": "KNA",
    "St. Lucia": "LCA",
    "St. Vincent & the Grenadines": "VCT",
    "Syria": "SYR",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "USSR/Russian Federation": "RUS",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "West Bank & Gaza": "PSE",
    "Yemen": "YEM",
    "Yugoslavia": "YUG",
}


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_default_database(path: Path = DEFAULT_DB_PATH) -> Path:
    """Download the BoC-BoE workbook once and reuse the local copy."""
    if not path.exists():
        print(f"Downloading sovereign default workbook to {path}")
        urlretrieve(DEFAULT_DB_URL, path)
    else:
        print(f"Using cached sovereign default workbook: {path}")
    return path


def select_default_sheet(sheet_names: Iterable[str]) -> str:
    """Prefer the current progress sheet, but fall back to likely alternatives."""
    names = list(sheet_names)
    if "2025 in Progress" in names:
        return "2025 in Progress"

    progress_matches = [s for s in names if "progress" in s.lower()]
    if progress_matches:
        return progress_matches[0]

    non_metadata = [s for s in names if "debt" not in s.lower()]
    if non_metadata:
        return non_metadata[0]

    return names[0]


def find_year_columns(raw: pd.DataFrame) -> dict[int, int]:
    """Find the row with annual year headers and return {column_index: year}."""
    best_row = None
    best_years: dict[int, int] = {}

    for row_idx in raw.index:
        years: dict[int, int] = {}
        for col_idx, value in raw.loc[row_idx].items():
            if pd.isna(value):
                continue
            try:
                year = int(value)
            except (TypeError, ValueError):
                continue
            if START_YEAR <= year <= END_YEAR + 1:
                years[int(col_idx)] = year

        if len(years) > len(best_years):
            best_row = row_idx
            best_years = years

    if len(best_years) < 10:
        raise ValueError(
            "Could not locate annual year columns. "
            f"Best row was {best_row} with {len(best_years)} candidate years."
        )

    return best_years


def find_default_section_start(raw: pd.DataFrame) -> int:
    """Locate the country-level sovereign debt in default section."""
    pattern = "sovereign debt in default"
    section_rows = raw.apply(
        lambda col: col.astype(str).str.contains(pattern, case=False, na=False)
    ).any(axis=1)
    matches = raw.index[section_rows].tolist()
    if matches:
        return int(matches[0])

    issuer_rows = raw.apply(
        lambda col: col.astype(str).str.contains("by issuer", case=False, na=False)
    ).any(axis=1)
    matches = raw.index[issuer_rows].tolist()
    if matches:
        return int(matches[0])

    raise ValueError("Could not locate the sovereign debt in default country section.")


def print_default_parse_debug(raw: pd.DataFrame, sheet_names: list[str]) -> None:
    print("Available workbook sheets:")
    for sheet in sheet_names:
        print(f"  - {sheet}")
    print("\nWorkbook preview:")
    print(raw.iloc[:15, :12].to_string(index=True, header=True))


def clean_country_name(name: object) -> str:
    return str(name).replace("\xa0", " ").strip()


def map_default_countries_to_iso3(default_countries: Iterable[str]) -> pd.DataFrame:
    """Map default-database country names to ISO3 and report unmatched names."""
    names = pd.Series(sorted(set(default_countries)), name="country")
    manual = names.map(MANUAL_COUNTRY_ISO3)

    auto_candidates = names.where(manual.isna(), np.nan)
    auto = pd.Series(index=names.index, dtype="object")
    to_convert = auto_candidates.dropna()
    if not to_convert.empty:
        converted = coco.convert(
            names=to_convert.tolist(),
            to="ISO3",
            not_found=None,
        )
        auto.loc[to_convert.index] = converted

    iso3 = manual.combine_first(auto)
    iso3 = iso3.where(iso3.astype(str).str.fullmatch(r"[A-Z]{3}"), np.nan)

    matched = pd.DataFrame({"country": names, "iso3": iso3})
    unmatched = matched[matched["iso3"].isna()].copy()
    unmatched.to_csv(OUTPUT_DIR / "unmatched_default_countries.csv", index=False)

    if not unmatched.empty:
        print("Unmatched default-database countries:")
        print(unmatched.to_string(index=False))

    return matched


def parse_default_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the BoC-BoE country-year default stock panel."""
    xl = pd.ExcelFile(path)
    print("Available workbook sheets:", xl.sheet_names)
    sheet_name = select_default_sheet(xl.sheet_names)
    print(f"Reading default workbook sheet: {sheet_name}")

    raw = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    try:
        year_cols = find_year_columns(raw)
        section_start = find_default_section_start(raw)

        # Country rows in this workbook have a numeric sequence ID in column 1
        # and the country name in column 2. Restricting to the default section
        # avoids summary and footnote rows.
        country_rows = raw.index[
            (raw.index > section_start)
            & pd.to_numeric(raw.iloc[:, 1], errors="coerce").notna()
            & raw.iloc[:, 2].notna()
        ].tolist()
        if not country_rows:
            raise ValueError("No country rows found after the default section label.")

        countries = raw.loc[country_rows, [1, 2]].rename(
            columns={1: "country_order", 2: "country"}
        )
        countries["country"] = countries["country"].map(clean_country_name)

        value_cols = sorted(year_cols)
        wide = raw.loc[country_rows, value_cols].copy()
        wide.columns = [year_cols[col] for col in value_cols]
        wide.insert(0, "country", countries["country"].to_numpy())

        panel = wide.melt(
            id_vars="country",
            var_name="year",
            value_name="debt_default_usd_mil",
        )
        panel["year"] = panel["year"].astype(int)
        panel["debt_default_usd_mil"] = pd.to_numeric(
            panel["debt_default_usd_mil"], errors="coerce"
        )

        country_map = map_default_countries_to_iso3(panel["country"])
        panel = panel.merge(country_map, on="country", how="left")

        panel["default_state"] = (
            panel["debt_default_usd_mil"].fillna(0).gt(0).astype(int)
        )
        panel = add_default_targets(panel)
        panel = panel[
            [
                "country",
                "iso3",
                "year",
                "debt_default_usd_mil",
                "default_state",
                "default_lag1",
                "default_lag2",
                "default_lead1",
                "onset_next_year",
                "hard_onset_next_year",
                "incidence_next_year",
                "continuation_next_year",
            ]
        ]

        unmatched = country_map[country_map["iso3"].isna()].copy()
        return panel, unmatched
    except Exception:
        print_default_parse_debug(raw, xl.sheet_names)
        raise


def add_default_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Create lag, lead, onset, incidence, and continuation outcomes."""
    out = panel.sort_values(["country", "year"]).copy()
    grouped = out.groupby("country", sort=False)["default_state"]

    out["default_lag1"] = grouped.shift(1)
    out["default_lag2"] = grouped.shift(2)
    out["default_lead1"] = grouped.shift(-1)

    out["onset_next_year"] = (
        out["default_state"].eq(0) & out["default_lead1"].eq(1)
    ).astype(int)
    out["hard_onset_next_year"] = (
        out["default_lag1"].eq(0)
        & out["default_state"].eq(0)
        & out["default_lead1"].eq(1)
    ).astype(int)
    out["incidence_next_year"] = out["default_lead1"].eq(1).astype(int)
    out["continuation_next_year"] = (
        out["default_state"].eq(1) & out["default_lead1"].eq(1)
    ).astype(int)

    return out.sort_values(["country", "year"]).reset_index(drop=True)


def country_economies_from_wb() -> tuple[list[str], dict[str, str]]:
    """Return World Bank country economy IDs, excluding aggregates/regions."""
    economies = list(wb.economy.list())
    countries = [e for e in economies if not e.get("aggregate", False)]
    iso3_codes = [e["id"] for e in countries]
    names = {e["id"]: e["value"] for e in countries}
    return iso3_codes, names


def download_world_bank_panel() -> pd.DataFrame:
    """Download and reshape annual World Bank macro-financial indicators."""
    economies, country_names = country_economies_from_wb()
    print(
        "Downloading World Bank indicators for "
        f"{len(economies)} country economies, {START_YEAR}-{END_YEAR}",
        flush=True,
    )

    long_frames = []
    for code, variable in WB_INDICATORS.items():
        print(f"  - downloading {variable} ({code})", flush=True)
        wide = wb.data.DataFrame(
            code,
            economy=economies,
            time=range(START_YEAR, END_YEAR + 1),
            numericTimeKeys=True,
            labels=True,
        )
        id_vars = ["economy", "Country"]
        indicator_long = wide.reset_index().melt(
            id_vars=id_vars,
            var_name="year",
            value_name=variable,
        )
        indicator_long = indicator_long.rename(
            columns={"economy": "iso3", "Country": "country_wb"}
        )
        indicator_long["year"] = pd.to_numeric(
            indicator_long["year"], errors="coerce"
        ).astype("Int64")
        indicator_long[variable] = pd.to_numeric(
            indicator_long[variable], errors="coerce"
        )
        long_frames.append(indicator_long)

    long = pd.concat(long_frames, ignore_index=True)
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")

    macro = long.groupby(["iso3", "country_wb", "year"], dropna=False)[
        MACRO_VARIABLES
    ].first()
    macro = macro.reset_index()

    full_index = pd.MultiIndex.from_product(
        [economies, range(START_YEAR, END_YEAR + 1)],
        names=["iso3", "year"],
    )
    macro = (
        macro.set_index(["iso3", "year"])
        .reindex(full_index)
        .reset_index()
    )

    # WBGAPI labels can be missing if the metadata request changes; keep a
    # deterministic fallback from the economy endpoint.
    macro["country_wb"] = macro["country_wb"].fillna(macro["iso3"].map(country_names))
    macro["year"] = macro["year"].astype(int)

    ordered = ["iso3", "country_wb", "year", *MACRO_VARIABLES]
    return macro[ordered].sort_values(["iso3", "year"]).reset_index(drop=True)


def build_country_missingness(
    default_panel: pd.DataFrame,
    joint_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Missingness diagnostics by default-database country."""
    default_years = (
        default_panel.groupby(["country", "iso3"], dropna=False)["year"]
        .nunique()
        .rename("n_years_default_panel")
    )
    joint_years = (
        joint_panel.groupby(["country", "iso3"], dropna=False)["year"]
        .nunique()
        .rename("n_years_joint_panel")
    )
    macro_missing = joint_panel.groupby(["country", "iso3"], dropna=False)[
        MACRO_VARIABLES
    ].apply(lambda x: x.isna().sum())

    out = pd.concat([default_years, joint_years, macro_missing], axis=1).reset_index()
    out["n_years_joint_panel"] = out["n_years_joint_panel"].fillna(0).astype(int)

    for var in MACRO_VARIABLES:
        out[f"{var}_missing"] = out[var].astype(int)
        denom = out["n_years_joint_panel"].replace(0, np.nan)
        out[f"{var}_share_missing"] = out[f"{var}_missing"] / denom
        out = out.drop(columns=var)

    missing_cols = [f"{var}_missing" for var in MACRO_VARIABLES]
    out["total_missing_macro_values"] = out[missing_cols].sum(axis=1)
    total_cells = out["n_years_joint_panel"] * len(MACRO_VARIABLES)
    out["share_all_macro_cells_missing"] = (
        out["total_missing_macro_values"] / total_cells.replace(0, np.nan)
    )

    return out.sort_values(
        ["total_missing_macro_values", "country"], ascending=[False, True]
    ).reset_index(drop=True)


def build_year_missingness(joint_panel: pd.DataFrame) -> pd.DataFrame:
    """Missingness diagnostics by year."""
    grouped = joint_panel.groupby("year", dropna=False)
    out = pd.DataFrame(
        {
            "year": sorted(joint_panel["year"].dropna().unique()),
        }
    )
    out["number_countries_in_panel"] = out["year"].map(grouped.size()).astype(int)
    out["number_countries_with_default_data"] = out["year"].map(
        grouped["debt_default_usd_mil"].apply(lambda x: x.notna().sum())
    ).astype(int)

    for var in MACRO_VARIABLES:
        nonmissing = grouped[var].apply(lambda x: x.notna().sum())
        out[f"{var}_nonmissing"] = out["year"].map(nonmissing).astype(int)
        out[f"{var}_share_missing"] = (
            1 - out[f"{var}_nonmissing"] / out["number_countries_in_panel"]
        )

    missing_counts = grouped[MACRO_VARIABLES].apply(lambda x: x.isna().sum().sum())
    out["total_missing_macro_values"] = out["year"].map(missing_counts).astype(int)
    total_cells = out["number_countries_in_panel"] * len(MACRO_VARIABLES)
    out["share_all_macro_cells_missing"] = (
        out["total_missing_macro_values"] / total_cells.replace(0, np.nan)
    )

    return out.sort_values("year").reset_index(drop=True)


def write_data_summary(
    default_panel: pd.DataFrame,
    joint_panel: pd.DataFrame,
    unmatched: pd.DataFrame,
    country_missingness: pd.DataFrame,
) -> None:
    default_obs = len(default_panel)
    default_countries = default_panel["country"].nunique()
    default_year_min = int(default_panel["year"].min())
    default_year_max = int(default_panel["year"].max())
    default_events = int(default_panel["default_state"].sum())
    default_share = default_events / default_obs if default_obs else np.nan
    joint_obs = len(joint_panel)

    top_missing = country_missingness[
        ["country", "iso3", "total_missing_macro_values", "share_all_macro_cells_missing"]
    ].head(10)
    variable_missing = (
        joint_panel[MACRO_VARIABLES]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("share_missing")
    )

    lines = [
        "First-stage sovereign default and macro-financial panel summary",
        "",
        f"Default countries: {default_countries}",
        f"Default year range: {default_year_min}-{default_year_max}",
        f"Default country-year observations: {default_obs}",
        f"Joint panel country-year observations: {joint_obs}",
        f"default_state == 1: {default_events} ({default_share:.3%})",
        f"Default onset observations: {int(default_panel['onset_next_year'].sum())}",
        f"Hard onset observations: {int(default_panel['hard_onset_next_year'].sum())}",
        "Macro variables downloaded:",
        *[f"  - {name} ({code})" for code, name in WB_INDICATORS.items()],
        f"Unmatched default countries: {len(unmatched)}",
        "",
        "Macro variables with highest missingness in the joint panel:",
        *[
            f"  - {var}: {share:.3%}"
            for var, share in variable_missing.head(10).items()
        ],
        "",
        "Countries with highest total macro missingness:",
        top_missing.to_string(index=False),
    ]

    (OUTPUT_DIR / "data_summary.txt").write_text("\n".join(lines) + "\n")


def print_final_diagnostics(
    joint_panel: pd.DataFrame,
    country_missingness: pd.DataFrame,
    year_missingness: pd.DataFrame,
) -> None:
    print("\nFirst rows of joint panel:")
    print(joint_panel.head().to_string(index=False))

    print("\nFinal joint panel dimensions:")
    print(joint_panel.shape)

    print("\nDefault-state frequency:")
    print(joint_panel["default_state"].value_counts(dropna=False).to_string())

    print("\nOnset frequency:")
    print(joint_panel["onset_next_year"].value_counts(dropna=False).to_string())

    print("\nHard-onset frequency:")
    print(joint_panel["hard_onset_next_year"].value_counts(dropna=False).to_string())

    print("\nIncidence frequency:")
    print(joint_panel["incidence_next_year"].value_counts(dropna=False).to_string())

    print("\nTop 15 countries with the most macro missingness:")
    cols = ["country", "iso3", "total_missing_macro_values", "share_all_macro_cells_missing"]
    print(country_missingness[cols].head(15).to_string(index=False))

    print("\nYears with the most macro missingness:")
    cols = ["year", "total_missing_macro_values", "share_all_macro_cells_missing"]
    print(
        year_missingness.sort_values(
            ["total_missing_macro_values", "year"], ascending=[False, True]
        )[cols]
        .head(15)
        .to_string(index=False)
    )


def main() -> None:
    ensure_directories()

    default_path = download_default_database()
    default_panel, unmatched = parse_default_workbook(default_path)
    default_panel.to_csv(OUTPUT_DIR / "default_panel_raw.csv", index=False)

    macro_panel = download_world_bank_panel()
    macro_panel.to_csv(OUTPUT_DIR / "macro_panel_raw.csv", index=False)

    joint_panel = default_panel.merge(macro_panel, on=["iso3", "year"], how="left")
    joint_panel = joint_panel.sort_values(["country", "year"]).reset_index(drop=True)
    joint_panel.to_csv(OUTPUT_DIR / "joint_panel_default_macro.csv", index=False)

    country_missingness = build_country_missingness(default_panel, joint_panel)
    country_missingness.to_csv(OUTPUT_DIR / "country_missingness.csv", index=False)

    year_missingness = build_year_missingness(joint_panel)
    year_missingness.to_csv(OUTPUT_DIR / "year_missingness.csv", index=False)

    write_data_summary(default_panel, joint_panel, unmatched, country_missingness)
    print_final_diagnostics(joint_panel, country_missingness, year_missingness)


if __name__ == "__main__":
    main()
