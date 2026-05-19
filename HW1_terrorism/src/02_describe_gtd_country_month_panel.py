"""Create initial descriptive tables and plots for the GTD country-month panel.

This script assumes src/01_build_gtd_country_month_panel.py has already created
data/processed/gtd_country_month_panel.parquet. It adds transparent forecasting
targets for incidence, onset, three-month onset, and hard onset, then writes
descriptive tables and plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COUNT_COLUMNS = [
    "n_attacks_all",
    "n_attacks_strict",
    "n_success_all",
    "n_success_strict",
    "n_suicide_all",
    "n_suicide_strict",
    "fatalities_all",
    "fatalities_strict",
    "injuries_all",
    "injuries_strict",
    "n_fatal_attacks_all",
    "n_fatal_attacks_strict",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Describe the GTD country-month panel and create plots."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/gtd_country_month_panel.parquet"),
        help="Path to the processed country-month panel.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("outputs/diagnostics"),
        help="Directory for summary tables and reports.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for exploratory figures.",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Optional country names to zoom into. Defaults to top countries by strict attacks.",
    )
    parser.add_argument(
        "--top-n-countries",
        type=int,
        default=8,
        help="Number of top countries to include in country zoom plots if --countries is omitted.",
    )
    parser.add_argument(
        "--hard-clean-months",
        type=int,
        default=60,
        help="Number of prior attack-free months required for hard onset.",
    )
    parser.add_argument(
        "--outcome-panel-parquet",
        type=Path,
        default=Path("data/processed/gtd_country_month_panel_with_outcomes.parquet"),
        help="Output parquet path for the panel with forecasting outcomes.",
    )
    parser.add_argument(
        "--outcome-panel-csv",
        type=Path,
        default=Path("data/processed/gtd_country_month_panel_with_outcomes.csv"),
        help="Output CSV path for the panel with forecasting outcomes.",
    )
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    """Load the processed panel and ensure expected date/indicator columns exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Panel file not found: {path}\n"
            "Run src/01_build_gtd_country_month_panel.py first."
        )

    panel = pd.read_parquet(path)
    panel["month"] = pd.to_datetime(panel["month"])
    if "attack_month_t" not in panel.columns:
        panel["attack_month_t"] = panel["n_attacks_strict"].gt(0).astype("int64")
    return panel.sort_values(["country", "month"]).reset_index(drop=True)


def nullable_binary(values: pd.Series) -> pd.Series:
    """Convert a series with 0/1/missing values to nullable integer."""
    return values.astype("Int64")


def add_forecasting_outcomes(
    panel: pd.DataFrame, hard_clean_months: int = 60
) -> pd.DataFrame:
    """Add incidence, onset, three-month onset, and hard-onset targets.

    The hard-onset definition requires no strict attack months in the prior
    `hard_clean_months`, including the current month t, and an attack in one of
    the next three months.
    """
    panel = panel.sort_values(["country", "month"]).copy()
    grouped = panel.groupby("country", sort=False)["attack_month_t"]

    future_1 = grouped.shift(-1)
    future_2 = grouped.shift(-2)
    future_3 = grouped.shift(-3)

    future_1_available = future_1.notna()
    future_3_available = future_1.notna() & future_2.notna() & future_3.notna()
    future_any_3 = (
        pd.concat([future_1, future_2, future_3], axis=1)
        .max(axis=1, skipna=False)
    )

    panel["incidence_next_month_t"] = nullable_binary(future_1)

    onset_1 = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    onset_1_risk = panel["attack_month_t"].eq(0) & future_1_available
    onset_1.loc[onset_1_risk] = future_1.loc[onset_1_risk].astype("int64")
    panel["onset_next_month_t"] = onset_1

    onset_3 = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    onset_3_risk = panel["attack_month_t"].eq(0) & future_3_available
    onset_3.loc[onset_3_risk] = future_any_3.loc[onset_3_risk].astype("int64")
    panel["onset_next_3months_t"] = onset_3

    prior_clean_sum = (
        panel.groupby("country", sort=False)["attack_month_t"]
        .rolling(window=hard_clean_months, min_periods=hard_clean_months)
        .sum()
        .reset_index(level=0, drop=True)
    )
    hard_risk = prior_clean_sum.eq(0) & future_3_available
    hard_onset = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    hard_onset.loc[hard_risk] = future_any_3.loc[hard_risk].astype("int64")
    panel["hard_onset_next_3months_t"] = hard_onset
    panel["hard_onset_clean_months"] = hard_clean_months

    return panel.reset_index(drop=True)


def save_outcome_panel(panel: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    """Save a copy of the panel with forecasting outcomes."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(parquet_path, index=False)
    panel.to_csv(csv_path, index=False)


def ensure_dirs(*paths: Path) -> None:
    """Create output directories."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table without extra dependencies."""
    if df.empty:
        return ""
    printable = df.copy().astype(str)
    headers = list(printable.columns)
    rows = printable.values.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line] + body_lines)


def check_balance(panel: pd.DataFrame) -> pd.DataFrame:
    """Check whether each country has the same full set of months."""
    all_months = set(panel["month"].drop_duplicates())
    expected_months = len(all_months)
    balance = (
        panel.groupby(["country", "country_txt"], as_index=False)
        .agg(
            n_rows=("month", "size"),
            n_unique_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
        )
        .sort_values(["n_unique_months", "country_txt"])
    )
    balance["expected_months"] = expected_months
    balance["is_balanced"] = balance["n_unique_months"].eq(expected_months)

    duplicate_counts = (
        panel.groupby(["country", "month"], as_index=False)
        .size()
        .rename(columns={"size": "duplicate_rows"})
    )
    n_duplicate_country_months = int(duplicate_counts["duplicate_rows"].gt(1).sum())
    balance.attrs["n_duplicate_country_months"] = n_duplicate_country_months
    balance.attrs["is_balanced_panel"] = bool(
        balance["is_balanced"].all() and n_duplicate_country_months == 0
    )
    return balance


def build_summary_tables(panel: pd.DataFrame, diagnostics_dir: Path) -> dict[str, pd.DataFrame]:
    """Create and save exploratory summary tables."""
    strict_attack_month = panel["n_attacks_strict"].gt(0).astype("int64")
    all_attack_month = panel["n_attacks_all"].gt(0).astype("int64")
    class_balance = pd.DataFrame(
        [
            {
                "outcome": "strict_attack_month",
                "definition": "n_attacks_strict > 0",
                "positive_rows": int(strict_attack_month.sum()),
                "negative_rows": int((1 - strict_attack_month).sum()),
                "total_rows": len(panel),
                "positive_rate": strict_attack_month.mean(),
                "negative_rate": 1 - strict_attack_month.mean(),
            },
            {
                "outcome": "all_gtd_attack_month",
                "definition": "n_attacks_all > 0",
                "positive_rows": int(all_attack_month.sum()),
                "negative_rows": int((1 - all_attack_month).sum()),
                "total_rows": len(panel),
                "positive_rate": all_attack_month.mean(),
                "negative_rate": 1 - all_attack_month.mean(),
            },
        ]
    )
    class_balance["extra_positive_vs_strict"] = [
        0,
        int(((all_attack_month == 1) & (strict_attack_month == 0)).sum()),
    ]

    forecasting_outcomes = [
        {
            "outcome": "incidence_next_month_t",
            "definition": "strict attack occurs in t+1",
        },
        {
            "outcome": "onset_next_month_t",
            "definition": "no strict attack in t; strict attack occurs in t+1",
        },
        {
            "outcome": "onset_next_3months_t",
            "definition": "no strict attack in t; strict attack occurs in t+1, t+2, or t+3",
        },
        {
            "outcome": "hard_onset_next_3months_t",
            "definition": "60 prior attack-free months through t; strict attack occurs in t+1, t+2, or t+3",
        },
    ]
    forecasting_balance_rows = []
    for item in forecasting_outcomes:
        outcome = item["outcome"]
        valid = panel[outcome].notna()
        positive = int(panel.loc[valid, outcome].sum())
        valid_rows = int(valid.sum())
        negative = valid_rows - positive
        forecasting_balance_rows.append(
            {
                "outcome": outcome,
                "definition": item["definition"],
                "positive_rows": positive,
                "negative_rows": negative,
                "valid_rows": valid_rows,
                "missing_rows": int((~valid).sum()),
                "positive_rate": positive / valid_rows if valid_rows else pd.NA,
                "negative_rate": negative / valid_rows if valid_rows else pd.NA,
            }
        )
    forecasting_balance = pd.DataFrame(forecasting_balance_rows)

    outcome_region_rows = []
    for item in forecasting_outcomes:
        outcome = item["outcome"]
        region_outcome = (
            panel.loc[panel[outcome].notna()]
            .groupby(["region", "region_txt"], as_index=False)
            .agg(
                valid_rows=(outcome, "size"),
                positive_rows=(outcome, "sum"),
            )
        )
        region_outcome["negative_rows"] = (
            region_outcome["valid_rows"] - region_outcome["positive_rows"]
        )
        region_outcome["positive_rate"] = (
            region_outcome["positive_rows"] / region_outcome["valid_rows"]
        )
        region_outcome["outcome"] = outcome
        outcome_region_rows.append(region_outcome)
    outcome_region_summary = pd.concat(outcome_region_rows, ignore_index=True)

    region_summary = (
        panel.groupby(["region", "region_txt"], as_index=False)
        .agg(
            countries=("country", "nunique"),
            country_months=("month", "size"),
            attack_months=("attack_month_t", "sum"),
            n_attacks_strict=("n_attacks_strict", "sum"),
            n_attacks_all=("n_attacks_all", "sum"),
            fatalities_all=("fatalities_all", "sum"),
            injuries_all=("injuries_all", "sum"),
        )
        .sort_values("n_attacks_strict", ascending=False)
    )
    region_summary["attack_month_share"] = (
        region_summary["attack_months"] / region_summary["country_months"]
    )

    country_summary = (
        panel.groupby(["country", "country_txt", "region_txt"], as_index=False)
        .agg(
            country_months=("month", "size"),
            attack_months=("attack_month_t", "sum"),
            n_attacks_strict=("n_attacks_strict", "sum"),
            n_attacks_all=("n_attacks_all", "sum"),
            fatalities_all=("fatalities_all", "sum"),
            injuries_all=("injuries_all", "sum"),
            max_monthly_attacks=("n_attacks_strict", "max"),
        )
        .sort_values("n_attacks_strict", ascending=False)
    )
    country_summary["attack_month_share"] = (
        country_summary["attack_months"] / country_summary["country_months"]
    )

    yearly_summary = (
        panel.assign(year=panel["month"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            n_attacks_strict=("n_attacks_strict", "sum"),
            n_attacks_all=("n_attacks_all", "sum"),
            active_country_months=("attack_month_t", "sum"),
            incidence_next_month=("incidence_next_month_t", "sum"),
            onset_next_3months=("onset_next_3months_t", "sum"),
            hard_onset_next_3months=("hard_onset_next_3months_t", "sum"),
            fatalities_all=("fatalities_all", "sum"),
            injuries_all=("injuries_all", "sum"),
        )
    )

    monthly_global = (
        panel.groupby("month", as_index=False)
        .agg(
            n_attacks_strict=("n_attacks_strict", "sum"),
            n_attacks_all=("n_attacks_all", "sum"),
            active_countries=("attack_month_t", "sum"),
            onset_next_3months=("onset_next_3months_t", "sum"),
            hard_onset_next_3months=("hard_onset_next_3months_t", "sum"),
            fatalities_all=("fatalities_all", "sum"),
            injuries_all=("injuries_all", "sum"),
        )
    )

    balance = check_balance(panel)

    tables = {
        "region_summary": region_summary,
        "country_summary": country_summary,
        "yearly_summary": yearly_summary,
        "monthly_global": monthly_global,
        "panel_balance_by_country": balance,
        "outcome_class_balance": class_balance,
        "forecasting_outcome_class_balance": forecasting_balance,
        "forecasting_outcome_region_summary": outcome_region_summary,
    }
    for name, table in tables.items():
        table.to_csv(diagnostics_dir / f"{name}.csv", index=False)
    return tables


def make_plots(
    panel: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    figures_dir: Path,
    countries: list[str] | None,
    top_n_countries: int,
) -> list[str]:
    """Create exploratory plots and return the written filenames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titleweight": "semibold",
        }
    )

    positive_color = "#2f6f73"
    negative_color = "#d9d3c7"
    accent_color = "#b45f45"
    line_color = "#264653"
    region_colors = [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#577590",
        "#43aa8b",
        "#f94144",
        "#90be6d",
        "#bc6c25",
        "#6d597a",
        "#4d908e",
    ]

    written = []
    monthly_global = tables["monthly_global"]
    region_summary = tables["region_summary"]
    country_summary = tables["country_summary"]
    class_balance = tables["outcome_class_balance"]
    forecasting_balance = tables["forecasting_outcome_class_balance"]
    outcome_region = tables["forecasting_outcome_region_summary"]

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_global["month"], monthly_global["active_countries"], color=line_color)
    plt.title("Countries with at Least One Strict Attack by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of countries")
    plt.tight_layout()
    path = figures_dir / "active_countries_by_month.png"
    plt.savefig(path, dpi=200)
    plt.close()
    written.append(str(path))

    class_plot = class_balance.set_index("outcome")[["positive_rate", "negative_rate"]]
    labels = ["Strict attacks", "All GTD attacks"]
    y_positions = range(len(class_plot))
    plt.figure(figsize=(8, 4.2))
    plt.barh(
        y_positions,
        class_plot["negative_rate"],
        color=negative_color,
        label="No attack month",
    )
    plt.barh(
        y_positions,
        class_plot["positive_rate"],
        left=class_plot["negative_rate"],
        color=positive_color,
        label="Attack month",
    )
    for y_pos, positive_rate in zip(y_positions, class_plot["positive_rate"]):
        plt.text(
            0.985,
            y_pos,
            f"{positive_rate:.1%}",
            va="center",
            ha="right",
            color="white",
            fontweight="semibold",
        )
    plt.title("Class Balance for Attack-Month Outcomes")
    plt.xlabel("Share of country-month rows")
    plt.yticks(y_positions, labels)
    plt.xlim(0, 1)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    plt.tight_layout()
    path = figures_dir / "class_balance_attack_months.png"
    plt.savefig(path, dpi=200)
    plt.close()
    written.append(str(path))

    outcome_labels = {
        "incidence_next_month_t": "Incidence\nnext month",
        "onset_next_month_t": "Onset\nnext month",
        "onset_next_3months_t": "Onset\nnext 3 months",
        "hard_onset_next_3months_t": "Hard onset\nnext 3 months",
    }
    forecast_plot = forecasting_balance.copy()
    forecast_plot["label"] = forecast_plot["outcome"].map(outcome_labels)
    y_positions = range(len(forecast_plot))
    plt.figure(figsize=(8.5, 5))
    plt.barh(
        y_positions,
        forecast_plot["negative_rate"],
        color=negative_color,
        label="0",
    )
    plt.barh(
        y_positions,
        forecast_plot["positive_rate"],
        left=forecast_plot["negative_rate"],
        color=accent_color,
        label="1",
    )
    for y_pos, row in zip(y_positions, forecast_plot.itertuples(index=False)):
        plt.text(
            0.985,
            y_pos,
            f"{row.positive_rate:.1%} ({row.positive_rows:,}/{row.valid_rows:,})",
            va="center",
            ha="right",
            color="white",
            fontweight="semibold",
            fontsize=9,
        )
    plt.title("Class Balance for Forecasting Targets")
    plt.xlabel("Share within valid risk set")
    plt.yticks(y_positions, forecast_plot["label"])
    plt.xlim(0, 1)
    plt.legend(
        ["Negative class", "Positive class"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout()
    path = figures_dir / "forecast_outcome_class_balance.png"
    plt.savefig(path, dpi=200)
    plt.close()
    written.append(str(path))

    outcome_region_plot = outcome_region.loc[
        outcome_region["outcome"].isin(
            ["onset_next_3months_t", "hard_onset_next_3months_t"]
        )
    ].copy()
    outcome_region_plot["label"] = outcome_region_plot["outcome"].map(
        {
            "onset_next_3months_t": "Onset next 3 months",
            "hard_onset_next_3months_t": "Hard onset next 3 months",
        }
    )
    top_regions_for_outcomes = (
        region_summary.sort_values("n_attacks_strict", ascending=False)
        .head(8)["region_txt"]
        .tolist()
    )
    outcome_region_plot = outcome_region_plot.loc[
        outcome_region_plot["region_txt"].isin(top_regions_for_outcomes)
    ]
    pivot_region_outcomes = outcome_region_plot.pivot(
        index="region_txt", columns="label", values="positive_rate"
    ).reindex(top_regions_for_outcomes)
    if not pivot_region_outcomes.empty:
        ax = pivot_region_outcomes.plot(
            kind="barh",
            figsize=(9, 5.5),
            color=[positive_color, accent_color],
        )
        ax.set_title("Three-Month Onset Rates by Region")
        ax.set_xlabel("Positive rate within valid risk set")
        ax.set_ylabel("")
        ax.legend(frameon=False)
        plt.tight_layout()
        path = figures_dir / "three_month_onset_rates_by_region.png"
        plt.savefig(path, dpi=200)
        plt.close()
        written.append(str(path))

    region_monthly = (
        panel.groupby(["month", "region_txt"], as_index=False)["n_attacks_strict"]
        .sum()
        .pivot(index="month", columns="region_txt", values="n_attacks_strict")
        .fillna(0)
    )
    plt.figure(figsize=(11, 6))
    for index, column in enumerate(region_monthly.columns):
        plt.plot(
            region_monthly.index,
            region_monthly[column],
            label=column,
            linewidth=1.5,
            color=region_colors[index % len(region_colors)],
        )
    plt.title("Strict GTD Attacks by Region and Month")
    plt.xlabel("Month")
    plt.ylabel("Number of strict attacks")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = figures_dir / "region_attacks_by_month.png"
    plt.savefig(path, dpi=200)
    plt.close()
    written.append(str(path))

    top_regions = region_summary.sort_values("n_attacks_strict").tail(12)
    plt.figure(figsize=(9, 5))
    plt.barh(top_regions["region_txt"], top_regions["n_attacks_strict"], color=positive_color)
    plt.title("Strict GTD Attacks by Region")
    plt.xlabel("Number of strict attacks")
    plt.ylabel("")
    plt.tight_layout()
    path = figures_dir / "region_total_attacks_strict.png"
    plt.savefig(path, dpi=200)
    plt.close()
    written.append(str(path))

    region_share = region_summary.sort_values("attack_month_share")
    plt.figure(figsize=(9, 5))
    plt.barh(region_share["region_txt"], region_share["attack_month_share"], color=accent_color)
    plt.title("Share of Country-Months with at Least One Strict Attack")
    plt.xlabel("Share")
    plt.ylabel("")
    plt.tight_layout()
    path = figures_dir / "region_attack_month_share.png"
    plt.savefig(path, dpi=200)
    plt.close()
    written.append(str(path))

    positive_counts = panel.loc[panel["n_attacks_strict"] > 0, "n_attacks_strict"]
    plt.figure(figsize=(9, 5))
    plt.hist(positive_counts, bins=50, color=positive_color, edgecolor="white")
    plt.yscale("log")
    plt.title("Distribution of Strict Attacks in Positive Country-Months")
    plt.xlabel("Strict attacks in country-month")
    plt.ylabel("Number of country-months, log scale")
    plt.tight_layout()
    path = figures_dir / "positive_country_month_attack_distribution.png"
    plt.savefig(path, dpi=200)
    plt.close()
    written.append(str(path))

    if countries:
        selected_countries = countries
    else:
        selected_countries = country_summary.head(top_n_countries)["country_txt"].tolist()

    country_panel = panel.loc[panel["country_txt"].isin(selected_countries)].copy()
    country_monthly = country_panel.pivot(
        index="month", columns="country_txt", values="n_attacks_strict"
    ).fillna(0)
    if not country_monthly.empty:
        plt.figure(figsize=(11, 6))
        for index, column in enumerate(country_monthly.columns):
            plt.plot(
                country_monthly.index,
                country_monthly[column],
                label=column,
                color=region_colors[index % len(region_colors)],
            )
        plt.title("Country Zoom: Strict Attacks by Month")
        plt.xlabel("Month")
        plt.ylabel("Number of strict attacks")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        path = figures_dir / "country_zoom_attacks_by_month.png"
        plt.savefig(path, dpi=200)
        plt.close()
        written.append(str(path))

    top_heatmap_countries = country_summary.head(30)["country_txt"].tolist()
    heatmap = (
        panel.loc[panel["country_txt"].isin(top_heatmap_countries)]
        .pivot(index="country_txt", columns="month", values="attack_month_t")
        .reindex(top_heatmap_countries)
    )
    if not heatmap.empty:
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="Greys")
        plt.title("Attack-Month Indicator for Top 30 Countries")
        plt.xlabel("Month")
        plt.ylabel("")
        tick_positions = range(0, len(heatmap.columns), 12)
        tick_labels = [heatmap.columns[i].strftime("%Y") for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.yticks(range(len(heatmap.index)), heatmap.index, fontsize=7)
        plt.tight_layout()
        path = figures_dir / "top30_country_attack_month_heatmap.png"
        plt.savefig(path, dpi=200)
        plt.close()
        written.append(str(path))

    return written


def write_report(
    panel: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    plot_paths: list[str],
    diagnostics_dir: Path,
    hard_clean_months: int,
) -> None:
    """Write a compact exploratory markdown report."""
    balance = tables["panel_balance_by_country"]
    region_summary = tables["region_summary"]
    country_summary = tables["country_summary"]
    yearly_summary = tables["yearly_summary"]
    class_balance = tables["outcome_class_balance"]
    forecasting_balance = tables["forecasting_outcome_class_balance"]
    outcome_region = tables["forecasting_outcome_region_summary"]

    total_country_months = len(panel)
    attack_months = int(panel["attack_month_t"].sum())
    zero_months = total_country_months - attack_months
    countries_with_no_attacks = int(
        country_summary["n_attacks_strict"].eq(0).sum()
    )

    lines = [
        "# GTD Initial Data Description",
        "",
        "This report describes the processed country-month panel. "
        "`attack_month_t` equals 1 when `n_attacks_strict > 0` in the same month. "
        "Forecasting outcomes are built from strict GTD attacks.",
        "",
        "## Panel Balance",
        "",
        f"- Country-month rows: {total_country_months:,}",
        f"- Countries: {panel['country'].nunique():,}",
        f"- Months: {panel['month'].nunique():,}",
        f"- Balanced panel: {balance.attrs['is_balanced_panel']}",
        f"- Duplicate country-month cells: {balance.attrs['n_duplicate_country_months']:,}",
        "",
        "## Attack-Month Indicator",
        "",
        f"- Country-months with `attack_month_t == 1`: {attack_months:,}",
        f"- Country-months with `attack_month_t == 0`: {zero_months:,}",
        f"- Share with at least one strict attack: {attack_months / total_country_months:.3f}",
        f"- Countries with zero strict attacks in the selected period: {countries_with_no_attacks:,}",
        "",
        "## Binary Outcome Class Balance",
        "",
        dataframe_to_markdown(
            class_balance.round(
                {
                    "positive_rate": 3,
                    "negative_rate": 3,
                }
            )
        ),
        "",
        "## Forecasting Outcomes",
        "",
        f"`hard_onset_next_3months_t` uses a {hard_clean_months}-month clean spell.",
        "",
        dataframe_to_markdown(
            forecasting_balance.round(
                {
                    "positive_rate": 3,
                    "negative_rate": 3,
                }
            )
        ),
        "",
        "## Three-Month Onset Rates by Region",
        "",
        dataframe_to_markdown(
            outcome_region.loc[
                outcome_region["outcome"].isin(
                    ["onset_next_3months_t", "hard_onset_next_3months_t"]
                ),
                [
                    "outcome",
                    "region_txt",
                    "valid_rows",
                    "positive_rows",
                    "negative_rows",
                    "positive_rate",
                ],
            ]
            .sort_values(["outcome", "positive_rate"], ascending=[True, False])
            .head(24)
            .round({"positive_rate": 3})
        ),
        "",
        "## Top Regions by Strict Attacks",
        "",
        dataframe_to_markdown(
            region_summary[
                [
                    "region_txt",
                    "countries",
                    "country_months",
                    "attack_months",
                    "attack_month_share",
                    "n_attacks_strict",
                    "fatalities_all",
                    "injuries_all",
                ]
            ].head(12).round({"attack_month_share": 3})
        ),
        "",
        "## Top 20 Countries by Strict Attacks",
        "",
        dataframe_to_markdown(
            country_summary[
                [
                    "country_txt",
                    "region_txt",
                    "attack_months",
                    "attack_month_share",
                    "n_attacks_strict",
                    "fatalities_all",
                    "injuries_all",
                    "max_monthly_attacks",
                ]
            ].head(20).round({"attack_month_share": 3})
        ),
        "",
        "## Yearly Summary",
        "",
        dataframe_to_markdown(yearly_summary),
        "",
        "## Figures Written",
        "",
    ]
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.append("")

    (diagnostics_dir / "gtd_initial_data_description.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    """Run initial GTD panel description."""
    args = parse_args()
    ensure_dirs(args.diagnostics_dir, args.figures_dir)

    panel = load_panel(args.panel)
    panel = add_forecasting_outcomes(panel, hard_clean_months=args.hard_clean_months)
    save_outcome_panel(panel, args.outcome_panel_parquet, args.outcome_panel_csv)
    tables = build_summary_tables(panel, args.diagnostics_dir)
    plot_paths = make_plots(
        panel=panel,
        tables=tables,
        figures_dir=args.figures_dir,
        countries=args.countries,
        top_n_countries=args.top_n_countries,
    )
    write_report(
        panel,
        tables,
        plot_paths,
        args.diagnostics_dir,
        hard_clean_months=args.hard_clean_months,
    )

    print(f"Loaded panel rows: {len(panel):,}")
    print(f"Countries: {panel['country'].nunique():,}")
    print(f"Months: {panel['month'].nunique():,}")
    print(f"attack_month_t == 1 rows: {int(panel['attack_month_t'].sum()):,}")
    print(f"Saved outcome panel: {args.outcome_panel_parquet}")
    print(f"Saved report: {args.diagnostics_dir / 'gtd_initial_data_description.md'}")


if __name__ == "__main__":
    main()
