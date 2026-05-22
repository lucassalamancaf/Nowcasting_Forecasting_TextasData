"""Describe text-topic signal before terrorism onset.

This diagnostic is not a forecasting model. It compares the 15 conflictforecast
topic stocks in country-months where onset occurs in the next three months
against country-months in the onset risk set where onset does not occur. The
standardized mean difference is useful for the report because it shows which
text features are empirically elevated before onset.

Run from HW1_terrorism:
    python src/08_topic_onset_signal.py

Outputs:
    outputs/diagnostics/topic_onset_signal.csv
    outputs/figures/topic_onset_signal.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_CSV = PROJECT_ROOT / "data/processed/gtd_panel_with_all_features.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
DIAG_DIR = PROJECT_ROOT / "outputs/diagnostics"

TOPIC_COLS = [f"stock_topic_{i}" for i in range(15)]

TOPIC_LABELS = {
    "stock_topic_2": "Topic 2: conflict discourse",
    "stock_topic_6": "Topic 6: geopolitical tension",
    "stock_topic_8": "Topic 8: Middle East instability",
    "stock_topic_11": "Topic 11: conflict discourse",
}


def build_signal_table() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV)
    df["month"] = pd.to_datetime(df["month"])
    df = df.dropna(subset=["onset_next_3months_t"] + TOPIC_COLS).copy()

    # Match the modelling period after the text merge.
    df = df[(df["month"] >= "2010-01-01") & (df["month"] <= "2020-12-01")]

    rows = []
    for col in TOPIC_COLS:
        pos = df.loc[df["onset_next_3months_t"] == 1, col]
        neg = df.loc[df["onset_next_3months_t"] == 0, col]
        pooled_sd = df[col].std(ddof=0)
        std_diff = (pos.mean() - neg.mean()) / pooled_sd if pooled_sd else np.nan
        rows.append(
            {
                "topic": col,
                "label": TOPIC_LABELS.get(col, col.replace("stock_", "").replace("_", " ")),
                "mean_before_onset": pos.mean(),
                "mean_no_onset": neg.mean(),
                "difference": pos.mean() - neg.mean(),
                "standardized_difference": std_diff,
                "positive_rows": len(pos),
                "negative_rows": len(neg),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        "standardized_difference", ascending=False
    )
    return out


def plot_signal(table: pd.DataFrame) -> None:
    plot_df = table.sort_values("standardized_difference", ascending=True)
    colors = [
        "#b45f45" if x > 0 else "#2f6f73"
        for x in plot_df["standardized_difference"]
    ]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["label"], plot_df["standardized_difference"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardized mean difference: onset risk months minus non-onset months")
    ax.set_ylabel("")
    ax.set_title("Text Topic Stocks Before Three-Month Terrorism Onset")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    out = FIGURE_DIR / "topic_onset_signal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    table = build_signal_table()
    out = DIAG_DIR / "topic_onset_signal.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
    print(table.head(8).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_signal(table)


if __name__ == "__main__":
    main()
