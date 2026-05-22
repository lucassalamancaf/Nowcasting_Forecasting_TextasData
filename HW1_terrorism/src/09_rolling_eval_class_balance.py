"""Plot class balance for the rolling evaluation sample.

This uses the actual out-of-sample prediction files for incidence and onset,
so those positive rates match the performance summary reported from the ROC
and precision-recall curves. It also adds hard onset over the same country-month
test rows as a descriptive risk-set comparison.

Run from HW1_terrorism:
    python src/09_rolling_eval_class_balance.py

Outputs:
    outputs/diagnostics/rolling_evaluation_class_balance.csv
    outputs/figures/rolling_evaluation_class_balance.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INC_PREDS = PROJECT_ROOT / "data/processed/gtd_preds_inc.csv"
ONS_PREDS = PROJECT_ROOT / "data/processed/gtd_preds_ons.csv"
ALL_FEATURES = PROJECT_ROOT / "data/processed/gtd_panel_with_all_features.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
DIAG_DIR = PROJECT_ROOT / "outputs/diagnostics"


def build_balance_table() -> pd.DataFrame:
    inc = pd.read_csv(INC_PREDS)
    ons = pd.read_csv(ONS_PREDS)
    panel = pd.read_csv(ALL_FEATURES)
    panel["month"] = pd.to_datetime(panel["month"])
    panel["period"] = panel["month"].dt.year * 100 + panel["month"].dt.month

    test_keys = ons[["iso3", "period"]].drop_duplicates()
    hard = test_keys.merge(
        panel[["iso3", "period", "hard_onset_next_3months_t"]],
        on=["iso3", "period"],
        how="left",
    )

    specs = [
        (
            "Incidence next month",
            "Any strict terrorist attack in t+1",
            inc,
            "inc_target",
            "preds_inc",
        ),
        (
            "Onset next 3 months",
            "No strict attack in t; any strict terrorist attack in t+1 to t+3",
            ons,
            "ons_target",
            "preds_ons",
        ),
        (
            "Hard onset next 3 months",
            "No strict attack in t or prior 59 months; any strict terrorist attack in t+1 to t+3",
            hard,
            "hard_onset_next_3months_t",
            None,
        ),
    ]

    rows = []
    for outcome, definition, df, target_col, pred_col in specs:
        subset = [target_col]
        if pred_col is not None:
            subset.append(pred_col)
        clean = df.dropna(subset=subset)
        positives = int(clean[target_col].sum())
        valid = len(clean)
        negatives = valid - positives
        rows.append(
            {
                "outcome": outcome,
                "definition": definition,
                "positive_rows": positives,
                "negative_rows": negatives,
                "valid_rows": valid,
                "positive_rate": positives / valid,
                "negative_rate": negatives / valid,
            }
        )

    return pd.DataFrame(rows)


def plot_balance(table: pd.DataFrame) -> None:
    plot_df = table.iloc[::-1].copy()

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.8))

    y = range(len(plot_df))
    ax.barh(
        y,
        plot_df["negative_rate"],
        color="#d8d1c5",
        label="Negative class",
    )
    ax.barh(
        y,
        plot_df["positive_rate"],
        left=plot_df["negative_rate"],
        color="#b45f45",
        label="Positive class",
    )

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        label = (
            f"{row['positive_rate']:.1%} "
            f"({row['positive_rows']:,}/{row['valid_rows']:,})"
        )
        ax.text(
            0.985,
            idx,
            label,
            va="center",
            ha="right",
            color="white",
            fontsize=10,
            fontweight="semibold",
        )

    ax.set_yticks(list(y), plot_df["outcome"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share within valid rolling evaluation rows")
    ax.set_title("Class Balance in Rolling Evaluation Sample")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    out = FIGURE_DIR / "rolling_evaluation_class_balance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    table = build_balance_table()
    out = DIAG_DIR / "rolling_evaluation_class_balance.csv"
    table.to_csv(out, index=False)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_balance(table)


if __name__ == "__main__":
    main()
