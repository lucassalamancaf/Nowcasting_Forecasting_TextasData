"""Feature importance and model insights for the GTD terrorism onset model.

Reads fitted estimators from data/processed/ and produces:
  1. Feature importance plots (mean across all CV folds)
  2. Feature importance comparison: incidence vs onset
  3. Prediction calibration by region
  4. Top false negatives and false positives analysis
  5. Prediction stability across CV folds

Run from HW1_terrorism/ directory.
"""

from __future__ import annotations
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score

OUTPUT_DIR = Path("outputs/figures")
DIAG_DIR   = Path("outputs/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
})

COLOR_INC = "#2f6f73"
COLOR_ONS = "#b45f45"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_all():
    X       = pd.read_parquet("data/processed/gtd_features.parquet")
    inc_df  = pd.read_parquet("data/processed/gtd_target_inc.parquet")
    ons_df  = pd.read_parquet("data/processed/gtd_target_ons.parquet")
    inc_preds = pd.read_csv("data/processed/gtd_preds_inc.csv", index_col=[0, 1])
    ons_preds = pd.read_csv("data/processed/gtd_preds_ons.csv", index_col=[0, 1])

    with open("data/processed/fitted_estimators_inc.pkl", "rb") as f:
        estimators_inc = pickle.load(f)
    with open("data/processed/fitted_estimators_ons.pkl", "rb") as f:
        estimators_ons = pickle.load(f)

    # Load panel for region/country labels
    panel = pd.read_parquet("data/processed/gtd_panel_with_all_features.parquet")
    panel["period"] = pd.to_datetime(panel["month"]).dt.year * 100 + \
                      pd.to_datetime(panel["month"]).dt.month
    panel = panel.set_index(["iso3", "period"])

    print(f"Loaded {len(estimators_inc)} incidence estimators")
    print(f"Loaded {len(estimators_ons)} onset estimators")
    print(f"Features: {X.shape[1]}")
    return X, inc_df, ons_df, inc_preds, ons_preds, estimators_inc, estimators_ons, panel


# ── 1. Feature importance ─────────────────────────────────────────────────────

def feature_importance(estimators: list, feature_names: list[str], label: str) -> pd.DataFrame:
    """Average feature importance across all CV folds."""
    importances = np.array([est.feature_importances_ for est in estimators])
    mean_imp = importances.mean(axis=0)
    std_imp  = importances.std(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_imp,
        "std": std_imp,
    }).sort_values("importance", ascending=False)

    # Add feature group labels
    def group(name):
        if "stock_topic" in name:   return "Text (LDA topics)"
        if "rm" in name:            return "Attack history (rolling mean)"
        if "since" in name:         return "Attack history (since)"
        if "ongoing" in name:       return "Attack history (ongoing)"
        if name in ["gdp_pc_growth","unemployment_rate",
                    "govt_expenditure_gdp","inflation_cpi"]: return "Macro (World Bank)"
        if name == "democracy_index": return "Political (V-Dem)"
        if name == "ongoing_conflict": return "Conflict (UCDP)"
        return "Other"

    df["group"] = df["feature"].apply(group)
    df.to_csv(DIAG_DIR / f"feature_importance_{label}.csv", index=False)
    return df


def plot_feature_importance(imp_inc: pd.DataFrame, imp_ons: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    group_colors = {
        "Text (LDA topics)":             "#2a9d8f",
        "Attack history (rolling mean)": "#264653",
        "Attack history (since)":        "#457b9d",
        "Attack history (ongoing)":      "#1d3557",
        "Macro (World Bank)":            "#e9c46a",
        "Political (V-Dem)":             "#f4a261",
        "Conflict (UCDP)":               "#e76f51",
        "Other":                         "#adb5bd",
    }

    for ax, imp, title, color in [
        (axes[0], imp_inc.head(20), "Incidence", COLOR_INC),
        (axes[1], imp_ons.head(20), "Onset (3-month)", COLOR_ONS),
    ]:
        plot_df = imp.head(20).sort_values("importance")
        colors  = [group_colors.get(g, "#adb5bd") for g in plot_df["group"]]

        bars = ax.barh(plot_df["feature"], plot_df["importance"],
                       color=colors, edgecolor="white", linewidth=0.5)
        ax.errorbar(plot_df["importance"], range(len(plot_df)),
                    xerr=plot_df["std"], fmt="none", color="gray",
                    capsize=2, linewidth=0.8)

        ax.set_title(f"Top 20 Features — {title}", fontweight="semibold")
        ax.set_xlabel("Mean importance (across 24 CV folds)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=g)
        for g, c in group_colors.items()
        if g != "Other"
    ]
    axes[1].legend(handles=legend_elements, loc="lower right",
                   fontsize=8, frameon=False)

    plt.suptitle("Random Forest Feature Importance\nMean across 24 rolling CV folds",
                 fontsize=12, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/figures/feature_importance.png")


def plot_group_importance(imp_inc: pd.DataFrame, imp_ons: pd.DataFrame) -> None:
    """Grouped bar chart: importance by feature category."""
    group_inc = imp_inc.groupby("group")["importance"].sum().sort_values(ascending=False)
    group_ons = imp_ons.groupby("group")["importance"].sum().reindex(group_inc.index)

    x = np.arange(len(group_inc))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, group_inc.values, width, label="Incidence", color=COLOR_INC, alpha=0.85)
    ax.bar(x + width/2, group_ons.values, width, label="Onset",     color=COLOR_ONS, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(group_inc.index, rotation=20, ha="right")
    ax.set_ylabel("Total importance share")
    ax.set_title("Feature Importance by Group", fontweight="semibold")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_by_group.png", dpi=200)
    plt.show()
    print("Saved: outputs/figures/feature_importance_by_group.png")

    print("\nImportance share by group:")
    summary = pd.DataFrame({"incidence": group_inc, "onset": group_ons})
    print(summary.round(3).to_string())


# ── 2. Performance by region ──────────────────────────────────────────────────

def performance_by_region(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    pred_col: str,
    target_col: str,
    label: str,
) -> None:
    """AUC-ROC per region for onset predictions."""
    merged = preds[[target_col, pred_col]].join(
        panel[["region_txt"]], how="left"
    ).dropna(subset=[target_col, pred_col, "region_txt"])

    results = []
    for region, group in merged.groupby("region_txt"):
        if group[target_col].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(group[target_col], group[pred_col])
            results.append({
                "region": region,
                "auc_roc": auc,
                "n_rows": len(group),
                "positive_rate": group[target_col].mean(),
            })
        except Exception:
            pass

    df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    df.to_csv(DIAG_DIR / f"performance_by_region_{label}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLOR_ONS if auc >= 0.8 else
              "#e9c46a" if auc >= 0.7 else "#e76f51"
              for auc in df["auc_roc"]]
    ax.barh(df["region"], df["auc_roc"], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, label="Random (0.5)")
    ax.axvline(0.8, color=COLOR_ONS, linestyle=":", lw=1, label="Good (0.8)")
    ax.set_xlabel("AUC-ROC")
    ax.set_title(f"Forecast Performance by Region — {label}", fontweight="semibold")
    ax.legend(frameon=False)
    ax.set_xlim([0.4, 1.0])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"performance_by_region_{label.lower()}.png", dpi=200)
    plt.show()
    print(f"Saved: outputs/figures/performance_by_region_{label.lower()}.png")
    print(f"\nRegional AUC-ROC ({label}):")
    print(df.round(3).to_string(index=False))


# ── 3. Prediction stability across folds ─────────────────────────────────────

def fold_stability(estimators: list, X: pd.DataFrame, label: str) -> None:
    """How stable are feature importances across the 24 CV folds?"""
    importances = np.array([est.feature_importances_ for est in estimators])

    # Coefficient of variation per feature
    cv = importances.std(axis=0) / (importances.mean(axis=0) + 1e-10)
    stability = pd.DataFrame({
        "feature": X.columns,
        "mean_importance": importances.mean(axis=0),
        "cv": cv,
    }).sort_values("mean_importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        stability["mean_importance"],
        stability["cv"],
        s=80,
        color=COLOR_ONS if "ons" in label.lower() else COLOR_INC,
        alpha=0.8
    )
    for _, row in stability.iterrows():
        ax.annotate(row["feature"], (row["mean_importance"], row["cv"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Mean importance")
    ax.set_ylabel("Coefficient of variation (lower = more stable)")
    ax.set_title(f"Feature Stability Across 24 CV Folds — {label}",
                 fontweight="semibold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"feature_stability_{label.lower()}.png", dpi=200)
    plt.show()
    print(f"Saved: outputs/figures/feature_stability_{label.lower()}.png")


# ── 4. Worst false negatives ──────────────────────────────────────────────────

def false_negative_analysis(
    preds: pd.DataFrame,
    panel: pd.DataFrame,
    pred_col: str,
    target_col: str,
    threshold: float = 0.106,
) -> None:
    """Which real onsets did the model most confidently miss?"""
    merged = preds[[target_col, pred_col]].join(
        panel[["country_txt", "region_txt"]], how="left"
    ).dropna(subset=[target_col, pred_col])

    merged["predicted_positive"] = (merged[pred_col] >= threshold).astype(int)
    fn = merged[
        (merged[target_col] == 1) & (merged["predicted_positive"] == 0)
    ].sort_values(pred_col)

    print(f"\n=== Top 15 False Negatives (real onsets the model missed) ===")
    print(f"Using threshold = {threshold} (optimal cost-model cutoff)")
    print(fn[["country_txt", "region_txt", pred_col, target_col]].head(15).to_string())

    fp = merged[
        (merged[target_col] == 0) & (merged["predicted_positive"] == 1)
    ].sort_values(pred_col, ascending=False)

    print(f"\n=== Top 15 False Positives (model raised alarm but no onset) ===")
    print(fp[["country_txt", "region_txt", pred_col, target_col]].head(15).to_string())

    fn.to_csv(DIAG_DIR / "false_negatives_onset.csv")
    fp.to_csv(DIAG_DIR / "false_positives_onset.csv")


# ── 5. Prediction score over time ─────────────────────────────────────────────

def predictions_over_time(
    preds: pd.DataFrame,
    pred_col: str,
    target_col: str,
    label: str,
) -> None:
    """Average predicted risk over time — does the model capture the 2012-2015 surge?"""
    df = preds[[pred_col, target_col]].copy()
    df = df.reset_index()
    df["year"] = df["period"].astype(str).str[:4].astype(int)

    yearly = df.groupby("year").agg(
        mean_pred=(pred_col, "mean"),
        actual_rate=(target_col, lambda x: x.dropna().mean()),
        n=(pred_col, "size"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly["year"], yearly["mean_pred"],
            color=COLOR_ONS if "ons" in label.lower() else COLOR_INC,
            lw=2, marker="o", label="Mean predicted risk")
    ax.plot(yearly["year"], yearly["actual_rate"],
            color="gray", lw=1.5, linestyle="--",
            marker="s", markersize=4, label="Actual positive rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate")
    ax.set_title(f"Predicted Risk vs Actual Rate Over Time — {label}",
                 fontweight="semibold")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"predictions_over_time_{label.lower()}.png", dpi=200)
    plt.show()
    print(f"Saved: outputs/figures/predictions_over_time_{label.lower()}.png")

    print(f"\nYearly summary ({label}):")
    print(yearly.round(3).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Model Insights ===\n")

    X, inc_df, ons_df, inc_preds, ons_preds, \
        estimators_inc, estimators_ons, panel = load_all()

    feature_names = X.columns.tolist()

    # 1. Feature importance
    print("\n--- Feature importance ---")
    imp_inc = feature_importance(estimators_inc, feature_names, "inc")
    imp_ons = feature_importance(estimators_ons, feature_names, "ons")
    plot_feature_importance(imp_inc, imp_ons)
    plot_group_importance(imp_inc, imp_ons)

    # 2. Performance by region (onset only — the interesting one)
    print("\n--- Performance by region ---")
    performance_by_region(
        ons_preds, panel,
        pred_col="preds_ons", target_col="ons_target",
        label="Onset"
    )

    # 3. Fold stability
    print("\n--- Fold stability ---")
    fold_stability(estimators_inc, X, "Incidence")
    fold_stability(estimators_ons, X, "Onset")

    # 4. False negative / positive analysis
    print("\n--- False negative analysis ---")
    false_negative_analysis(
        ons_preds, panel,
        pred_col="preds_ons", target_col="ons_target",
        threshold=0.106
    )

    # 5. Predictions over time
    print("\n--- Predictions over time ---")
    predictions_over_time(ons_preds, "preds_ons", "ons_target", "Onset")
    predictions_over_time(inc_preds, "preds_inc", "inc_target", "Incidence")

    print("\n=== Done. Check outputs/figures/ and outputs/diagnostics/ ===")


if __name__ == "__main__":
    main()
