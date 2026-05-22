"""Compare GTD rolling forecasts with and without text features.

This extends the evaluation notebook by running a feature ablation:
  1. full model with conflictforecast.org topic-stock text features
  2. baseline model with the same history/macro/political features but no text

The rolling forecast setup intentionally matches src/05_rolling_forecast.py:
PanelSplit, 24 monthly test splits, 3-month horizon gap, and the same random
forest parameters.

Run from HW1_terrorism:
    python src/07_compare_text_ablation.py

Outputs:
    data/processed/gtd_text_ablation_preds_inc.csv
    data/processed/gtd_text_ablation_preds_ons.csv
    outputs/figures/text_ablation_roc_pr_curves.png
    outputs/diagnostics/text_ablation_performance_summary.csv
    outputs/diagnostics/text_ablation_key_numbers.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from panelsplit.application import cross_val_fit_predict
from panelsplit.cross_validation import PanelSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PARQUET = PROJECT_ROOT / "data/processed/gtd_features.parquet"
FEATURES_CSV = PROJECT_ROOT / "data/processed/gtd_features.csv"
INC_PARQUET = PROJECT_ROOT / "data/processed/gtd_target_inc.parquet"
ONS_PARQUET = PROJECT_ROOT / "data/processed/gtd_target_ons.parquet"
ALL_FEATURES_CSV = PROJECT_ROOT / "data/processed/gtd_panel_with_all_features.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
DIAG_DIR = PROJECT_ROOT / "outputs/diagnostics"

TOPIC_PREFIX = "stock_topic_"
SINCE_COL = "n_attacks_strict_since_0"

N_SPLITS = 24
TEST_SIZE = 1
HORIZON = 3
GAP = HORIZON - 1

RF_PARAMS = dict(
    max_depth=4,
    max_features=0.2,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=42,
)

MODEL_SPECS = {
    "with_text": {
        "label": "With text topics",
        "pred_col": "pred_with_text",
        "color": "#2f6f73",
    },
    "without_text": {
        "label": "Without text topics",
        "pred_col": "pred_without_text",
        "color": "#b45f45",
    },
}


def read_feature_matrix() -> pd.DataFrame:
    """Read features, using CSV if parquet support is unavailable."""
    if FEATURES_PARQUET.exists():
        try:
            X = pd.read_parquet(FEATURES_PARQUET)
            return normalize_multiindex(X)
        except Exception as exc:
            print(f"Could not read parquet features ({exc}); falling back to CSV.")

    X = pd.read_csv(FEATURES_CSV)
    X["period"] = X["period"].astype(int)
    X["iso3"] = X["iso3"].astype(str)
    X = X.set_index(["iso3", "period"]).sort_index()
    return X


def normalize_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(["iso3", "period"])
    if df.index.names != ["iso3", "period"]:
        df.index = df.index.set_names(["iso3", "period"])
    df = df.sort_index()
    df.index = pd.MultiIndex.from_arrays(
        [
            df.index.get_level_values("iso3").astype(str),
            df.index.get_level_values("period").astype(int),
        ],
        names=["iso3", "period"],
    )
    return df


def targets_from_panel_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate target files from the all-features CSV when parquet is unavailable."""
    df = pd.read_csv(ALL_FEATURES_CSV)
    df["month"] = pd.to_datetime(df["month"])
    df = df.dropna(subset=[f"{TOPIC_PREFIX}0"]).copy()
    df["period"] = df["month"].dt.year * 100 + df["month"].dt.month
    df["iso3"] = df["iso3"].astype(str)
    base = df.set_index(["iso3", "period"]).sort_index()

    inc = base[["incidence_next_month_t"]].rename(
        columns={"incidence_next_month_t": "inc_target"}
    )
    ons = base[["onset_next_3months_t"]].rename(
        columns={"onset_next_3months_t": "ons_target"}
    )
    return inc, ons


def read_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    if INC_PARQUET.exists() and ONS_PARQUET.exists():
        try:
            inc = normalize_multiindex(pd.read_parquet(INC_PARQUET))
            ons = normalize_multiindex(pd.read_parquet(ONS_PARQUET))
            return inc, ons
        except Exception as exc:
            print(f"Could not read parquet targets ({exc}); deriving targets from CSV.")

    return targets_from_panel_csv()


def make_feature_sets(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    topic_cols = [col for col in X.columns if col.startswith(TOPIC_PREFIX)]
    if not topic_cols:
        raise ValueError("No stock_topic_* columns found; cannot run text ablation.")

    return {
        "with_text": X.copy(),
        "without_text": X.drop(columns=topic_cols),
    }


def run_rolling_forecast(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target_col: str,
) -> np.ndarray:
    ps = PanelSplit(
        periods=y.index.get_level_values("period"),
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        gap=GAP,
    )

    X_aligned = X.reindex(y.index)
    if not X_aligned.index.equals(y.index):
        raise ValueError("Feature and target indices do not match after reindexing.")

    preds, _ = cross_val_fit_predict(
        estimator=RandomForestClassifier(**RF_PARAMS),
        X=X_aligned,
        y=y[target_col],
        cv=ps,
        method="predict_proba",
        drop_na_in_y=True,
    )
    return preds[:, 1]


def build_prediction_frame(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target_col: str,
    target_name: str,
) -> pd.DataFrame:
    ps = PanelSplit(
        periods=y.index.get_level_values("period"),
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        gap=GAP,
    )
    labels_df = y.copy()
    if SINCE_COL in X.columns:
        labels_df = labels_df.merge(
            X[[SINCE_COL]], left_index=True, right_index=True, how="left"
        )
    final_preds = ps.gen_test_labels(labels_df).copy()

    feature_sets = make_feature_sets(X)
    for model_key, X_model in feature_sets.items():
        label = MODEL_SPECS[model_key]["label"]
        pred_col = MODEL_SPECS[model_key]["pred_col"]
        print(f"  Running {target_name}: {label} ({X_model.shape[1]} features)")
        t0 = time.time()
        final_preds[pred_col] = run_rolling_forecast(X_model, y, target_col)
        elapsed = (time.time() - t0) / 60
        print(f"    done in {elapsed:.1f} minutes")

    return final_preds


def load_or_build_predictions(
    X: pd.DataFrame,
    inc: pd.DataFrame,
    ons: pd.DataFrame,
    reuse_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inc_out = PROCESSED_DIR / "gtd_text_ablation_preds_inc.csv"
    ons_out = PROCESSED_DIR / "gtd_text_ablation_preds_ons.csv"

    if reuse_predictions and inc_out.exists() and ons_out.exists():
        print("Reusing existing text-ablation prediction files.")
        inc_preds = pd.read_csv(inc_out, index_col=[0, 1])
        ons_preds = pd.read_csv(ons_out, index_col=[0, 1])
        return normalize_multiindex(inc_preds), normalize_multiindex(ons_preds)

    print("Running rolling ablation forecasts.")
    inc_preds = build_prediction_frame(X, inc, "inc_target", "incidence")
    ons_preds = build_prediction_frame(X, ons, "ons_target", "onset")

    inc_preds.to_csv(inc_out)
    ons_preds.to_csv(ons_out)
    print(f"Saved: {inc_out.relative_to(PROJECT_ROOT)}")
    print(f"Saved: {ons_out.relative_to(PROJECT_ROOT)}")
    return inc_preds, ons_preds


def curve_metrics(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
) -> dict[str, object]:
    clean = df.dropna(subset=[target_col, pred_col]).copy()
    y_true = clean[target_col].astype(int)
    y_score = clean[pred_col]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return {
        "n": len(clean),
        "positive_rate": y_true.mean(),
        "auc_roc": auc(fpr, tpr),
        "avg_precision": average_precision_score(y_true, y_score),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def evaluate_predictions(
    inc_preds: pd.DataFrame,
    ons_preds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    rows = []
    metrics = {}
    targets = [
        ("incidence", "Incidence: attack next month", inc_preds, "inc_target"),
        ("onset", "Onset: attack in next 3 months", ons_preds, "ons_target"),
    ]

    for target_key, target_label, df, target_col in targets:
        for model_key, spec in MODEL_SPECS.items():
            metric = curve_metrics(df, target_col, spec["pred_col"])
            metrics[(target_key, model_key)] = metric
            rows.append(
                {
                    "target": target_label,
                    "model": spec["label"],
                    "valid_rows": metric["n"],
                    "positive_rate": metric["positive_rate"],
                    "auc_roc": metric["auc_roc"],
                    "average_precision": metric["avg_precision"],
                    "ap_skill_vs_random": metric["avg_precision"]
                    / metric["positive_rate"],
                }
            )

    summary = pd.DataFrame(rows)
    deltas = []
    for target_key, target_label, _, _ in targets:
        with_text = summary[
            (summary["target"] == target_label)
            & (summary["model"] == MODEL_SPECS["with_text"]["label"])
        ].iloc[0]
        without_text = summary[
            (summary["target"] == target_label)
            & (summary["model"] == MODEL_SPECS["without_text"]["label"])
        ].iloc[0]
        deltas.append(
            {
                "target": target_label,
                "model": "Text gain",
                "valid_rows": with_text["valid_rows"],
                "positive_rate": with_text["positive_rate"],
                "auc_roc": with_text["auc_roc"] - without_text["auc_roc"],
                "average_precision": with_text["average_precision"]
                - without_text["average_precision"],
                "ap_skill_vs_random": np.nan,
            }
        )
    summary = pd.concat([summary, pd.DataFrame(deltas)], ignore_index=True)
    return summary, metrics


def plot_curves(metrics: dict[tuple[str, str], dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    targets = [
        ("incidence", "Incidence: attack next month"),
        ("onset", "Onset: attack in next 3 months"),
    ]

    for col, (target_key, target_title) in enumerate(targets):
        ax = axes[0, col]
        for model_key, spec in MODEL_SPECS.items():
            m = metrics[(target_key, model_key)]
            ax.plot(
                m["fpr"],
                m["tpr"],
                color=spec["color"],
                lw=2,
                label=f"{spec['label']} (AUC={m['auc_roc']:.3f})",
            )
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
        ax.set_title(f"ROC curve - {target_title}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)

        ax = axes[1, col]
        baseline = metrics[(target_key, "with_text")]["positive_rate"]
        for model_key, spec in MODEL_SPECS.items():
            m = metrics[(target_key, model_key)]
            ax.plot(
                m["recall"],
                m["precision"],
                color=spec["color"],
                lw=2,
                label=f"{spec['label']} (AP={m['avg_precision']:.3f})",
            )
        ax.axhline(
            y=baseline,
            color="gray",
            linestyle="--",
            lw=1,
            label=f"Random ({baseline:.3f})",
        )
        ax.set_title(f"Precision-recall curve - {target_title}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)

    fig.suptitle(
        "Text Feature Ablation in Rolling Out-of-Sample Forecasts",
        fontsize=13,
        fontweight="semibold",
        y=1.01,
    )
    fig.tight_layout()
    out = FIGURE_DIR / "text_ablation_roc_pr_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def write_key_numbers(summary: pd.DataFrame) -> None:
    out = DIAG_DIR / "text_ablation_key_numbers.txt"
    lines = [
        "Text ablation key numbers",
        "=========================",
        "",
        "Same rolling forecast design as src/05_rolling_forecast.py:",
        f"n_splits={N_SPLITS}, test_size={TEST_SIZE}, horizon={HORIZON}, gap={GAP}",
        "",
    ]

    for target in summary["target"].drop_duplicates():
        target_rows = summary[summary["target"] == target]
        lines.append(target)
        for _, row in target_rows.iterrows():
            if row["model"] == "Text gain":
                lines.append(
                    f"  Text gain: delta AUC={row['auc_roc']:.3f}, "
                    f"delta AP={row['average_precision']:.3f}"
                )
            else:
                lines.append(
                    f"  {row['model']}: AUC={row['auc_roc']:.3f}, "
                    f"AP={row['average_precision']:.3f}, "
                    f"baseline={row['positive_rate']:.3f}, "
                    f"AP/random={row['ap_skill_vs_random']:.2f}x"
                )
        lines.append("")

    lines.extend(
        [
            "Report note:",
            "Use the ROC curves to discuss ranking/discrimination, and the",
            "precision-recall curves to discuss performance under class imbalance.",
            "The text-gain rows say whether the conflictforecast.org topic stocks",
            "add incremental signal beyond terrorism history, macro variables,",
            "democracy, and ongoing conflict.",
        ]
    )
    out.write_text("\n".join(lines))
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Skip model fitting if text-ablation prediction CSVs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 7: Text feature ablation ===")
    print(
        f"Rolling setup: n_splits={N_SPLITS}, test_size={TEST_SIZE}, "
        f"gap={GAP}, horizon={HORIZON}"
    )

    X = read_feature_matrix()
    inc, ons = read_targets()
    inc = inc.reindex(X.index)
    ons = ons.reindex(X.index)

    topic_cols = [col for col in X.columns if col.startswith(TOPIC_PREFIX)]
    print(f"Features: {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"Text features removed in ablation: {len(topic_cols)} topic stocks")

    inc_preds, ons_preds = load_or_build_predictions(
        X=X,
        inc=inc,
        ons=ons,
        reuse_predictions=args.reuse_predictions,
    )
    summary, metrics = evaluate_predictions(inc_preds, ons_preds)

    summary_out = DIAG_DIR / "text_ablation_performance_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved: {summary_out.relative_to(PROJECT_ROOT)}")
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_curves(metrics)
    write_key_numbers(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
