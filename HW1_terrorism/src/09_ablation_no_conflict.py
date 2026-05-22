"""Ablation: remove UCDP ongoing_conflict feature and compare to full model.

Run from HW1_terrorism/ directory.
"""

from __future__ import annotations
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from panelsplit.cross_validation import PanelSplit
from panelsplit.application import cross_val_fit_predict

OUTPUT_DIR = Path("outputs/figures")
DIAG_DIR   = Path("outputs/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_COLS = [f"stock_topic_{i}" for i in range(15)]
N_SPLITS   = 24
TEST_SIZE  = 1
GAP        = 2

COLOR_FULL    = "#2f6f73"
COLOR_NOTEXT  = "#adb5bd"
COLOR_NOCONF  = "#e76f51"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
})

RF = RandomForestClassifier(
    max_depth=4, max_features=0.2, min_samples_leaf=100,
    n_jobs=-1, random_state=42
)


def run_cv(X, y, label=""):
    ps = PanelSplit(
        periods=y.index.get_level_values("period"),
        n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP,
    )
    final_preds = ps.gen_test_labels(y)
    t0 = time.time()
    preds, _ = cross_val_fit_predict(
        estimator=RF, X=X.reindex(y.index), y=y.iloc[:, 0],
        cv=ps, method="predict_proba", drop_na_in_y=True,
    )
    print(f"  {label}: done in {(time.time()-t0)/60:.1f} min")
    final_preds["preds"] = preds[:, 1]
    return final_preds


def metrics(df, target_col="ons_target", pred_col="preds"):
    valid = df[[target_col, pred_col]].dropna()
    fpr, tpr, _ = roc_curve(valid[target_col], valid[pred_col])
    prec, rec, _ = precision_recall_curve(valid[target_col], valid[pred_col])
    return {
        "auc": auc(fpr, tpr),
        "ap": average_precision_score(valid[target_col], valid[pred_col]),
        "baseline": valid[target_col].mean(),
        "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
    }


def main():
    print("=== Ablation: No conflict (UCDP) feature ===\n")

    X      = pd.read_parquet("data/processed/gtd_features.parquet")
    ons_df = pd.read_parquet("data/processed/gtd_target_ons.parquet")

    X_no_text   = X.drop(columns=[c for c in TOPIC_COLS if c in X.columns])
    X_no_conf   = X.drop(columns=["ongoing_conflict"], errors="ignore")

    print(f"Full features:        {X.shape[1]}")
    print(f"No-text features:     {X_no_text.shape[1]}")
    print(f"No-conflict features: {X_no_conf.shape[1]}")
    print()

    print("Running full model...")
    p_full    = run_cv(X,           ons_df, "Full")
    print("Running no-text model...")
    p_notext  = run_cv(X_no_text,   ons_df, "No text")
    print("Running no-conflict model...")
    p_noconf  = run_cv(X_no_conf,   ons_df, "No conflict")

    m_full   = metrics(p_full)
    m_notext = metrics(p_notext)
    m_noconf = metrics(p_noconf)

    # Summary
    print("\n=== Results ===")
    print(f"{'Model':<30} {'AUC-ROC':>8} {'Avg Prec':>10} {'Skill':>8}")
    print("-" * 60)
    for label, m in [("Full model", m_full),
                     ("No text (topics removed)", m_notext),
                     ("No conflict (UCDP removed)", m_noconf)]:
        skill = m["ap"] / m["baseline"]
        print(f"{label:<30} {m['auc']:>8.3f} {m['ap']:>10.3f} {skill:>7.2f}x")

    print(f"\nText lift:     ΔAUC={m_full['auc']-m_notext['auc']:+.3f}  "
          f"ΔAP={m_full['ap']-m_notext['ap']:+.3f}")
    print(f"Conflict lift: ΔAUC={m_full['auc']-m_noconf['auc']:+.3f}  "
          f"ΔAP={m_full['ap']-m_noconf['ap']:+.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for (m, color, label, ls) in [
        (m_full,   COLOR_FULL,   f"Full model (AUC={m_full['auc']:.3f})",           "-"),
        (m_notext, COLOR_NOTEXT, f"No text (AUC={m_notext['auc']:.3f})",             "--"),
        (m_noconf, COLOR_NOCONF, f"No conflict/UCDP (AUC={m_noconf['auc']:.3f})",   "-."),
    ]:
        axes[0].plot(m["fpr"], m["tpr"], color=color, lw=2,
                     linestyle=ls, label=label)
        axes[1].plot(m["rec"], m["prec"], color=color, lw=2,
                     linestyle=ls,
                     label=label.replace("AUC", "AP").replace(
                         f"{m['auc']:.3f}", f"{m['ap']:.3f}"))

    axes[0].plot([0,1],[0,1], color="lightgray", lw=1)
    axes[0].set_title("ROC Curve — Onset (3-month)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].axhline(m_full["baseline"], color="lightgray", lw=1,
                    label=f"Random ({m_full['baseline']:.3f})")
    axes[1].set_title("Precision-Recall Curve — Onset (3-month)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False, fontsize=9)

    plt.suptitle(
        "Feature Ablation Study: Full Model vs No Text vs No Conflict\n"
        "Onset prediction, 24-fold rolling cross-validation",
        fontweight="semibold", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_three_way.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("\nSaved: outputs/figures/ablation_three_way.png")


if __name__ == "__main__":
    main()
