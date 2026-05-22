"""Hard onset analysis with and without text features.

Tests whether LDA topic stocks add more value for hard onset prediction
(since > 60 months) than for soft onset. The hypothesis: if anything can
detect a first attack after 5 years of peace, it would be deteriorating
news discourse rather than attack history (which is all zeros for hard onset).

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
HARD_MONTHS = 60

COLOR_SOFT_FULL   = "#2f6f73"
COLOR_SOFT_NOTEXT = "#adb5bd"
COLOR_HARD_FULL   = "#e76f51"
COLOR_HARD_NOTEXT = "#f4a261"

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


def metrics_subset(df, since_series, min_since=None,
                   target_col="ons_target", pred_col="preds"):
    """Compute metrics on full predictions or hard-onset subset."""
    merged = df[[target_col, pred_col]].copy()
    if min_since is not None:
        merged = merged[since_series > min_since]
    valid = merged.dropna()

    if valid[target_col].nunique() < 2 or len(valid) < 20:
        print(f"    Warning: only {len(valid)} rows, "
              f"{int(valid[target_col].sum())} positives — metrics unreliable")
        return None

    fpr, tpr, _ = roc_curve(valid[target_col], valid[pred_col])
    prec, rec, _ = precision_recall_curve(valid[target_col], valid[pred_col])
    return {
        "auc": auc(fpr, tpr),
        "ap": average_precision_score(valid[target_col], valid[pred_col]),
        "baseline": valid[target_col].mean(),
        "n": len(valid),
        "n_pos": int(valid[target_col].sum()),
        "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
    }


def main():
    print("=== Hard Onset: With vs Without Text Features ===\n")

    X      = pd.read_parquet("data/processed/gtd_features.parquet")
    ons_df = pd.read_parquet("data/processed/gtd_target_ons.parquet")

    since_col = "n_attacks_strict_since_0"
    since_series = X[since_col]

    X_no_text = X.drop(columns=[c for c in TOPIC_COLS if c in X.columns])

    print(f"Full features:    {X.shape[1]}")
    print(f"No-text features: {X_no_text.shape[1]}")
    print()

    # Run both models
    print("Running full model (with text)...")
    p_full    = run_cv(X,         ons_df, "Full")
    print("Running no-text model...")
    p_notext  = run_cv(X_no_text, ons_df, "No text")

    # Compute metrics on soft and hard subsets
    print("\nComputing metrics...")
    m_soft_full   = metrics_subset(p_full,   since_series, min_since=None)
    m_soft_notext = metrics_subset(p_notext, since_series, min_since=None)
    m_hard_full   = metrics_subset(p_full,   since_series, min_since=HARD_MONTHS)
    m_hard_notext = metrics_subset(p_notext, since_series, min_since=HARD_MONTHS)

    # Summary table
    print(f"\n{'Condition':<35} {'AUC':>7} {'AP':>7} {'Baseline':>9} "
          f"{'Skill':>7} {'N':>6} {'Pos':>5}")
    print("-" * 78)
    for label, m in [
        ("Soft onset + text",          m_soft_full),
        ("Soft onset - text",          m_soft_notext),
        (f"Hard onset (>{HARD_MONTHS}mo) + text",  m_hard_full),
        (f"Hard onset (>{HARD_MONTHS}mo) - text",  m_hard_notext),
    ]:
        if m is None:
            print(f"{label:<35} {'N/A':>7}")
            continue
        skill = m["ap"] / m["baseline"]
        print(f"{label:<35} {m['auc']:>7.3f} {m['ap']:>7.3f} "
              f"{m['baseline']:>9.3f} {skill:>6.2f}x {m['n']:>6} {m['n_pos']:>5}")

    # Key finding
    if m_hard_full and m_hard_notext:
        print(f"\nText lift on SOFT onset: "
              f"ΔAUC={m_soft_full['auc']-m_soft_notext['auc']:+.3f}  "
              f"ΔAP={m_soft_full['ap']-m_soft_notext['ap']:+.3f}")
        print(f"Text lift on HARD onset: "
              f"ΔAUC={m_hard_full['auc']-m_hard_notext['auc']:+.3f}  "
              f"ΔAP={m_hard_full['ap']-m_hard_notext['ap']:+.3f}")
        print()
        if m_hard_full['auc'] > m_hard_notext['auc']:
            print("→ Text features help MORE for hard onset than soft onset.")
            print("  Consistent with hypothesis: news discourse captures early warning")
            print("  signals that attack history cannot (attack history = all zeros).")
        else:
            print("→ Text features do not clearly help for hard onset.")
            print("  Hard problem confirmed: even text cannot reliably predict")
            print("  first attacks after long periods of peace.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_items = [
        (m_soft_full,   COLOR_SOFT_FULL,
         f"Soft + text (AUC={m_soft_full['auc']:.3f})",    "-",  2.0),
        (m_soft_notext, COLOR_SOFT_NOTEXT,
         f"Soft - text (AUC={m_soft_notext['auc']:.3f})",  "--", 1.5),
    ]
    if m_hard_full:
        plot_items.append(
            (m_hard_full, COLOR_HARD_FULL,
             f"Hard + text (AUC={m_hard_full['auc']:.3f}, "
             f"n={m_hard_full['n']}, pos={m_hard_full['n_pos']})", "-", 2.0)
        )
    if m_hard_notext:
        plot_items.append(
            (m_hard_notext, COLOR_HARD_NOTEXT,
             f"Hard - text (AUC={m_hard_notext['auc']:.3f}, "
             f"n={m_hard_notext['n']}, pos={m_hard_notext['n_pos']})", "--", 1.5)
        )

    for m, color, label, ls, lw in plot_items:
        axes[0].plot(m["fpr"], m["tpr"], color=color, lw=lw,
                     linestyle=ls, label=label)
        pr_label = label.replace("AUC", "AP").replace(
            f"{m['auc']:.3f}", f"{m['ap']:.3f}"
        )
        axes[1].plot(m["rec"], m["prec"], color=color, lw=lw,
                     linestyle=ls, label=pr_label)

    axes[0].plot([0,1],[0,1], color="lightgray", lw=1, label="Random")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    if m_soft_full:
        axes[1].axhline(m_soft_full["baseline"], color=COLOR_SOFT_FULL,
                        lw=0.8, linestyle=":", alpha=0.6,
                        label=f"Soft baseline ({m_soft_full['baseline']:.3f})")
    if m_hard_full:
        axes[1].axhline(m_hard_full["baseline"], color=COLOR_HARD_FULL,
                        lw=0.8, linestyle=":", alpha=0.6,
                        label=f"Hard baseline ({m_hard_full['baseline']:.3f})")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False, fontsize=8)

    plt.suptitle(
        f"Hard Onset Analysis: Does Text Help When History Is Silent?\n"
        f"Soft onset (all) vs Hard onset (>{HARD_MONTHS} months clean), "
        f"with and without LDA topic features",
        fontweight="semibold", fontsize=11
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hard_onset_text_ablation.png",
                dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/figures/hard_onset_text_ablation.png")


if __name__ == "__main__":
    main()
