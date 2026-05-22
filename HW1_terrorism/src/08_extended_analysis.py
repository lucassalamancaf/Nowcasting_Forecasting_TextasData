"""Extended analysis: ablation study, hard onset, and multi-model comparison.

Runs three additional analyses:
  1. Ablation: RandomForest WITHOUT text features (topics) to quantify text contribution
  2. Hard onset: evaluate predictions on violence_since_0 > 60 months subset
     (following Ben's class notebooks which use violence_since_0 > N as hard onset filter)
  3. Multi-model comparison: RandomForest vs LogisticRegression vs GradientBoosting
     on onset target

Run from HW1_terrorism/ directory.
"""

from __future__ import annotations
from pathlib import Path
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
from panelsplit.cross_validation import PanelSplit
from panelsplit.application import cross_val_fit_predict

OUTPUT_DIR = Path("outputs/figures")
DIAG_DIR   = Path("outputs/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_COLS = [f"stock_topic_{i}" for i in range(15)]
N_SPLITS   = 24
TEST_SIZE  = 1
GAP        = 2

COLOR_RF   = "#2f6f73"
COLOR_LR   = "#e9c46a"
COLOR_GB   = "#e76f51"
COLOR_ABL  = "#adb5bd"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
})


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data():
    X       = pd.read_parquet("data/processed/gtd_features.parquet")
    ons_df  = pd.read_parquet("data/processed/gtd_target_ons.parquet")
    inc_df  = pd.read_parquet("data/processed/gtd_target_inc.parquet")

    # Load existing onset predictions for hard onset analysis
    ons_preds = pd.read_csv("data/processed/gtd_preds_ons.csv", index_col=[0, 1])

    # Load since variable from features for hard onset filter
    since_col = "n_attacks_strict_since_0"

    print(f"Features: {X.shape}")
    print(f"Onset target valid rows: {ons_df['ons_target'].notna().sum():,}")
    return X, ons_df, inc_df, ons_preds, since_col


def run_cv(estimator, X, y, label=""):
    """Run PanelSplit rolling CV and return merged predictions."""
    ps = PanelSplit(
        periods=y.index.get_level_values("period"),
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        gap=GAP,
    )
    final_preds = ps.gen_test_labels(y)

    t0 = time.time()
    preds, estimators = cross_val_fit_predict(
        estimator=estimator,
        X=X.reindex(y.index),
        y=y.iloc[:, 0],
        cv=ps,
        method="predict_proba",
        drop_na_in_y=True,
    )
    elapsed = (time.time() - t0) / 60
    print(f"  {label}: done in {elapsed:.1f} min")

    target_col = y.columns[0]
    final_preds["preds"] = preds[:, 1]
    return final_preds, estimators


def compute_metrics(df, target_col, pred_col):
    valid = df[[target_col, pred_col]].dropna()
    if valid[target_col].nunique() < 2:
        return None
    fpr, tpr, _ = roc_curve(valid[target_col], valid[pred_col])
    prec, rec, _ = precision_recall_curve(valid[target_col], valid[pred_col])
    return {
        "auc_roc": auc(fpr, tpr),
        "avg_precision": average_precision_score(valid[target_col], valid[pred_col]),
        "baseline": valid[target_col].mean(),
        "n_valid": len(valid),
        "fpr": fpr, "tpr": tpr,
        "prec": prec, "rec": rec,
    }


# ── 1. Ablation study ─────────────────────────────────────────────────────────

def run_ablation(X, ons_df):
    print("\n=== 1. Ablation study: with vs without text features ===")

    X_no_text = X.drop(columns=[c for c in TOPIC_COLS if c in X.columns])
    print(f"  Full features: {X.shape[1]} | No-text features: {X_no_text.shape[1]}")

    rf = RandomForestClassifier(
        max_depth=4, max_features=0.2, min_samples_leaf=100,
        n_jobs=-1, random_state=42
    )

    print("  Running WITH text...")
    preds_full, _ = run_cv(rf, X, ons_df, "Full model")

    print("  Running WITHOUT text...")
    preds_notext, _ = run_cv(rf, X_no_text, ons_df, "No-text model")

    m_full    = compute_metrics(preds_full,    "ons_target", "preds")
    m_notext  = compute_metrics(preds_notext,  "ons_target", "preds")

    print(f"\n  Full model:    AUC={m_full['auc_roc']:.3f}  AP={m_full['avg_precision']:.3f}")
    print(f"  No-text model: AUC={m_notext['auc_roc']:.3f}  AP={m_notext['avg_precision']:.3f}")
    print(f"  Text lift:     ΔAUC={m_full['auc_roc']-m_notext['auc_roc']:+.3f}  "
          f"ΔAP={m_full['avg_precision']-m_notext['avg_precision']:+.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (fpr_f, tpr_f, fpr_n, tpr_n, title) in enumerate([
        (m_full["fpr"], m_full["tpr"],
         m_notext["fpr"], m_notext["tpr"], "ROC"),
    ]):
        axes[0].plot(m_full["fpr"],   m_full["tpr"],   color=COLOR_RF,
                     lw=2, label=f"With text (AUC={m_full['auc_roc']:.3f})")
        axes[0].plot(m_notext["fpr"], m_notext["tpr"], color=COLOR_ABL,
                     lw=2, linestyle="--", label=f"No text (AUC={m_notext['auc_roc']:.3f})")
        axes[0].plot([0,1],[0,1], color="lightgray", lw=1)
        axes[0].set_title("ROC Curve — Onset")
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].legend(frameon=False)

    axes[1].plot(m_full["rec"],   m_full["prec"],   color=COLOR_RF,
                 lw=2, label=f"With text (AP={m_full['avg_precision']:.3f})")
    axes[1].plot(m_notext["rec"], m_notext["prec"], color=COLOR_ABL,
                 lw=2, linestyle="--", label=f"No text (AP={m_notext['avg_precision']:.3f})")
    axes[1].axhline(m_full["baseline"], color="lightgray", lw=1,
                    label=f"Baseline ({m_full['baseline']:.3f})")
    axes[1].set_title("Precision-Recall Curve — Onset")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False)

    plt.suptitle("Ablation Study: Value of Text Features (LDA Topics)",
                 fontweight="semibold", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_text_features.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("  Saved: outputs/figures/ablation_text_features.png")

    return {
        "full": m_full,
        "no_text": m_notext,
        "preds_full": preds_full,
        "preds_notext": preds_notext,
    }


# ── 2. Hard onset analysis ────────────────────────────────────────────────────

def run_hard_onset(ons_preds, X, ons_df, hard_months=60):
    """Following Ben's class notebooks: evaluate on violence_since_0 > N subset."""
    print(f"\n=== 2. Hard onset analysis (since_0 > {hard_months} months) ===")

    since_col = "n_attacks_strict_since_0"

    # Add since variable to predictions
    preds_with_since = ons_preds.copy()
    preds_with_since["since_0"] = X[since_col]

    # Soft onset: all valid predictions (current approach)
    soft = preds_with_since[["ons_target","preds_ons"]].dropna()

    # Hard onset: only country-months with long clean spells
    hard = preds_with_since[
        preds_with_since["since_0"] > hard_months
    ][["ons_target","preds_ons"]].dropna()

    print(f"  Soft onset rows: {len(soft):,} | positives: {int(soft['ons_target'].sum())}")
    print(f"  Hard onset rows: {len(hard):,} | positives: {int(hard['ons_target'].sum())}")

    if hard["ons_target"].nunique() < 2:
        print("  Not enough variation in hard onset subset to compute metrics.")
        print("  (This illustrates the 'hard problem of prediction' — too few positive cases)")
        return

    m_soft = compute_metrics(soft, "ons_target", "preds_ons")
    m_hard = compute_metrics(hard, "ons_target", "preds_ons")

    print(f"\n  Soft onset: AUC={m_soft['auc_roc']:.3f}  AP={m_soft['avg_precision']:.3f}  "
          f"baseline={m_soft['baseline']:.3f}")
    print(f"  Hard onset: AUC={m_hard['auc_roc']:.3f}  AP={m_hard['avg_precision']:.3f}  "
          f"baseline={m_hard['baseline']:.3f}")
    print(f"  AUC drop:   {m_hard['auc_roc'] - m_soft['auc_roc']:+.3f}  "
          f"(hard problem confirmed if negative)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(m_soft["fpr"], m_soft["tpr"], color=COLOR_RF,
                 lw=2, label=f"Soft onset (AUC={m_soft['auc_roc']:.3f}, n={len(soft):,})")
    axes[0].plot(m_hard["fpr"], m_hard["tpr"], color=COLOR_GB,
                 lw=2, linestyle="--",
                 label=f"Hard onset >{hard_months}mo (AUC={m_hard['auc_roc']:.3f}, n={len(hard):,})")
    axes[0].plot([0,1],[0,1], color="lightgray", lw=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(m_soft["rec"], m_soft["prec"], color=COLOR_RF,
                 lw=2, label=f"Soft onset (AP={m_soft['avg_precision']:.3f})")
    axes[1].plot(m_hard["rec"], m_hard["prec"], color=COLOR_GB,
                 lw=2, linestyle="--",
                 label=f"Hard onset (AP={m_hard['avg_precision']:.3f})")
    axes[1].axhline(m_soft["baseline"], color=COLOR_RF, lw=0.8, linestyle=":")
    axes[1].axhline(m_hard["baseline"], color=COLOR_GB, lw=0.8, linestyle=":")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False, fontsize=9)

    plt.suptitle(f"Hard Onset Analysis: Effect of Clean Spell Requirement ({hard_months} months)\n"
                 f"Illustrating the 'Hard Problem of Prediction'",
                 fontweight="semibold", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hard_onset_analysis.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("  Saved: outputs/figures/hard_onset_analysis.png")


# ── 3. Multi-model comparison ─────────────────────────────────────────────────

def run_model_comparison(X, ons_df):
    print("\n=== 3. Multi-model comparison (onset target) ===")

    models = {
        "Random Forest\n(Ben's model)": RandomForestClassifier(
            max_depth=4, max_features=0.2, min_samples_leaf=100,
            n_jobs=-1, random_state=42
        ),
        "Logistic Regression\n(L2 regularised)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, max_iter=500, random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.05,
            min_samples_leaf=50, random_state=42
        ),
    }

    colors = [COLOR_RF, COLOR_LR, COLOR_GB]
    results = {}

    for (name, model), color in zip(models.items(), colors):
        print(f"  Running {name.replace(chr(10),' ')}...")
        preds, _ = run_cv(model, X, ons_df, name.replace("\n"," "))
        m = compute_metrics(preds, "ons_target", "preds")
        results[name] = {"metrics": m, "color": color}

    # Summary table
    print("\n  Model comparison summary:")
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        skill = m["avg_precision"] / m["baseline"]
        rows.append({
            "Model": name.replace("\n"," "),
            "AUC-ROC": round(m["auc_roc"], 3),
            "Avg Precision": round(m["avg_precision"], 3),
            "Baseline": round(m["baseline"], 3),
            "Skill ratio": round(skill, 2),
        })
    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(DIAG_DIR / "model_comparison.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for name, res in results.items():
        m = res["metrics"]
        c = res["color"]
        axes[0].plot(m["fpr"], m["tpr"], color=c, lw=2,
                     label=f"{name.replace(chr(10),' ')} (AUC={m['auc_roc']:.3f})")
        axes[1].plot(m["rec"], m["prec"], color=c, lw=2,
                     label=f"{name.replace(chr(10),' ')} (AP={m['avg_precision']:.3f})")

    axes[0].plot([0,1],[0,1], color="lightgray", lw=1, label="Random")
    axes[0].set_title("ROC Curve — Onset (3-month)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    baseline = list(results.values())[0]["metrics"]["baseline"]
    axes[1].axhline(baseline, color="lightgray", lw=1,
                    label=f"Random ({baseline:.3f})")
    axes[1].set_title("Precision-Recall Curve — Onset (3-month)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False, fontsize=8)

    plt.suptitle("Model Comparison: Random Forest vs Logistic Regression vs Gradient Boosting\n"
                 "Onset prediction, 24-fold rolling cross-validation",
                 fontweight="semibold", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("  Saved: outputs/figures/model_comparison.png")

    return df_summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Extended Analysis ===\n")
    X, ons_df, inc_df, ons_preds, since_col = load_data()

    ablation_results = run_ablation(X, ons_df)
    run_hard_onset(ons_preds, X, ons_df, hard_months=60)
    model_summary = run_model_comparison(X, ons_df)

    print("\n=== Summary ===")
    print("Files saved to outputs/figures/:")
    print("  ablation_text_features.png")
    print("  hard_onset_analysis.png")
    print("  model_comparison.png")
    print("  outputs/diagnostics/model_comparison.csv")


if __name__ == "__main__":
    main()
