"""Rolling forecast for GTD terrorism onset.

Adapted directly from Ben Seimon's prediction_solution.py.
Uses PanelSplit for rolling cross-validation and RandomForestClassifier,
exactly as in the course infrastructure.

Reads:
  data/processed/gtd_features.parquet
  data/processed/gtd_target_inc.parquet
  data/processed/gtd_target_ons.parquet

Writes:
  data/processed/gtd_preds_inc.csv
  data/processed/gtd_preds_ons.csv
  data/processed/fitted_estimators_inc.pkl
  data/processed/fitted_estimators_ons.pkl
"""

from __future__ import annotations
from pathlib import Path
import pickle
import time
import pandas as pd
from functools import reduce
from panelsplit.cross_validation import PanelSplit
from panelsplit.application import cross_val_fit_predict
from sklearn.ensemble import RandomForestClassifier

FEATURES_PATH = Path("data/processed/gtd_features.parquet")
INC_PATH      = Path("data/processed/gtd_target_inc.parquet")
ONS_PATH      = Path("data/processed/gtd_target_ons.parquet")
OUTPUT_DIR    = Path("data/processed")

# PanelSplit parameters — same as Ben's solution
N_SPLITS  = 24
TEST_SIZE = 1
HORIZON   = 3
GAP       = HORIZON - 1   # = 2, same as class

# RandomForest hyperparameters — same as Ben's solution
RF_PARAMS = dict(
    max_depth=4,
    max_features=0.2,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=42,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X      = pd.read_parquet(FEATURES_PATH)
    inc_df = pd.read_parquet(INC_PATH)
    ons_df = pd.read_parquet(ONS_PATH)

    print(f"Features:  {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"Inc target: {inc_df['inc_target'].notna().sum():,} valid rows")
    print(f"Ons target: {ons_df['ons_target'].notna().sum():,} valid rows")
    return X, inc_df, ons_df


def run_incidence(
    X: pd.DataFrame,
    inc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    print("\n--- Running incidence prediction ---")

    ps = PanelSplit(
        periods=inc_df.index.get_level_values("period"),
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        gap=GAP,
    )

    # Add since variable to labels df for gen_test_labels (matches Ben's approach)
    since_col = "n_attacks_strict_since_0"
    if since_col in X.columns:
        labels_df = inc_df.merge(
            X[[since_col]], left_index=True, right_index=True, how="left"
        )
    else:
        labels_df = inc_df.copy()

    final_preds = ps.gen_test_labels(labels_df)

    # Align features index to target index
    X_aligned = X.reindex(inc_df.index)
    assert X_aligned.index.equals(inc_df.index), \
        "Feature and target indices do not match."

    t0 = time.time()
    preds, estimators = cross_val_fit_predict(
        estimator=RandomForestClassifier(**RF_PARAMS),
        X=X_aligned,
        y=inc_df["inc_target"],
        cv=ps,
        method="predict_proba",
        drop_na_in_y=True,
    )
    elapsed = (time.time() - t0) / 60
    print(f"Incidence done in {elapsed:.1f} minutes")

    final_preds["preds_inc"] = preds[:, 1]
    return final_preds, estimators


def run_onset(
    X: pd.DataFrame,
    ons_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    print("\n--- Running onset prediction ---")

    ps = PanelSplit(
        periods=ons_df.index.get_level_values("period"),
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        gap=GAP,
    )

    since_col = "n_attacks_strict_since_0"
    if since_col in X.columns:
        labels_df = ons_df.merge(
            X[[since_col]], left_index=True, right_index=True, how="left"
        )
    else:
        labels_df = ons_df.copy()

    final_preds = ps.gen_test_labels(labels_df)

    X_aligned = X.reindex(ons_df.index)
    assert X_aligned.index.equals(ons_df.index), \
        "Feature and target indices do not match."

    t0 = time.time()
    preds, estimators = cross_val_fit_predict(
        estimator=RandomForestClassifier(**RF_PARAMS),
        X=X_aligned,
        y=ons_df["ons_target"],
        cv=ps,
        method="predict_proba",
        drop_na_in_y=True,
    )
    elapsed = (time.time() - t0) / 60
    print(f"Onset done in {elapsed:.1f} minutes")

    final_preds["preds_ons"] = preds[:, 1]
    return final_preds, estimators


def save_outputs(
    inc_preds: pd.DataFrame,
    ons_preds: pd.DataFrame,
    inc_estimators: list,
    ons_estimators: list,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inc_preds.to_csv(OUTPUT_DIR / "gtd_preds_inc.csv")
    ons_preds.to_csv(OUTPUT_DIR / "gtd_preds_ons.csv")

    with open(OUTPUT_DIR / "fitted_estimators_inc.pkl", "wb") as f:
        pickle.dump(inc_estimators, f)
    with open(OUTPUT_DIR / "fitted_estimators_ons.pkl", "wb") as f:
        pickle.dump(ons_estimators, f)

    print(f"\nSaved: data/processed/gtd_preds_inc.csv  ({len(inc_preds):,} rows)")
    print(f"Saved: data/processed/gtd_preds_ons.csv  ({len(ons_preds):,} rows)")
    print(f"Saved: fitted estimator pickles")


def main() -> None:
    print("=== Step 5: Rolling forecast (PanelSplit + RandomForest) ===\n")
    print(f"Parameters: n_splits={N_SPLITS}, test_size={TEST_SIZE}, "
          f"gap={GAP}, horizon={HORIZON}")

    X, inc_df, ons_df = load_data()

    inc_preds, inc_estimators = run_incidence(X, inc_df)
    ons_preds, ons_estimators = run_onset(X, ons_df)

    save_outputs(inc_preds, ons_preds, inc_estimators, ons_estimators)

    print("\nDone. Next step: src/06_evaluate.py")


if __name__ == "__main__":
    main()
