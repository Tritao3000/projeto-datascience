"""Step 3 — Stepwise NB-gated data preparation."""
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import (
    PROCESSED_DIR, ARTIFACTS_DIR, RANDOM_STATE, TEST_SIZE, CFG,
)
from src.utils.io import save_parquet, load_parquet
from src.utils.metrics import evaluate_classifier

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_evaluate_nb(X_train, y_train, X_test, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return evaluate_classifier(nb, X_test, y_test)


def compare_alternatives(results_dict, metric="f1"):
    """Print comparison table and return the key of the winner."""
    print(f"\n  {'Alternative':<25s} {'Accuracy':>9s} {'Precision':>9s} "
          f"{'Recall':>9s} {'F1':>9s} {'ROC-AUC':>9s}")
    print("  " + "-" * 72)
    for name, m in results_dict.items():
        print(f"  {name:<25s} {m['accuracy']:9.4f} {m['precision']:9.4f} "
              f"{m['recall']:9.4f} {m['f1']:9.4f} {m['roc_auc']:9.4f}")
    winner = max(results_dict, key=lambda k: results_dict[k][metric])
    print(f"  >> Winner: {winner} (best {metric}={results_dict[winner][metric]:.4f})")
    return winner


# ---------------------------------------------------------------------------
# Preparation steps
# ---------------------------------------------------------------------------

def apply_missing_values(X_train, X_test, strategy):
    if strategy == "median":
        imp = SimpleImputer(strategy="median")
    elif strategy == "knn":
        imp = KNNImputer(n_neighbors=5)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    X_tr = pd.DataFrame(imp.fit_transform(X_train),
                         columns=X_train.columns, index=X_train.index)
    X_te = pd.DataFrame(imp.transform(X_test),
                         columns=X_test.columns, index=X_test.index)
    return X_tr, X_te


def apply_scaling(X_train, X_test, scaler_type):
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")
    X_tr = pd.DataFrame(scaler.fit_transform(X_train),
                         columns=X_train.columns, index=X_train.index)
    X_te = pd.DataFrame(scaler.transform(X_test),
                         columns=X_test.columns, index=X_test.index)
    return X_tr, X_te


def apply_balancing(X_train, y_train, strategy):
    if strategy == "oversample":
        # Random oversample minority class
        majority = y_train.value_counts().idxmax()
        minority = y_train.value_counts().idxmin()
        n_majority = (y_train == majority).sum()
        idx_min = y_train[y_train == minority].index
        oversampled = idx_min.to_series().sample(
            n=n_majority - len(idx_min), replace=True,
            random_state=RANDOM_STATE
        )
        new_idx = X_train.index.tolist() + oversampled.tolist()
        return X_train.loc[new_idx].reset_index(drop=True), y_train.loc[new_idx].reset_index(drop=True)
    elif strategy == "undersample":
        # Random undersample majority class
        majority = y_train.value_counts().idxmax()
        minority = y_train.value_counts().idxmin()
        n_minority = (y_train == minority).sum()
        idx_maj = y_train[y_train == majority].index
        undersampled = idx_maj.to_series().sample(
            n=n_minority, replace=False, random_state=RANDOM_STATE
        )
        idx_min = y_train[y_train == minority].index
        new_idx = undersampled.tolist() + idx_min.tolist()
        return X_train.loc[new_idx].reset_index(drop=True), y_train.loc[new_idx].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    df = load_parquet(os.path.join(PROCESSED_DIR, "base_numeric.parquet"))
    target = "explicit"
    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Train class balance: {y_train.value_counts(normalize=True).to_dict()}")

    all_results = []

    # ------------------------------------------------------------------
    # Step 1: Missing Values
    # ------------------------------------------------------------------
    print("\n=== STEP 1: Missing Values ===")
    has_missing = X_train.isnull().any().any()
    results_mv = {}

    if has_missing:
        # Baseline: drop rows with missing (for NB comparison)
        mask_tr = X_train.dropna().index
        mask_te = X_test.dropna().index
        results_mv["baseline_dropna"] = train_evaluate_nb(
            X_train.loc[mask_tr], y_train.loc[mask_tr],
            X_test.loc[mask_te], y_test.loc[mask_te],
        )

        for alt in CFG["preparation"]["missing_values"]["alternatives"]:
            print(f"  Trying {alt} imputation …")
            X_tr_imp, X_te_imp = apply_missing_values(X_train, X_test, alt)
            results_mv[alt] = train_evaluate_nb(X_tr_imp, y_train, X_te_imp, y_test)

        winner_mv = compare_alternatives(results_mv)

        # Apply winner
        if winner_mv == "baseline_dropna":
            X_train = X_train.loc[mask_tr]
            y_train = y_train.loc[mask_tr]
            X_test = X_test.loc[mask_te]
            y_test = y_test.loc[mask_te]
        else:
            X_train, X_test = apply_missing_values(X_train, X_test, winner_mv)
    else:
        print("  No missing values — skipping.")
        results_mv["no_missing"] = train_evaluate_nb(X_train, y_train, X_test, y_test)
        winner_mv = "no_missing"

    for name, m in results_mv.items():
        all_results.append({"step": "missing_values", "alternative": name, **m})

    # ------------------------------------------------------------------
    # Step 2: Scaling
    # ------------------------------------------------------------------
    print("\n=== STEP 2: Scaling ===")
    results_sc = {}
    results_sc["unscaled"] = train_evaluate_nb(X_train, y_train, X_test, y_test)

    for alt in CFG["preparation"]["scaling"]["alternatives"]:
        print(f"  Trying {alt} scaling …")
        X_tr_sc, X_te_sc = apply_scaling(X_train, X_test, alt)
        results_sc[alt] = train_evaluate_nb(X_tr_sc, y_train, X_te_sc, y_test)

    winner_sc = compare_alternatives(results_sc)

    if winner_sc != "unscaled":
        X_train, X_test = apply_scaling(X_train, X_test, winner_sc)

    for name, m in results_sc.items():
        all_results.append({"step": "scaling", "alternative": name, **m})

    # ------------------------------------------------------------------
    # Step 3: Balancing (train only)
    # ------------------------------------------------------------------
    print("\n=== STEP 3: Balancing ===")
    results_bal = {}
    results_bal["unbalanced"] = train_evaluate_nb(X_train, y_train, X_test, y_test)

    for alt in CFG["preparation"]["balancing"]["alternatives"]:
        print(f"  Trying {alt} …")
        X_tr_bal, y_tr_bal = apply_balancing(X_train, y_train, alt)
        results_bal[alt] = train_evaluate_nb(X_tr_bal, y_tr_bal, X_test, y_test)

    winner_bal = compare_alternatives(results_bal)

    if winner_bal != "unbalanced":
        X_train, y_train = apply_balancing(X_train, y_train, winner_bal)

    for name, m in results_bal.items():
        all_results.append({"step": "balancing", "alternative": name, **m})

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\nSaving prepared datasets …")
    train_df = X_train.copy()
    train_df["explicit"] = y_train.values
    test_df = X_test.copy()
    test_df["explicit"] = y_test.values

    save_parquet(train_df, os.path.join(PROCESSED_DIR, "prepared_train.parquet"))
    save_parquet(test_df, os.path.join(PROCESSED_DIR, "prepared_test.parquet"))

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(ARTIFACTS_DIR, "preparation_results.csv")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"  Saved {results_path}")

    print("\n--- Final Preparation Summary ---")
    print(f"  Missing values winner: {winner_mv}")
    print(f"  Scaling winner:        {winner_sc}")
    print(f"  Balancing winner:      {winner_bal}")
    print(f"  Final train shape: {X_train.shape}")
    print(f"  Final test shape:  {X_test.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
