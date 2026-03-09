"""Step 4 — Train all 5 classifiers with hyperparameter grids."""
import os
import sys
import time
import itertools
import ast

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import (
    PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR, FIGURES_DIR,
    RANDOM_STATE, CFG,
)
from src.utils.io import load_parquet, save_model, save_figure
from src.utils.metrics import evaluate_classifier

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


MODEL_CLASSES = {
    "naive_bayes": GaussianNB,
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}

# Map YAML param names → sklearn constructor args (most are identical)
EXTRA_KWARGS = {
    "logistic_regression": {"max_iter": 1000, "random_state": RANDOM_STATE, "l1_ratio": 0},
    "decision_tree": {"random_state": RANDOM_STATE},
    "random_forest": {"random_state": RANDOM_STATE, "n_jobs": -1},
}


def _param_combos(grid: dict):
    """Yield all combinations from a param grid dict (like sklearn ParameterGrid)."""
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def train_model(model_class, params, X_train, y_train, X_eval, y_eval, extra=None):
    kw = dict(params)
    if extra:
        kw.update(extra)
    model = model_class(**kw)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    metrics = evaluate_classifier(model, X_eval, y_eval)
    metrics["train_time"] = round(train_time, 2)
    return model, metrics


def run_grid_search(model_name, param_grid, X_train, y_train, X_val, y_val):
    model_class = MODEL_CLASSES[model_name]
    extra = EXTRA_KWARGS.get(model_name, {})
    results = []
    best_f1 = -1
    best_model = None

    combos = list(_param_combos(param_grid))
    print(f"\n  {model_name}: {len(combos)} combinations")

    for i, params in enumerate(combos, 1):
        model, metrics = train_model(
            model_class, params, X_train, y_train, X_val, y_val, extra
        )
        row = {"model": model_name, "params": str(params), **metrics}
        results.append(row)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
        if i % 5 == 0 or i == len(combos):
            print(f"    [{i}/{len(combos)}] best F1 so far: {best_f1:.4f}")

    return results, best_model


def plot_param_impact(results_df, model_name, param_grid, path):
    model_rows = results_df[results_df["model"] == model_name].copy()
    if model_rows.empty or not param_grid:
        return

    params_to_plot = list(param_grid.keys())
    n = len(params_to_plot)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, param in zip(axes, params_to_plot):
        # Extract param value from the params string
        vals = []
        for _, row in model_rows.iterrows():
            p = ast.literal_eval(row["params"])
            vals.append(str(p.get(param, "N/A")))
        model_rows = model_rows.copy()
        model_rows["_param_val"] = vals
        grouped = model_rows.groupby("_param_val")["f1"].mean()
        ax.bar(range(len(grouped)), grouped.values)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{model_name}: {param}")

    fig.suptitle(f"Parameter Impact — {model_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, path)


def main():
    # Load prepared data
    train_df = load_parquet(os.path.join(PROCESSED_DIR, "prepared_train.parquet"))
    val_df = load_parquet(os.path.join(PROCESSED_DIR, "prepared_val.parquet"))

    target = "explicit"
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]
    print(f"Train: {X_train.shape}  Validation: {X_val.shape}")

    grids = CFG["modeling"]["grids"]
    all_results = []
    best_models = {}

    for model_name, param_grid in grids.items():
        if param_grid is None:
            param_grid = {}
        results, best_model = run_grid_search(
            model_name, param_grid, X_train, y_train, X_val, y_val
        )
        all_results.extend(results)
        best_models[model_name] = best_model

        # Save best model
        model_path = os.path.join(MODELS_DIR, f"{model_name}_best.joblib")
        save_model(best_model, model_path)

        # Parameter impact plot
        if param_grid:
            results_df = pd.DataFrame(results)
            plot_param_impact(
                results_df, model_name, param_grid,
                os.path.join(FIGURES_DIR, f"param_impact_{model_name}.png"),
            )

    # Save all results
    results_df = pd.DataFrame(all_results)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    results_path = os.path.join(ARTIFACTS_DIR, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved {results_path}")

    print("\n--- Best Model per Family (selected on validation) ---")
    for model_name in grids:
        model_rows = results_df[results_df["model"] == model_name]
        best = model_rows.loc[model_rows["f1"].idxmax()]
        print(f"  {model_name:25s}  F1={best['f1']:.4f}  "
              f"AUC={best['roc_auc']:.4f}  time={best['train_time']:.1f}s  "
              f"params={best['params']}")
    print("Done.")


if __name__ == "__main__":
    main()
