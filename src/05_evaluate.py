"""Step 5 — Generate evaluation charts and cross-model comparison."""
import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import (
    PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR, FIGURES_DIR, CFG,
)
from src.utils.io import load_parquet, load_model, save_figure
from src.utils.metrics import evaluate_classifier


MODEL_NAMES = list(CFG["modeling"]["grids"].keys())


def load_best_models():
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}_best.joblib")
        models[name] = load_model(path)
    return models


def plot_all_confusion_matrices(models, X_test, y_test):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name.replace("_", " ").title())
    fig.suptitle("Confusion Matrices — Best Models", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, os.path.join(FIGURES_DIR, "confusion_matrices_all.png"))

    # Individual confusion matrices
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name.replace('_', ' ').title()}")
        save_figure(fig, os.path.join(FIGURES_DIR, f"confusion_matrix_{name}.png"))


def plot_all_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test).astype(float)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        label = name.replace("_", " ").title()
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    save_figure(fig, os.path.join(FIGURES_DIR, "roc_curves_comparison.png"))


def plot_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = results["model"].values
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, results[metric].values, width,
                       label=metric.replace("_", " ").title())
        for bar, val in zip(bars, results[metric].values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.set_xticks(x + width * 2)
    labels = [m.replace("_", " ").title() for m in models]
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(FIGURES_DIR, "model_comparison.png"))


def extract_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Save CSV
    csv_path = os.path.join(ARTIFACTS_DIR, f"feature_importance_{model_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # Plot
    top = df.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), top["importance"].values[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name.replace('_', ' ').title()}")
    fig.tight_layout()
    save_figure(fig, os.path.join(FIGURES_DIR, f"feature_importance_{model_name}.png"))

    return df


def cross_model_analysis(results):
    print("\n--- Cross-Model Analysis ---")
    print(results.to_string(index=False))

    best_idx = results["f1"].idxmax()
    best = results.loc[best_idx]
    print(f"\nBest overall model: {best['model']} (F1={best['f1']:.4f})")
    return results


def main():
    # Load test data
    test_df = load_parquet(os.path.join(PROCESSED_DIR, "prepared_test.parquet"))
    target = "explicit"
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    feature_names = X_test.columns.tolist()

    # Load best models
    models = load_best_models()

    # Evaluate all
    print("\nEvaluating best models …")
    summary_rows = []
    for name, model in models.items():
        metrics = evaluate_classifier(model, X_test, y_test)
        summary_rows.append({"model": name, **metrics})
        print(f"  {name:25s}  F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")

    summary_df = pd.DataFrame(summary_rows)

    # Generate all evaluation artifacts
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\nGenerating confusion matrices …")
    plot_all_confusion_matrices(models, X_test, y_test)

    print("Generating ROC curves …")
    plot_all_roc_curves(models, X_test, y_test)

    print("Generating model comparison chart …")
    plot_model_comparison(summary_df)

    print("Extracting feature importances …")
    for name, model in models.items():
        extract_feature_importance(model, feature_names, name)

    # Final summary CSV
    summary_path = os.path.join(ARTIFACTS_DIR, "final_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved {summary_path}")

    cross_model_analysis(summary_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
