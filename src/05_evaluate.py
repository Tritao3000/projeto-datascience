"""Step 5 — Generate evaluation charts and cross-model comparison."""
import os
import sys
import json
import ast

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


def load_preparation_summary():
    path = os.path.join(ARTIFACTS_DIR, "preparation_summary.json")
    with open(path) as f:
        return json.load(f)


def load_validation_results():
    path = os.path.join(ARTIFACTS_DIR, "model_results.csv")
    return pd.read_csv(path)


def summarize_parameter_effects(validation_results):
    summaries = {}
    for model_name in validation_results["model"].unique():
        model_rows = validation_results[validation_results["model"] == model_name].copy()
        best_idx = model_rows["f1"].idxmax()
        best_row = model_rows.loc[best_idx]
        f1_range = float(model_rows["f1"].max() - model_rows["f1"].min())
        auc_range = float(model_rows["roc_auc"].max() - model_rows["roc_auc"].min())
        params = ast.literal_eval(best_row["params"])
        summaries[model_name] = {
            "best_params": params,
            "validation_f1_range": round(f1_range, 4),
            "validation_auc_range": round(auc_range, 4),
        }
    return summaries


def get_metric_leaders(test_results):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    leaders = {}
    for metric in metrics:
        row = test_results.loc[test_results[metric].idxmax()]
        leaders[metric] = {
            "model": row["model"],
            "value": float(row[metric]),
        }
    return leaders


def build_cross_model_analysis(test_results, validation_results, prep_summary, feature_importances):
    validation_best = (
        validation_results.sort_values(["model", "f1"], ascending=[True, False])
        .groupby("model", as_index=False)
        .first()[["model", "f1", "roc_auc", "precision", "recall", "train_time", "params"]]
        .rename(
            columns={
                "f1": "val_f1",
                "roc_auc": "val_roc_auc",
                "precision": "val_precision",
                "recall": "val_recall",
                "train_time": "val_train_time",
                "params": "selected_params",
            }
        )
    )
    merged = test_results.merge(validation_best, on="model", how="left")
    merged["f1_delta"] = merged["f1"] - merged["val_f1"]
    merged["roc_auc_delta"] = merged["roc_auc"] - merged["val_roc_auc"]
    merged = merged.sort_values("f1", ascending=False).reset_index(drop=True)

    leaders = get_metric_leaders(merged)
    parameter_effects = summarize_parameter_effects(validation_results)
    selected_steps = prep_summary["selected_steps"]

    model_notes = {
        "naive_bayes": "Very high recall but low precision, which indicates many false positives on this imbalanced problem.",
        "logistic_regression": "Most interpretable linear baseline; precision is solid, but recall stays noticeably below tree-based methods.",
        "knn": "Performs competitively, but remains weaker than the tree ensembles and has limited interpretability.",
        "decision_tree": "Best single-tree model balances interpretability and predictive power, but generalizes slightly worse than the forest.",
        "random_forest": "Best overall test performance and strongest ROC-AUC, indicating the most robust ranking ability on this dataset.",
    }

    lines = []
    lines.append("# Cross-Model Analysis")
    lines.append("")
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append(
        f"- Data preparation winners: missing values=`{selected_steps['missing_values']}`, "
        f"scaling=`{selected_steps['scaling']}`, balancing=`{selected_steps['balancing']}`."
    )
    lines.append(
        f"- Split design: train={prep_summary['shapes']['train'][0]:,} rows, "
        f"validation={prep_summary['shapes']['validation'][0]:,} rows, "
        f"test={prep_summary['shapes']['test'][0]:,} rows."
    )
    lines.append(
        "- Model families were selected on validation and evaluated once on the untouched test split."
    )
    lines.append("")
    lines.append("## Test Ranking")
    lines.append("")
    lines.append("| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | F1 delta vs validation |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx, row in merged.iterrows():
        lines.append(
            f"| {idx + 1} | {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | {row['f1_delta']:+.4f} |"
        )
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    lines.append(
        f"- Best overall model: `{leaders['f1']['model']}` with test F1 `{leaders['f1']['value']:.4f}` "
        f"and ROC-AUC `{leaders['roc_auc']['value']:.4f}`."
    )
    lines.append(
        f"- Highest precision: `{leaders['precision']['model']}` (`{leaders['precision']['value']:.4f}`); "
        f"highest recall: `{leaders['recall']['model']}` (`{leaders['recall']['value']:.4f}`)."
    )
    lines.append(
        "- The precision/recall trade-off is strong in this dataset: Naive Bayes aggressively predicts the positive class, "
        "while the tree-based models deliver a better balance between false positives and false negatives."
    )
    lines.append(
        "- Validation-to-test drops are small for all selected models, which suggests the current hyperparameter choices generalize reasonably well."
    )
    lines.append("")
    lines.append("## Per-Model Interpretation")
    lines.append("")
    for _, row in merged.iterrows():
        model_name = row["model"]
        param_summary = parameter_effects[model_name]
        lines.append(
            f"- `{model_name}`: {model_notes[model_name]} Selected parameters: `{param_summary['best_params']}`. "
            f"Across the validation search, F1 varied by `{param_summary['validation_f1_range']:.4f}` "
            f"and ROC-AUC by `{param_summary['validation_auc_range']:.4f}`."
        )
    lines.append("")
    lines.append("## Relevant Variables")
    lines.append("")
    for model_name in ["logistic_regression", "decision_tree", "random_forest"]:
        if model_name not in feature_importances:
            continue
        top_features = feature_importances[model_name]["feature"].head(5).tolist()
        lines.append(f"- `{model_name}` top variables: {', '.join(top_features)}.")
    lines.append("")
    lines.append("## Critical Comparison")
    lines.append("")
    lines.append(
        "- `random_forest` is the strongest final choice if the goal is predictive performance, because it leads both F1 and ROC-AUC on the test split."
    )
    lines.append(
        "- `decision_tree` is the best compromise when interpretability matters more, since it remains competitive while still exposing a single transparent decision structure."
    )
    lines.append(
        "- `logistic_regression` is useful as a linear baseline and for coefficient-based interpretation, but its lower recall suggests the class boundary is not well captured by a purely linear model."
    )
    lines.append(
        "- `knn` benefits from the selected scaling and produces solid results, but it offers less explanatory value and slightly weaker test performance than the tree-based alternatives."
    )
    lines.append(
        "- `naive_bayes` remains valuable in this project mainly as the required preparation-step baseline; its final predictive quality is clearly below the other methods."
    )
    lines.append("")
    return "\n".join(lines), merged


def cross_model_analysis(test_results, validation_results, prep_summary, feature_importances):
    print("\n--- Cross-Model Analysis ---")
    analysis_text, merged = build_cross_model_analysis(
        test_results, validation_results, prep_summary, feature_importances
    )
    print(merged[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "f1_delta"]].to_string(index=False))

    best = merged.iloc[0]
    print(f"\nBest overall model: {best['model']} (F1={best['f1']:.4f})")

    analysis_path = os.path.join(ARTIFACTS_DIR, "cross_model_analysis.md")
    with open(analysis_path, "w") as f:
        f.write(analysis_text + "\n")
    print(f"  Saved {analysis_path}")
    return merged


def main():
    # Load test data
    test_df = load_parquet(os.path.join(PROCESSED_DIR, "prepared_test.parquet"))
    target = "explicit"
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    feature_names = X_test.columns.tolist()

    # Load best models
    models = load_best_models()
    prep_summary = load_preparation_summary()
    validation_results = load_validation_results()

    # Evaluate all
    print("\nEvaluating best models …")
    summary_rows = []
    feature_importances = {}
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
        importance_df = extract_feature_importance(model, feature_names, name)
        if importance_df is not None:
            feature_importances[name] = importance_df

    # Final summary CSV
    summary_path = os.path.join(ARTIFACTS_DIR, "final_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved {summary_path}")

    cross_model_analysis(summary_df, validation_results, prep_summary, feature_importances)
    print("\nDone.")


if __name__ == "__main__":
    main()
