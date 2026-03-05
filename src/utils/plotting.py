"""Shared chart helpers."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from .io import save_figure


def plot_confusion_matrix(model, X_test, y_test, title, path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    save_figure(fig, path)


def plot_roc_curve(models_dict, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test).astype(float)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_feature_importance(importances, feature_names, title, path, top_n=20):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    save_figure(fig, path)


def plot_distributions(df, columns, path_prefix):
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(df[col].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_title(f"{col} — Histogram")
        axes[1].boxplot(df[col].dropna(), vert=True)
        axes[1].set_title(f"{col} — Boxplot")
        fig.suptitle(col)
        fig.tight_layout()
        save_figure(fig, f"{path_prefix}_{col}.png")


def plot_missing_values(df, path):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        ax.set_title("Missing Values")
    else:
        fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.3)))
        ax.barh(missing.index, missing.values)
        ax.set_xlabel("Count")
        ax.set_title("Missing Values by Feature")
        fig.tight_layout()
    save_figure(fig, path)
