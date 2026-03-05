"""Step 2 — Profile the base numeric dataset for the report."""
import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import PROCESSED_DIR, PROFILING_DIR, FIGURES_DIR
from src.utils.io import load_parquet, save_figure


def profile_dimensionality(df):
    target = "explicit"
    balance = df[target].value_counts(normalize=True).to_dict()
    return {
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "feature_names": [c for c in df.columns if c != target],
        "target": target,
        "class_balance": {str(k): round(v, 4) for k, v in balance.items()},
    }


def profile_distributions(df):
    stats = df.describe().T
    stats["skewness"] = df.skew(numeric_only=True)
    return stats


def profile_sparsity(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    zero_counts = (df == 0).sum()
    return pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
        "zero_count": zero_counts,
    })


def plot_class_balance(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["explicit"].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + len(df) * 0.01, f"{v:,}", ha="center", fontsize=10)
    ax.set_xlabel("Explicit")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Balance")
    save_figure(fig, path)


def plot_correlation_heatmap(df, path, max_features=25):
    # Select most varying numeric features for readability
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "explicit" in numeric_cols:
        numeric_cols.remove("explicit")
    # Take features with highest variance
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_cols = variances.head(max_features).index.tolist() + ["explicit"]
    corr = df[top_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Correlation Heatmap (top features by variance)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    save_figure(fig, path)


def plot_feature_distributions(df, path_prefix, max_features=15):
    numeric_cols = [c for c in df.columns if c != "explicit"]
    # Pick a representative subset: audio features + metadata
    priority = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_sec", "track_popularity", "album_popularity",
        "artist_popularity", "followers", "genre_count",
    ]
    cols = [c for c in priority if c in numeric_cols][:max_features]

    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=10)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, f"{path_prefix}_histograms.png")

    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(col, fontsize=10)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Boxplots", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, f"{path_prefix}_boxplots.png")


def plot_missing_bar(df, path):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Missing Values")
    else:
        fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.4)))
        pct = (missing / len(df) * 100)
        ax.barh(missing.index, pct.values, color="#DD8452")
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values by Feature")
        for i, (v, p) in enumerate(zip(missing.values, pct.values)):
            ax.text(p + 0.02, i, f"{v:,} ({p:.2f}%)", va="center", fontsize=8)
        fig.tight_layout()
    save_figure(fig, path)


def main():
    df = load_parquet(os.path.join(PROCESSED_DIR, "base_numeric.parquet"))

    # Profiling
    dim = profile_dimensionality(df)
    dist_stats = profile_distributions(df)
    sparsity = profile_sparsity(df)

    # Save JSON summary
    os.makedirs(PROFILING_DIR, exist_ok=True)
    summary = {
        "dimensionality": dim,
        "distribution_stats": dist_stats.to_dict(),
        "sparsity": sparsity.to_dict(),
    }
    summary_path = os.path.join(PROFILING_DIR, "profiling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved {summary_path}")

    # Plots
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating profiling plots …")
    plot_class_balance(df, os.path.join(FIGURES_DIR, "profiling_class_balance.png"))
    plot_correlation_heatmap(df, os.path.join(FIGURES_DIR, "profiling_correlation.png"))
    plot_feature_distributions(df, os.path.join(FIGURES_DIR, "profiling_distributions"))
    plot_missing_bar(df, os.path.join(FIGURES_DIR, "profiling_missing_values.png"))

    # Print summary
    print(f"\n--- Profiling Summary ---")
    print(f"  Rows: {dim['n_rows']:,}  Features: {dim['n_features']}")
    print(f"  Class balance: {dim['class_balance']}")
    missing_total = sparsity["missing_count"].sum()
    print(f"  Total missing cells: {int(missing_total):,}")
    print("Done.")


if __name__ == "__main__":
    main()
