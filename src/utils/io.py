"""I/O helpers for dataframes, models, and figures."""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def save_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  Saved {path}  ({df.shape[0]} rows, {df.shape[1]} cols)")


def load_parquet(path):
    df = pd.read_parquet(path)
    print(f"  Loaded {path}  ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved model → {path}")


def load_model(path):
    return joblib.load(path)


def save_figure(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {path}")
