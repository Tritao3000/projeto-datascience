"""Step 1 — Extract, join, and engineer features from raw Spotify data."""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import RAW_DIR, PROCESSED_DIR, RANDOM_STATE
from src.utils.io import save_parquet


def load_raw_data(raw_dir):
    print("Loading raw CSVs …")
    data = {}
    for name in ["tracks", "albums", "artists", "train_labels"]:
        path = os.path.join(raw_dir, f"{name}.csv")
        data[name] = pd.read_csv(path)
        print(f"  {name}: {data[name].shape}")
    return data


def join_tables(tracks, albums, artists, labels):
    print("Joining tables …")
    # tracks → labels (inner: only labeled rows)
    df = tracks.merge(labels, on="track_id", how="inner")
    print(f"  After tracks ⋈ labels: {df.shape}")
    # → albums
    df = df.merge(albums, on="album_id", how="left")
    print(f"  After ⋈ albums: {df.shape}")
    # → artists
    df = df.merge(artists, on="artist_id", how="left")
    print(f"  After ⋈ artists: {df.shape}")
    return df


def engineer_features(df):
    print("Engineering features …")

    # --- album_type one-hot ---
    if "album_type" in df.columns:
        dummies = pd.get_dummies(df["album_type"], prefix="album_type")
        df = pd.concat([df, dummies], axis=1)

    # --- release_date → year, month ---
    if "release_date" in df.columns:
        dt = pd.to_datetime(df["release_date"], errors="coerce", utc=True)
        df["release_year"] = dt.dt.year
        df["release_month"] = dt.dt.month

    # --- artist_genres → genre count + top-N binary flags ---
    # Genres are pipe-delimited strings, e.g. "pop|rock|indie"
    if "artist_genres" in df.columns:
        def _parse_genres(val):
            if pd.isna(val):
                return []
            return [g.strip() for g in str(val).split("|") if g.strip()]

        genres_series = df["artist_genres"].apply(_parse_genres)
        df["genre_count"] = genres_series.apply(len)

        # find top-N most frequent genres
        from collections import Counter
        all_genres = Counter()
        for glist in genres_series:
            all_genres.update(glist)
        top_genres = [g for g, _ in all_genres.most_common(20)]
        for g in top_genres:
            df[f"genre_{g}"] = genres_series.apply(lambda x, g=g: int(g in x))
        print(f"  Created {len(top_genres)} genre flags + genre_count")

    return df


def select_numeric(df):
    print("Selecting numeric features …")
    # Convert explicit target: True/False → 1/0
    df["explicit"] = df["explicit"].astype(int)

    # Drop text/id columns
    drop_cols = [
        "track_id", "track_name", "album_id", "album_name",
        "artist_id", "name", "label", "release_date",
        "album_type", "artist_genres",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number, "bool"])
    # Ensure bool columns are int
    for col in numeric_df.select_dtypes(include="bool").columns:
        numeric_df[col] = numeric_df[col].astype(int)

    print(f"  Final shape: {numeric_df.shape}")
    return numeric_df


def main():
    data = load_raw_data(RAW_DIR)
    merged = join_tables(
        data["tracks"], data["albums"], data["artists"], data["train_labels"]
    )
    featured = engineer_features(merged)
    numeric = select_numeric(featured)

    # Data dictionary
    print("\n--- Data Dictionary ---")
    for col in numeric.columns:
        print(f"  {col:30s}  non-null: {numeric[col].notna().sum():>7d}  "
              f"dtype: {numeric[col].dtype}  "
              f"unique: {numeric[col].nunique()}")

    out_path = os.path.join(PROCESSED_DIR, "base_numeric.parquet")
    save_parquet(numeric, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
