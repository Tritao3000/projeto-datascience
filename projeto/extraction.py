import pandas as pd
import numpy as np

# Data reading
tracks = pd.read_csv("tracks.csv")
albums = pd.read_csv("albums.csv")
artists = pd.read_csv("artists.csv")
train_labels = pd.read_csv("train_labels.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Procesing of non-numeric variables (album_type, release_date and artist_genres)
# release_date (becomes 2 separate variables, year and month)
albums["release_date"] = pd.to_datetime(albums["release_date"], errors="coerce")
albums["release_year"] = albums["release_date"].dt.year
albums["release_month"] = albums["release_date"].dt.month

# album_type (becomes n separate binary columns with n being each type of album (album, single, compilation...)
# 1- belongs to that category)
albums = pd.get_dummies(albums, columns=["album_type"], dtype=int)

# artist_genres (counts how many genres an artist has)
artists["artist_genres"] = artists["artist_genres"].fillna("") # turns missing genres into an empty string

artists["n_genres"] = artists["artist_genres"].apply(
    lambda x: 0 if str(x).strip() == "" else len([g.strip() for g in str(x).split("|") if g.strip()])
)

# Datasets merge (training)
train_dataframe = train_labels.merge(tracks, on = "track_id", how = "left") # left to keep the samples defined in 
# train_labels, on "track_id" to match rows with track_id in both tables
train_dataframe = train_dataframe.merge(albums, on = "album_id", how = "left")
train_dataframe = train_dataframe.merge(artists, on = "artist_id", how = "left")


# Test 
test_dataframe = sample_submission[["track_id"]].merge(tracks, on="track_id", how = "left") # only track_id is selected 
# since "explicit" is all set to false
test_dataframe = test_dataframe.merge(albums, on="album_id", how = "left")
test_dataframe = test_dataframe.merge(artists, on = "artist_id", how = "left")


# removal of non-numeric columns
train_df_numeric= train_dataframe.drop(columns = ["explicit"]).select_dtypes(include = [np.number]) # keeps only numeric 
# variables and also removes explicit since it is what we want to predict
test_df_numeric = test_dataframe.select_dtypes(include =[np.number])


