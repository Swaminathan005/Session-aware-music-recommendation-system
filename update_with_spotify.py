#!/usr/bin/env python3
"""
update_with_spotify.py

1. Pull recent listening data from Spotify (via Spotipy).
2. Map Spotify tracks to the MSD (initial_data.csv);
   for unmatched tracks enrich the catalogue with Spotify metadata.
3. (Re)train LightFM on the combined item-feature matrix and the
   interaction matrix built from your listening history.
4. Derive:
   * item (song) embeddings for the combined space,
   * a user vector (average of listened‑song vectors, weighted by play count),
   * mappings to keep track of IDs.
5. Save the results for downstream steps (session modelling, alignment,
   freshness, etc.).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# 0️⃣  Helper: ensure spotipy is installed (should already be)
# ----------------------------------------------------------------------
def _ensure_spotipy():
    try:
        import spotipy  # noqa: F401
    except ImportError:
        print("Spotipy not found – installing …")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spotipy"])
        print("Spotipy installed.")

_ensure_spotipy()

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ----------------------------------------------------------------------
# 1️⃣  Configuration – hardcoded credentials (provided by user)
# ----------------------------------------------------------------------
CLIENT_ID     = "edc9f067830e4d83a12c5ae6e8dde46e"
CLIENT_SECRET = "7af2df8249734f9cb85191c2eaba770e"
REDIRECT_URI  = "http://127.0.0.1:8888/callback"   # must match what you set in the Dashboard
SCOPE         = "user-read-recently-played"

# ----------------------------------------------------------------------
# 2️⃣  Authenticate with Spotify
# ----------------------------------------------------------------------
auth_manager = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=".spotify_cache",
    show_dialog=True,          # forces login prompt each run (useful while testing)
)
sp = spotipy.Spotify(auth_manager=auth_manager)

# ----------------------------------------------------------------------
# 3️⃣  Load the MSD catalogue (initial_data.csv)
# ----------------------------------------------------------------------
msd_path = Path("initial_data.csv")
if not msd_path.is_file():
    raise FileNotFoundError(f"Cannot find {msd_path.resolve()}")
msd = pd.read_csv(msd_path)

# Normalise keys for matching: lower‑case, stripped
msd["lookup_key"] = (
    msd["name"].astype(str).str.lower().str.strip()
    + "||"
    + msd["artist"].astype(str).str.lower().str.strip()
)
msd_lookup = dict(zip(msd["lookup_key"], msd["track_id"]))   # key → MSD track_id

# ----------------------------------------------------------------------
# 4️⃣  Pull your recent listening history (played_at timestamps)
# ----------------------------------------------------------------------
def fetch_recently_played(limit_per_call=50, total_wanted=500):
    """Return up to `total_wanted` recently played tracks (newest first)."""
    results = []
    after = None                     # None → start with the most recent
    while len(results) < total_wanted:
        this_lim = min(limit_per_call, total_wanted - len(results))
        kwargs = {"limit": this_lim}
        if after is not None:
            kwargs["after"] = after   # Unix ms timestamp
        page = sp.current_user_recently_played(**kwargs)
        items = page["items"]
        if not items:
            break
        results.extend(items)
        # set `after` to the timestamp of the oldest item we just got
        oldest = items[-1]
        after = int(pd.to_datetime(oldest["played_at"]).value // 1_000_000)  # to ms
        # be polite to the API
        import time; time.sleep(0.15)
    return results

raw_plays = fetch_recently_played(total_wanted=800)   # adjust as you like
print(f"Fetched {len(raw_plays)} recent plays from Spotify")

# ----------------------------------------------------------------------
# 5️⃣  Build a DataFrame of listening events
# ----------------------------------------------------------------------
play_rows = []
for entry in raw_plays:
    trk = entry["track"]
    played_at = entry["played_at"]
    play_rows.append({
        "spotify_id": trk["id"],
        "track_name": trk["name"],
        "artist_name": trk["artists"][0]["name"],
        "played_at": played_at,
    })
listening_df = pd.DataFrame(play_rows)
listening_df["played_at"] = pd.to_datetime(listening_df["played_at"])

# ----------------------------------------------------------------------
# 6️⃣  Try to match each Spotify track to the MSD
# ----------------------------------------------------------------------
def normalise(s):  # helper for matching
    return str(s).lower().strip()

def match_to_msd(track_name, artist_name):
    key = normalise(track_name) + "||" + normalise(artist_name)
    return msd_lookup.get(key)          # returns MSD track_id or None

matched = []      # (msd_track_id, played_at, weight)
unmatched = []    # (spotify_id, track_name, artist_name, played_at)

for _, row in listening_df.iterrows():
    msd_id = match_to_msd(row["track_name"], row["artist_name"])
    if msd_id is not None:
        matched.append((msd_id, row["played_at"], 1.0))   # weight = 1 per play
    else:
        unmatched.append((
            row["spotify_id"],
            row["track_name"],
            row["artist_name"],
            row["played_at"],
        ))

print(f"Matched to MSD:   {len(matched)}")
print(f"Unmatched (Spotify‑only): {len(unmatched)}")

# ----------------------------------------------------------------------
# 7️⃣  Enrich unmatched tracks with Spotify metadata (audio features + artist genres)
# ----------------------------------------------------------------------
def enrich_spotify_track(spotify_id, track_name, artist_name, played_at):
    """Return a dict with the same columns as initial_data.csv for a Spotify‑only track."""
    # Audio features (danceability, energy, …, loudness, key, mode, tempo, duration_ms)
    feats = sp.audio_features([spotify_id])[0]
    if feats is None:
        # fallback: try to get at least duration & popularity from the track endpoint
        trk = sp.track(spotify_id)
        feats = {
            "danceability": None,
            "energy": None,
            "key": None,
            "loudness": None,
            "mode": None,
            "speechiness": None,
            "acacetone": None,
            "instrumentalness": None,
            "liveness": None,
            "valence": None,
            "tempo": None,
            "duration_ms": trk["duration_ms"],
        }
        popularity = trk["popularity"]
        album_release = trk["album"]["release_date"]
    else:
        popularity = feats.pop("popularity")
        album_release = None  # audio_features doesn't contain release date; get from track
        trk = sp.track(spotify_id)
        album_release = trk["album"]["release_date"]

    # Artist info – we want genres as "tags"
    artist_info = sp.artist(sp.track(spotify_id)["artists"][0]["id"])
    genres = ", ".join(artist_info.get("genres", []))   # empty string if none

    # Year from release_date (format: YYYY-MM-DD)
    try:
        year = pd.to_datetime(album_release).year
    except Exception:
        year = 0

    return {
        "track_id": spotify_id,               # we keep the Spotify ID as the item key
        "name": track_name,
        "artist": artist_name,
        "tags": genres,                       # treat genres as tags
        "year": year,
        "duration_ms": feats["duration_ms"],
        "danceability": feats["danceability"],
        "energy": feats["energy"],
        "instrumentalness": feats["instrumentalness"],
        "liveness": feats["liveness"],
        # the original MSD also had these columns (already 0‑1 normalized);
        # we will later normalise the continuous features globally.
    }

new_items = []
for sid, tname, artist, played_at in unmatched:
    new_items.append(enrich_spotify_track(sid, tname, artist, played_at))

new_items_df = pd.DataFrame(new_items) if new_items else pd.DataFrame(
    columns=["track_id","name","artist","tags","year","duration_ms",
             "danceability","energy","instrumentalness","liveness"]
)

# ----------------------------------------------------------------------
# 8️⃣  Build the combined item table (MSD + Spotify‑only)
# ----------------------------------------------------------------------
# Ensure both dataframes share the same column order
base_cols = ["track_id","name","artist","tags","year","duration_ms",
             "danceability","energy","instrumentalness","liveness"]
msd_clean = msd[base_cols].copy()
msd_clean["source"] = "msd"

if not new_items_df.empty:
    new_items_df["source"] = "spotify"
    items_all = pd.concat([msd_clean, new_items_df], ignore_index=True)
else:
    items_all = msd_clean.copy()
    items_all["source"] = "msd"

# ----------------------------------------------------------------------
# 9️⃣  Normalise the continuous audio features (0‑1) across the whole set
# ----------------------------------------------------------------------
cont_cols = ["year","duration_ms","danceability","energy","instrumentalness","liveness"]
# For year we treat the raw year; you may prefer to min‑max scale it as well.
for col in cont_cols:
    col_min = items_all[col].min()
    col_max = items_all[col].max()
    if col_max > col_min:
        items_all[col] = (items_all[col] - col_min) / (col_max - col_min)
    else:
        items_all[col] = 0.0   # all values identical → set to 0

# ----------------------------------------------------------------------
# 🔟  Build the interaction matrix (user × item) from your listening history
# ----------------------------------------------------------------------
# Get your Spotify user ID
me = sp.current_user()
user_id = me["id"]
print(f"Authenticated as Spotify user: {user_id} (display_name: {me['display_name']})")

# Aggregate plays per (user_id, item_id) – weight = number of plays
interact_rows = []
# matched MSD tracks
for msd_id, played_at, _ in matched:
    interact_rows.append((user_id, msd_id, 1.0))
# unmatched Spotify‑only tracks (use Spotify ID as item key)
for sid, _, _, played_at in unmatched:
    interact_rows.append((user_id, sid, 1.0))

interact_df = pd.DataFrame(interact_rows, columns=["user_id","item_id","weight"])
# Sum weights per (user, item) – this gives total play count
interact_agg = interact_df.groupby(["user_id","item_id"], as_index=False).agg(
    weight=("weight","sum")
)
# Apply log1p weighting as in the original embeddings.py
interact_agg["weight"] = np.log1p(interact_agg["weight"].values)

print(f"Unique items in interaction matrix: {interact_agg['item_id'].nunique()}")

# ----------------------------------------------------------------------
# 1️⃣1️⃣  (Re)train LightFM on the combined data
# ----------------------------------------------------------------------
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import coo_matrix

dataset = Dataset()
# Fit on users (we only have one) and all items (track_id from items_all)
dataset.fit(
    users=interact_agg["user_id"].unique(),
    items=items_all["track_id"].unique(),
)

# Build interaction matrix
(interact_matrix, weights) = dataset.build_interactions(
    (row.user_id, row.item_id, row.weight) for row in interact_agg.itertuples()
)

# Build item‑feature matrix (same feature set as embeddings.py)
def build_item_feature_dict(row):
    feats = {}
    # artist token(s)
    for a in str(row["artist"]).split(","):
        a = a.strip()
        if a:
            feats[f"artist:{a}"] = 1.0
    # tag tokens (genres from Spotify)
    for t in str(row["tags"]).split(","):
        t = t.strip()
        if t:
            feats[f"tag:{t}"] = 1.0
    # continuous audio features (already 0‑1 normalised)
    for col in cont_cols:
        feats[col] = float(row[col])
    return feats

item_features = dataset.build_item_features(
    (row["track_id"], build_item_feature_dict(row)) for row in items_all.itertuples()
)

# Train/test split (just to report AUC)
train, test = random_train_test_split(interact_matrix, test_percentage=0.2, random_state=42)

# Re‑build weights for the training set
train_weights = coo_matrix(
    (weights.tocsr()[train.row, train.col],
     (train.row, train.col)),
    shape=train.shape
)

model = LightFM(
    loss='warp',
    no_components=64,
    item_alpha=1e-6,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    train,
    item_features=item_features,
    sample_weight=train_weights,
    epochs=30,
    num_threads=4
)

test_auc = auc_score(model, test, item_features=item_features).mean()
print(f"LightFM test AUC: {test_auc:.4f}")

# ----------------------------------------------------------------------
# 1️⃣2️⃣  Extract item (song) embeddings for the combined space
# ----------------------------------------------------------------------
_, item_embeddings = model.get_item_representations(features=item_features)
print(f"Item embeddings shape: {item_embeddings.shape}")   # (n_items, 64)

# ----------------------------------------------------------------------
# 1️⃣3️⃣  Build a user vector (weighted average of listened‑song vectors)
# ----------------------------------------------------------------------
# Map item_id → row index in the embedding matrix
_, _, item_id_map, _ = dataset.mapping()   # returns (user_id_map, item_id_map, ...)
item_id_to_idx = item_id_map               # dict: item_id -> row index

# Gather vectors for items the user interacted with
user_weights = interact_agg.set_index("item_id")["weight"]   # item_id -> weight
valid_items = user_weights.index.intersection(item_id_to_idx.keys())
if len(valid_items) == 0:
    raise RuntimeError("No overlapping items between your listening history and the model.")

idx_list = [item_id_to_idx[i] for i in valid_items]
weight_list = user_weights.loc[valid_items].values
selected_vecs = item_embeddings[idx_list]                  # shape (n_valid, 64)

# Weighted average
user_vector = np.average(selected_vecs, axis=0, weights=weight_list)
print(f"User vector shape: {user_vector.shape}")

# ----------------------------------------------------------------------
# 1️⃣4️⃣  Save everything you need for downstream steps
# ----------------------------------------------------------------------
out_dir = Path(".")
out_dir.mkdir(exist_ok=True)

# Item embeddings and ID map
np.save(out_dir / "song_embeddings_combined.npy", item_embeddings)
with open(out_dir / "song_id_map_combined.pkl", "wb") as f:
    pickle.dump(item_id_to_idx, f)

# User vector
np.save(out_dir / "user_vector.npy", user_vector)

# Combined item table (for reference)
items_all.to_csv(out_dir / "items_all_combined.csv", index=False)

# Optional: also save the raw listening events with matched item IDs
listening_with_match = listening_df.copy()
listening_with_match["matched_msd_id"] = [
    match_to_msd(row["track_name"], row["artist_name"])
    for _, row in listening_df.iterrows()
]
listening_with_match.to_csv(out_dir / "listening_events_with_match.csv", index=False)

print("\n=== Saved files ===")
print(f"  {out_dir / 'song_embeddings_combined.npy'}")
print(f"  {out_dir / 'song_id_map_combined.pkl'}")
print(f"  {out_dir / 'user_vector.npy'}")
print(f"  {out_dir / 'items_all_combined.csv'}")
print(f"  {out_dir / 'listening_events_withmatch.csv'}")
print("\nYou can now load these files in your session‑modelling / alignment scripts.")