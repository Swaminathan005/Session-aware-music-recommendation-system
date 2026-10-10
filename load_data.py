#!/usr/bin/env python3
"""
Load MSD embeddings and metadata into Redis and MongoDB.
Run after ensuring Redis and MongoDB are running locally on default ports.
"""

import os
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

# Redis
import redis
# MongoDB
from pymongo import MongoClient

# -------------------------- CONFIGURATION --------------------------
BASE_DIR = Path(__file__).parent
EMBEDDINGS_PATH = BASE_DIR / "song_embeddings.npy"
ID_MAP_PATH     = BASE_DIR / "song_id_map.pkl"
METADATA_PATH   = BASE_DIR / "initial_data.csv"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB_NAME = "music_rec"
MONGO_COLLECTION_SONGS = "songs"

USER_ID_PREFIX = "user1"  # not used here
# -----------------------------------------------------------------


def load_embeddings_and_ids():
    """Load embeddings and ID↔index maps from local files."""
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}")
    if not ID_MAP_PATH.is_file():
        raise FileNotFoundError(f"Missing {ID_MAP_PATH}")
    E = np.load(EMBEDDINGS_PATH)          # (n_songs, 64)
    with open(ID_MAP_PATH, 'rb') as f:
        id2idx = pickle.load(f)           # dict: song_id -> index
    idx2id = {v: k for k, v in id2idx.items()}
    n = E.shape[0]
    if set(id2idx.values()) != set(range(n)):
        raise ValueError("id2idx values are not a permutation of 0..n-1")
    return E, id2idx, idx2id


def load_metadata_csv():
    """Load metadata from CSV and return list of dicts for MongoDB."""
    import pandas as pd
    df = pd.read_csv(METADATA_PATH)
    # Ensure we have track_id column
    if 'track_id' not in df.columns:
        raise ValueError("CSV must contain 'track_id' column")
    records = []
    for _, row in df.iterrows():
        song_id = str(row['track_id'])
        name = str(row.get('name', ''))
        artist = str(row.get('artist', ''))
        # tags column appears to be a string of comma-separated genres with quotes?
        tags = row.get('tags', '')
        if isinstance(tags, str):
            # Remove surrounding quotes and split by ', '
            # Example: "rock, alternative, indie, alternative_rock, indie_rock, 00s"
            # It may already be a list-like string.
            # We'll split by ',' and strip spaces and quotes.
            raw = tags.strip()
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1]
            genres = [g.strip() for g in raw.split(',')] if raw else []
        else:
            genres = []
        # numeric fields with possible NaN
        def maybe_int(val):
            try:
                return int(val) if not pd.isna(val) else None
            except:
                return None
        def maybe_float(val):
            try:
                return float(val) if not pd.isna(val) else None
            except:
                return None
        year = maybe_int(row.get('year'))
        duration_ms = maybe_int(row.get('duration_ms'))
        # audio features we have
        danceability = maybe_float(row.get('danceability'))
        energy = maybe_float(row.get('energy'))
        instrumentalness = maybe_float(row.get('instrumentalness'))
        liveness = maybe_float(row.get('liveness'))
        # missing fields set to None
        valence = None
        tempo = None
        key = None
        loudness = None
        mode = None
        speechiness = None
        acousticness = None
        audio_features = {
            "danceability": danceability,
            "energy": energy,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "key": key,
            "loudness": loudness,
            "mode": mode,
            "speechiness": speechiness,
            "acousticness": acousticness,
        }
        source = "msd"
        now = datetime.now()
        record = {
            "song_id": song_id,
            "name": name,
            "artist": artist,
            "album": None,  # not in CSV
            "genres": genres,
            "year": year,
            "duration_ms": duration_ms,
            "audio_features": audio_features,
            "source": source,
            "created_at": now,
            "updated_at": now,
        }
        records.append(record)
    return records


def main():
    print("Connecting to Redis and MongoDB...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    mongo_client = MongoClient(host=MONGO_HOST, port=MONGO_PORT)
    db = mongo_client[MONGO_DB_NAME]
    coll = db[MONGO_COLLECTION_SONGS]

    # Optionally clear existing data (comment out if you want to keep)
    print("Clearing old Redis keys matching our patterns (optional)...")
    for pattern in ["song_emb:*", "user_vec:*", "sess_vec:*", "sess_hist:*",
                    "user_playcount:*", "top_n:*", "lock:*"]:
        for key in r.scan_iter(match=pattern):
            r.delete(key)
    print("Clearing MongoDB songs collection...")
    coll.delete_many({})

    print("Loading embeddings and IDs...")
    E, id2idx, idx2id = load_embeddings_and_ids()
    print(f"Loaded {E.shape[0]} songs with dimension {E.shape[1]}")

    print("Loading metadata from CSV...")
    metadata_records = load_metadata_csv()
    print(f"Loaded {len(metadata_records)} metadata records")

    # Insert embeddings into Redis
    print("Inserting embeddings into Redis...")
    pipe = r.pipeline()
    for song_id, idx in id2idx.items():
        vec = E[idx]  # numpy float64
        # Convert to raw bytes (little endian)
        blob = vec.astype(np.float64).tobytes()
        norm = float(np.linalg.norm(vec))
        # Store as hash
        pipe.hset(
            f"song_emb:{song_id}",
            mapping={
                "vec": blob,
                "norm": norm,
                "source": "msd",
                "updated_at": int(datetime.now().timestamp() * 1000),
            },
        )
    pipe.execute()
    print("Done inserting embeddings.")

    # Insert metadata into MongoDB
    print("Inserting metadata into MongoDB...")
    if metadata_records:
        coll.insert_many(metadata_records)
    print("Done inserting metadata.")

    print("Data loading complete.")
    # Close connections
    mongo_client.close()
    r.close()


if __name__ == "__main__":
    main()