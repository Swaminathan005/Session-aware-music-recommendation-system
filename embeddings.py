import numpy as np
import pandas as pd
import pickle
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score

# --- load ---
interactions_df = pd.read_csv("user_char.csv")
interactions_df = interactions_df.rename(columns={'track_id': 'song_id', 'playcount': 'play_count'})

metadata_df = pd.read_csv("initial_data.csv")
metadata_df = metadata_df.rename(columns={'track_id': 'song_id'})

# Keep interactions only for songs that have metadata
has_metadata = interactions_df['song_id'].isin(metadata_df['song_id'])

missing_song_ids = interactions_df.loc[~has_metadata, 'song_id'].nunique()
missing_interactions = (~has_metadata).sum()

if missing_interactions:
    print(
        f"Dropping {missing_interactions:,} interactions for "
        f"{missing_song_ids:,} songs missing from initial_data.csv."
    )
    interactions_df = interactions_df.loc[has_metadata].copy()
# --- parse tags: "rock, alternative, indie" -> ['rock', 'alternative', 'indie'] ---
metadata_df['tag_list'] = metadata_df['tags'].fillna('').apply(
    lambda s: [t.strip() for t in s.split(',') if t.strip()]
)

# --- normalize duration_ms (year, danceability, energy, instrumentalness, liveness assumed already 0-1) ---
dur_min, dur_max = metadata_df['duration_ms'].min(), metadata_df['duration_ms'].max()
metadata_df['duration_ms'] = (metadata_df['duration_ms'] - dur_min) / (dur_max - dur_min)

CONTINUOUS_FEATURES = ['year', 'duration_ms', 'danceability', 'energy', 'instrumentalness', 'liveness']

# 1. Build vocabulary — artist tokens, tag tokens, and the continuous feature names themselves
artist_tokens = metadata_df['artist'].radd('artist:').unique().tolist()
tag_tokens = sorted({f'tag:{t}' for tags in metadata_df['tag_list'] for t in tags})

dataset = Dataset()
dataset.fit(
    users=interactions_df['user_id'].unique(),
    items=metadata_df['song_id'].unique(),
    item_features=artist_tokens + tag_tokens + CONTINUOUS_FEATURES
)

# 2. Interactions matrix — log-transform play_count so it's not dominated by outliers
interactions_df['weight'] = np.log1p(interactions_df['play_count'])
(interactions, weights) = dataset.build_interactions(
    (row.user_id, row.song_id, row.weight) for row in interactions_df.itertuples()
)

# 3. Item features — categorical tokens (weight 1.0) + continuous audio features (their actual value)
def build_feature_dict(row):
    feats = {f'artist:{row.artist}': 1.0}
    feats.update({f'tag:{t}': 1.0 for t in row.tag_list})
    for col in CONTINUOUS_FEATURES:
        feats[col] = float(getattr(row, col))
    return feats

item_features = dataset.build_item_features(
    (row.song_id, build_feature_dict(row)) for row in metadata_df.itertuples()
)
from scipy.sparse import coo_matrix

# 4. Train/test split
train, test = random_train_test_split(
    interactions,
    test_percentage=0.2,
    random_state=42
)

# Rebuild weights in exactly the same entry order as `train`
weight_values = np.asarray(
    weights.tocsr()[train.row, train.col]
).ravel()

train_weights = coo_matrix(
    (weight_values, (train.row, train.col)),
    shape=train.shape
)

# 5. Train
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

# 6. Sanity check before trusting the embeddings
print("Test AUC:", auc_score(model, test, item_features=item_features).mean())

# 7. Extract ONLY the fused song embeddings — never touch get_user_representations()
item_biases, item_embeddings = model.get_item_representations(features=item_features)
print(item_embeddings.shape)  # (n_songs, 64)

# 8. Save embeddings + the ID mapping (needed to translate song_id <-> row index later)
_, _, item_id_map, _ = dataset.mapping()
np.save('song_embeddings.npy', item_embeddings)

with open('song_id_map.pkl', 'wb') as f:
    pickle.dump(item_id_map, f)

# also save the duration normalization bounds — needed to normalize a NEW song's
# duration_ms the same way before scoring it against these embeddings later
with open('duration_norm.pkl', 'wb') as f:
    pickle.dump({'min': dur_min, 'max': dur_max}, f)