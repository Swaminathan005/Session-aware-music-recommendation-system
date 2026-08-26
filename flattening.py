import numpy as np
import pandas as pd
import pickle
import umap.umap_ as umap
from sklearn.preprocessing import normalize

# Load learned song vectors
embeddings = np.load("song_embeddings.npy")

with open("song_id_map.pkl", "rb") as f:
    song_id_map = pickle.load(f)

metadata = pd.read_csv("initial_data.csv").rename(
    columns={"track_id": "song_id"}
)

# Normalize before cosine-like similarity / UMAP
vectors = normalize(embeddings)

# Reduce 64 dimensions to 2
reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.15,
    metric="cosine",
    random_state=42
)

coordinates = reducer.fit_transform(vectors)

# Ensure metadata follows the exact embedding-row order
ordered_song_ids = [
    song_id for song_id, index in sorted(song_id_map.items(), key=lambda x: x[1])
]

plot_data = metadata.set_index("song_id").loc[ordered_song_ids].reset_index()

plot_data["x"] = coordinates[:, 0]
plot_data["y"] = coordinates[:, 1]

plot_data[["song_id", "name", "artist", "tags", "x", "y"]].to_csv(
    "song_embedding_2d.csv",
    index=False
)