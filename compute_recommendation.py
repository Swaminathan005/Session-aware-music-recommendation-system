#!/usr/bin/env python3
"""
compute_recommendation.py

Utility to compute music recommendations using a blended session/user vector
via cosine similarity with pre‑computed song embeddings.

Usage:
    from compute_recommendation import get_recommendations
    recs = get_recommendations(s_vec, u_vec, top_n=5)

Or run as a script to see an example (requires the embedding files).
"""

import pickle
from pathlib import Path
import numpy as np


def load_embeddings(embed_path: Path, id_map_path: Path):
    """Load embeddings and ID↔index maps."""
    if not embed_path.is_file():
        raise FileNotFoundError(f"Missing {embed_path}")
    if not id_map_path.is_file():
        raise FileNotFoundError(f"Missing {id_map_path}")
    E = np.load(embed_path)                      # (N_songs, 64)
    with open(id_map_path, 'rb') as f:
        id2idx = pickle.load(f)                  # dict: item_id -> index
    idx2id = {v: k for k, v in id2idx.items()}
    # sanity check
    n = E.shape[0]
    if set(id2idx.values()) != set(range(n)):
        raise ValueError("id2idx values are not a permutation of 0..N_songs-1")
    return E, id2idx, idx2id


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return unit‑length vector (avoid div‑by‑zero)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    return v / norm


def get_recommendations(session_vector: np.ndarray,
                        user_vector: np.ndarray,
                        embeddings_path: Path = None,
                        id_map_path: Path = None,
                        top_n: int = 5,
                        lam: float = 0.70,
                        exclude_ids=None):
    """
    Compute top‑N song recommendations.

    Parameters
    ----------
    session_vector : np.ndarray, shape (64,)
        Current session vector (updated per song).
    user_vector : np.ndarray, shape (64,)
        Long‑term user vector (updated per session).
    embeddings_path : Path or str, optional
        Path to .npy file with song embeddings. Defaults to
        `song_embeddings.npy` beside this script.
    id_map_path : Path or str, optional
        Path to .pkl file with ID→index map. Defaults to
        `song_id_map.pkl` beside this script.
    top_n : int, default 5
        Number of recommendations to return.
    lam : float, default 0.70
        Weight for session vector in the blend (0..1).
    exclude_ids : iterable of song IDs, optional
        Song IDs to omit from recommendations (e.g., currently playing).

    Returns
    -------
    list of song IDs (as stored in the ID map) ordered by descending similarity.
    """
    # Resolve default paths relative to this file if not supplied
    if embeddings_path is None:
        embeddings_path = Path(__file__).parent / "song_embeddings.npy"
    if id_map_path is None:
        id_map_path = Path(__file__).parent / "song_id_map.pkl"

    E, id2idx, idx2id = load_embeddings(Path(embeddings_path), Path(id_map_path))

    # Blend vectors
    blended = lam * session_vector + (1 - lam) * user_vector
    blended_norm = _normalize(blended)

    # Normalize item vectors
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    E_norm = E / norms

    # Cosine similarity = dot product of normalized vectors
    scores = E_norm.dot(blended_norm)  # (N_songs,)

    if exclude_ids:
        mask = np.zeros(E.shape[0], dtype=bool)
        for sid in exclude_ids:
            if sid in id2idx:
                mask[id2idx[sid]] = True
        scores[mask] = -np.inf

    top_idx = np.argsort(-scores)[:top_n]
    return [idx2id[i] for i in top_idx]


# ----------------------------------------------------------------------
# Example usage when run directly
# ----------------------------------------------------------------------
def _demo():
    """Run a simple demo with zero vectors (returns arbitrary ordering)."""
    print("=== Recommendation Computation Demo ===")
    # Load data
    E, id2idx, idx2id = load_embeddings(
        Path(__file__).parent / "song_embeddings.npy",
        Path(__file__).parent / "song_id_map.pkl"
    )
    n_songs = E.shape[0]
    print(f"Loaded {n_songs} songs.")

    # Example vectors (replace with real session/user vectors)
    s_vec = np.zeros(E.shape[1], dtype=E.dtype)   # session vector
    u_vec = np.zeros(E.shape[1], dtype=E.dtype)   # user vector

    # Get recommendations
    recs = get_recommendations(s_vec, u_vec, top_n=5)
    print("Top‑5 song IDs (example with zero vectors):", recs)

    # Show what happens if we bias toward a random direction
    rng = np.random.default_rng(42)
    s_vec = rng.normal(size=E.shape[1]).astype(E.dtype)
    u_vec = rng.normal(size=E.shape[1]).astype(E.dtype)
    recs2 = get_recommendations(s_vec, u_vec, top_n=5, lam=0.5)
    print("Top‑5 with random session/user vectors (λ=0.5):", recs2)


if __name__ == "__main__":
    _demo()