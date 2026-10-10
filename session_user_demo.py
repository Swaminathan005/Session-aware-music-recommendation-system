#!/usr/bin/env python3
"""
Simple terminal demo for session/user vector interaction.

- Load MSD song embeddings (song_embeddings.npy) and id map (song_id_map.pkl).
- User enters song indices (0 .. N-1) to simulate song selections.
- Type 'start' to begin a session, 'end' to finish the current session,
  'rec' to see top‑5 recommendations based on blended session/user vector,
  'quit' to exit.
- Session vector updates quickly with each song (high learning rate).
- User vector updates slowly when a session ends (low learning rate).
Both vectors live in the same 64‑dim embedding space as the song embeddings.
"""

import os
import pickle
import numpy as np
from pathlib import Path

# -------------------------- CONFIGURATION --------------------------
BASE_DIR = Path(__file__).parent
EMBEDDINGS_PATH = BASE_DIR / "song_embeddings.npy"
ID_MAP_PATH     = BASE_DIR / "song_id_map.pkl"

# Learning rates
LR_SESSION = 0.30   # fast adaptation per song
LR_USER    = 0.03   # slow adaptation per session (applied when session ends)
BLEND_LAMBDA = 0.70  # weight for session vs user in recommendation (0..1)

# Reset policy for session vector after a session ends:
#   "zero"  -> start from zero vector
#   "user"  -> start from the current user vector
RESET_TO = "zero"

# -----------------------------------------------------------------
def load_embeddings():
    """Load embeddings and id↔index maps."""
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}")
    if not ID_MAP_PATH.is_file():
        raise FileNotFoundError(f"Missing {ID_MAP_PATH}")
    E = np.load(EMBEDDINGS_PATH)          # shape (N_songs, 64)
    with open(ID_MAP_PATH, 'rb') as f:
        id2idx = pickle.load(f)           # dict: item_id -> index
    # Build reverse map index -> item_id for convenience
    idx2id = {v: k for k, v in id2idx.items()}
    # Sanity check: ensure indices are dense 0..N_songs-1
    n = E.shape[0]
    if set(id2idx.values()) != set(range(n)):
        raise ValueError("id2idx values are not a permutation of 0..N_songs-1")
    return E, id2idx, idx2id

# -------------------------- CORE LOGIC --------------------------
def update_session(s_vec, song_vec, lr):
    """Exponential moving average update for the session vector (64‑dim)."""
    return (1 - lr) * s_vec + lr * song_vec

def update_user(u_vec, s_vec, lr):
    """Exponential moving average update for the user vector (64‑dim) at session end."""
    return (1 - lr) * u_vec + lr * s_vec

def reset_session(reset_to, u_vec):
    """Return a fresh session vector according to reset policy."""
    if reset_to == "zero":
        return np.zeros_like(u_vec)
    elif reset_to == "user":
        return u_vec.copy()
    else:
        raise ValueError(f"Unknown reset_to: {reset_to}")

def recommend(s_vec, u_vec, E, id2idx, idx2id, topk=5, lam=BLEND_LAMBDA, exclude=None):
    """Return top‑k song indices sorted by similarity to blended vector."""
    blended = lam * s_vec + (1 - lam) * u_vec
    norm_blended = np.linalg.norm(blended)
    if norm_blended == 0:
        # zero vector -> return arbitrary ordering (e.g., first topk)
        order = np.arange(E.shape[0])[:topk]
        return [idx2id[i] for i in order]
    blended_norm = blended / norm_blended
    # Normalize item vectors
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    E_norm = E / norms
    scores = E_norm.dot(blended_norm)  # (N_songs,)
    if exclude:
        mask = np.zeros(E.shape[0], dtype=bool)
        for ex in exclude:
            if ex in id2idx:
                mask[id2idx[ex]] = True
        scores[mask] = -np.inf
    top_idx = np.argsort(-scores)[:topk]
    return [idx2id[i] for i in top_idx]

def main():
    print("\n=== Session/User Vector Demo ===")
    print("Loading embeddings...")
    E, id2idx, idx2id = load_embeddings()
    N_songs = E.shape[0]
    print(f"Loaded {N_songs} unique songs (indices 0 … {N_songs-1}).")
    print("Commands:")
    print("  <num>   -> select song with that index (only works inside a session)")
    print("  start   -> begin a new session")
    print("  end     -> finish current session and update user vector")
    print("  rec     -> show top‑5 recommendations based on blended vector")
    print("  quit    -> exit program")
    print("-" * 50)

    # State vectors (64‑dim)
    s_vec = np.zeros(E.shape[1], dtype=E.dtype)   # session vector
    u_vec = np.zeros(E.shape[1], dtype=E.dtype)   # user vector
    session_active = False

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd:
            continue

        if cmd.lower() == "quit":
            print("Goodbye.")
            break

        if cmd.lower() == "start":
            if session_active:
                print("A session is already active. Type 'end' to finish it first.")
            else:
                # Reset session vector according to policy
                s_vec = reset_session(RESET_TO, u_vec)
                session_active = True
                print("Session started. Session vector reset.")
            continue

        if cmd.lower() == "end":
            if not session_active:
                print("No active session to end.")
                continue
            # Update user vector with the current session vector (slow)
            u_vec = update_user(u_vec, s_vec, LR_USER)
            print(f"Session ended. User vector updated (norm={np.linalg.norm(u_vec):.3f}).")
            # Reset session vector for next session
            s_vec = reset_session(RESET_TO, u_vec)
            session_active = False
            continue

        if cmd.lower() == "rec":
            if not session_active:
                print("No active session. Start a session first to get meaningful recommendations.")
            else:
                # Exclude the currently selected song? We'll just recommend top‑5 overall.
                recs = recommend(s_vec, u_vec, E, id2idx, idx2id, topk=5, lam=BLEND_LAMBDA, exclude=None)
                print("Top‑5 recommendations (song indices):", recs)
            continue

        # Try to interpret as an integer song index
        try:
            idx = int(cmd)
        except ValueError:
            print("Please enter a number, 'start', 'end', 'rec', or 'quit'.")
            continue

        if not (0 <= idx < N_songs):
            print(f"Index out of range (0 … {N_songs-1}).")
            continue

        if not session_active:
            print("Please start a session first (type 'start') before selecting songs.")
            continue

        # Retrieve the song's embedding vector (64‑dim)
        song_vec = E[idx]  # assuming embeddings are in the same order as indices 0..N-1
        # Update session vector (fast)
        s_vec = update_session(s_vec, song_vec, LR_SESSION)
        print(f"Selected song {idx}. Session vector norm: {np.linalg.norm(s_vec):.3f}")

if __name__ == "__main__":
    main()