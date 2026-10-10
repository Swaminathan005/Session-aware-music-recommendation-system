#!/usr/bin/env python3
"""
interactive_recommendation.py

Simple terminal UI for session-aware music recommendations (no curses).
- Displays header with project name "GROVE" in large ASCII art, colored blue/orange.
- Shows session info, table of selected songs in current session, and table of recommendations.
- User types song numbers or commands (start, end, rec, quit, schema).
- Uses MongoDB for song metadata, Redis for embeddings and vectors, Cassandra for event logs.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime
from db_schema import print_schemas
import db_connections as db

# -------------------------- CONFIGURATION --------------------------
BASE_DIR = Path(__file__).parent
EMBEDDINGS_PATH = BASE_DIR / "song_embeddings.npy"
ID_MAP_PATH     = BASE_DIR / "song_id_map.pkl"
METADATA_PATH   = BASE_DIR / "initial_data.csv"

# Learning rates
LR_SESSION = 0.30   # fast adaptation per song
LR_USER    = 0.03   # slow adaptation per session (applied when session ends)

# Blend weights for context vector C = w_u * U + w_s * S
W_U = 0.4   # weight for long-term user vector
W_S = 0.6   # weight for session vector

# Scoring weights (must sum to 1.0)
W_SIM   = 0.60   # weight for similarity to context
W_DIR   = 0.25   # weight for direction alignment
W_FRESH = 0.15   # weight for freshness (set to 1.0 when no history)

# Number of candidates to consider for scoring (speed‑up)
TOP_CANDIDATES = 500
# Number of final recommendations to show
TOP_FINAL = 10

# Reset policy for session vector after a session ends:
#   "zero"  -> start from zero vector
#   "user"  -> start from the current user vector
RESET_TO = "zero"
# -----------------------------------------------------------------


def load_embeddings():
    """Load embeddings and ID↔index maps, trying Redis first, then local files."""
    # Try Redis
    E, id2idx, idx2id = db.load_embeddings_from_redis()
    if E is not None and id2idx is not None and idx2id is not None:
        #print("Loaded embeddings from Redis")
        return E, id2idx, idx2id
    # Fallback to local files
    #print("Loading embeddings from local files (fallback)")
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


def load_metadata():
    """Load song metadata from MongoDB, then local CSV as fallback."""
    # Try MongoDB
    metadata = db.load_songs_metadata()
    if metadata:
        #print("Loaded metadata from MongoDB")
        return metadata
    # Fallback to local CSV
    #print("Loading metadata from local CSV (fallback)")
    if not METADATA_PATH.is_file():
        return {}
    df = pd.read_csv(METADATA_PATH)
    def make_display(row):
        name = str(row.get('name', ''))
        artist = str(row.get('artist', ''))
        if artist and artist != 'nan':
            return f"{name} - {artist}"
        else:
            return name
    df['display'] = df.apply(make_display, axis=1)
    return dict(zip(df['track_id'], df['display']))


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return unit‑length vector (avoid div‑by‑zero)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    return v / norm


def update_session(s_vec: np.ndarray, song_vec: np.ndarray, lr: float) -> np.ndarray:
    """Exponential moving average update for the session vector."""
    new_s = (1 - lr) * s_vec + lr * song_vec
    # Persist to Redis if we have a session_id (set later in main)
    # We'll handle persistence in main after updating s_vec
    return new_s


def update_user(u_vec: np.ndarray, s_vec: np.ndarray, lr: float) -> np.ndarray:
    """Exponential moving average update for the user vector at session end."""
    new_u = (1 - lr) * u_vec + lr * s_vec
    # Persist to Redis and Cassandra in main after session ends
    return new_u


def reset_session(reset_to: str, u_vec: np.ndarray) -> np.ndarray:
    """Return a fresh session vector according to reset policy."""
    if reset_to == "zero":
        return np.zeros_like(u_vec)
    elif reset_to == "user":
        return u_vec.copy()
    else:
        raise ValueError(f"Unknown reset_to: {reset_to}")


def compute_direction_vector(session_embeds: list) -> np.ndarray:
    """
    Estimate session movement direction as the difference between
    average embedding of first half and second half of the session.
    Returns a unit direction vector; if insufficient data, returns zero.
    """
    if len(session_embeds) < 2:
        return np.zeros_like(session_embeds[0]) if session_embeds else np.zeros(64)
    mat = np.stack(session_embeds)  # (n_session, 64)
    split = len(mat) // 2
    first = mat[:split]
    second = mat[split:]
    dir_vec = np.mean(second, axis=0) - np.mean(first, axis=0)
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        return np.zeros_like(dir_vec)
    return dir_vec / norm


def recommend(s_vec: np.ndarray,
              u_vec: np.ndarray,
              session_embeds: list,
              E: np.ndarray,
              id2idx: dict,
              idx2id: dict,
              metadata: dict,
              topn: int = TOP_FINAL):
    """
    Compute top‑N recommendations using the three‑score formula.
    Returns list of tuples (song_id, display_name, score).
    """
    # Context vector C = 0.4*U + 0.6*S
    c_vec = W_U * u_vec + W_S * s_vec
    # Direction vector D from session history
    d_vec = compute_direction_vector(session_embeds)
    # Freshness: without listening history we set to 1.0 for all songs
    fresh = 1.0

    # Normalize vectors for cosine similarity (zero vector stays zero)
    c_unit = _normalize(c_vec)
    d_unit = _normalize(d_vec)

    # Normalize all song embeddings
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    E_norm = E / norms

    # Compute similarities
    sim_to_c = E_norm.dot(c_unit)          # (n_songs,)
    sim_to_d = E_norm.dot(d_unit)          # (n_songs,)

    # Combined score
    scores = W_SIM * sim_to_c + W_DIR * sim_to_d + W_FRESH * fresh

    # Get top indices
    top_idx = np.argsort(-scores)[:topn]

    results = []
    for idx in top_idx:
        song_id = idx2id[int(idx)]
        display = metadata.get(song_id, str(song_id))
        score = float(scores[idx])
        results.append((song_id, display, score))
    return results


def draw_header():
    """Print the header with large ASCII art for GROVE ♪, colored blue/orange."""
    # ANSI colors
    BLUE = "\033[94m"
    WHITE = "\033[97m"  # bright white
    ORANGE = "\033[38;5;214m"  # orange 214 in 256-color
    RESET = "\033[0m"
    # The ASCII art lines (7 lines) as provided by the user (after revert)
    lines = [
        " ██████╗ ██████╗  ██████╗ ██╗   ██╗███████╗ ♪",
        "██╔════╝ ██╔══██╗██╔═══██╗██║   ██║██╔════╝",
        "██║  ███╗██████╔╝██║   ██║██║   ██║█████╗",
        "██║   ██║██╔══██╗██║   ██║╚██╗ ██╔╝██╔══╝",
        "╚██████╔╝██║  ██║╚██████╔╝ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪",
        " ╚═╝     ╚═╝╚═╝  ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪ ⚪",
        "                    GROVE ♪"
    ]
    # Padding to help center (fixed left padding)
    PAD = " " * 12
    # Process each line: we want block characters blue, GROVE letters alternating blue/white, note orange
    for line in lines:
        # First, apply blue to the whole line (blocks and spaces)
        colored_line = BLUE + line + RESET
        # Replace GROVE with alternating blue/white letters
        if "GROVE" in line:
            # Build colored GROVE: G-Blue, R-White, O-Blue, V-White, E-Blue
            grove_colored = (
                BLUE + "G" + RESET +
                WHITE + "R" + RESET +
                BLUE + "O" + RESET +
                WHITE + "V" + RESET +
                BLUE + "E" + RESET
            )
            colored_line = colored_line.replace("GROVE", grove_colored)
        # Replace note with orange
        if "♪" in line:
            note_colored = ORANGE + "♪" + RESET
            colored_line = colored_line.replace("♪", note_colored)
        # Increase size by printing each line twice (vertical scaling)
        padded_line = PAD + colored_line
        print(padded_line)
        print(padded_line)  # duplicate line for increased height
    # After art lines, print a blank line (separator before caption)
    print()


def draw_ui(state):
    """Draw the entire UI (clear screen and reprint)."""
    # Clear screen and move cursor to top-left
    print("\033[2J\033[H", end="")
    draw_header()
    # Add caption
    CAPTION = "Every note finds its place."
    # Center caption: we'll just print it as is; could add spaces but terminal width unknown.
    # We'll print it in white color.
    WHITE = "\033[97m"
    RESET = "\033[0m"
    print(WHITE + CAPTION + RESET)
    # Show total songs and commands
    n_songs = state['n_songs']
    DIM = "\033[90m"  # dark gray
    msg = f"Input song numbers from 0 to {n_songs-1} | Commands: start, end, rec, quit, schema"
    print(DIM + msg + RESET)
    print()  # blank line

    # Session info line
    info_line = (
        f"Session: {'ON' if state['session_active'] else 'OFF'} | "
        f"Session vec norm: {state['s_vec_norm']:.3f} | "
        f"User vec norm: {state['u_vec_norm']:.3f} | "
        f"Session length: {len(state['session_embeds'])}"
    )
    print(info_line)
    print()

    # Selected Songs table
    print("Selected Songs:")
    if state['session_song_ids']:
        # Header
        print(f"{'SL No.':<6} {'Name':<50}")
        print("-" * 60)
        for i, song_id in enumerate(state['session_song_ids'], start=1):
            name = state['metadata'].get(song_id, str(song_id))
            # Truncate name if too long
            if len(name) > 48:
                name = name[:45] + "..."
            print(f"{i:<6} {name:<50}")
    else:
        print("(none)")
    print()

    # Recommendations table
    print("Recommendations:")
    if state.get('last_recs'):
        # Header
        print(f"{'SL No.':<6} {'Name':<40} {'Confidence':>10}")
        print("-" * 60)
        for i, (_, name, score) in enumerate(state['last_recs'], start=1):
            # Truncate name if too long
            if len(name) > 38:
                name = name[:35] + "..."
            print(f"{i:<6} {name:<40} {score:>10.4f}")
    else:
        print("(none)")
    print()

    # Prompt
    print("> ", end="", flush=True)


def main():
    # Load data (try DB first, fallback to files)
    E, id2idx, idx2id = load_embeddings()
    metadata = load_metadata()
    n_songs = E.shape[0]

    # State variables
    s_vec = np.zeros(E.shape[1], dtype=E.dtype)   # session vector (in-memory)
    u_vec = np.zeros(E.shape[1], dtype=E.dtype)   # user vector (in-memory)
    session_active = False
    session_embeds = []   # raw embeddings for direction
    session_song_ids = [] # IDs of selected songs in current session
    session_id = None     # current session identifier (string)
    input_buf = ""        # not used; we use input()
    last_recs = []        # last recommendations to show

    # Attempt to load user vector from Redis (fallback to zero)
    u_vec_from_redis = db.get_user_vector()
    if u_vec_from_redis is not None:
        u_vec = u_vec_from_redis
        #print("Loaded user vector from Redis")
    else:
        #print("Using zero user vector (no Redis data)")
        pass

    draw_ui({
        'session_active': session_active,
        's_vec_norm': np.linalg.norm(s_vec),
        'u_vec_norm': np.linalg.norm(u_vec),
        'session_embeds': session_embeds,
        'session_song_ids': session_song_ids,
        'last_recs': last_recs,
        'metadata': metadata,
        'n_songs': n_songs
    })

    while True:
        try:
            cmd = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd:
            # blank line, just redraw
            draw_ui({
                'session_active': session_active,
                's_vec_norm': np.linalg.norm(s_vec),
                'u_vec_norm': np.linalg.norm(u_vec),
                'session_embeds': session_embeds,
                'session_song_ids': session_song_ids,
                'last_recs': last_recs,
                'metadata': metadata,
                'n_songs': n_songs
            })
            continue

        # Process command
        if cmd.lower() == "quit":
            print("Goodbye.")
            break
        elif cmd.lower() == "start":
            if session_active:
                pass  # ignore or could show error
            else:
                # Generate a new session ID
                session_id = str(uuid.uuid4())
                # Reset session vector according to policy
                s_vec = reset_session(RESET_TO, u_vec)
                session_active = True
                session_embeds = []
                session_song_ids = []
                last_recs = []
                # Initialize session vector in Redis
                db.set_session_vector(session_id, s_vec)
                # Initialize empty session history in Redis
                # (will be built as we add songs)
                # Optionally set a start time in Cassandra? We'll log at end.
        elif cmd.lower() == "end":
            if not session_active:
                pass
            else:
                # Update user vector with the current session vector (slow)
                u_vec = update_user(u_vec, s_vec, LR_USER)
                # Persist user vector to Redis and Cassandra
                db.set_user_vector(u_vec)
                db.set_user_vector_to_cassandra(u_vec)
                # Persist session vector to Cassandra as part of session summary
                # Gather session stats
                start_time = None  # we didn't store start time; we can approximate
                # For simplicity, we'll not store start/end times; we'll just store the vector.
                # We'll create a minimal session summary.
                # We'll need track_ids (list of song IDs), event_count, context_tags (empty set)
                track_ids = list(session_song_ids)
                event_count = len(track_ids)
                context_tags = set()  # we don't have context yet
                # Get the session vector blob
                sess_vec_blob = db._vec_to_blob(s_vec) if hasattr(db, '_vec_to_blob') else s_vec.tobytes()
                # Log session summary to Cassandra
                db.log_session_summary(
                    session_id=session_id,
                    user_id=db.USER_ID,
                    start_time=datetime.now(),  # placeholder
                    end_time=datetime.now(),
                    track_ids=track_ids,
                    event_count=event_count,
                    context_tags=context_tags,
                    vector_blob=sess_vec_blob,
                )
                # Clear session history in Redis
                db.clear_session_history(session_id)
                # Reset session vector
                s_vec = reset_session(RESET_TO, u_vec)
                session_active = False
                session_embeds = []
                session_song_ids = []
                session_id = None
                # last_recs maybe keep or clear; we'll keep
        elif cmd.lower() == "rec":
            if session_active:
                recs = recommend(s_vec, u_vec, session_embeds, E, id2idx, idx2id, metadata, topn=TOP_FINAL)
                last_recs = recs
            else:
                last_recs = []  # no recs if no session
        elif cmd.lower() == "schema":
            # Print schema and then show prompt again without redrawing UI
            print(print_schemas())
            print("> ", end="", flush=True)
            continue
        else:
            # try integer
            try:
                idx = int(cmd)
            except ValueError:
                idx = None
            if idx is not None and 0 <= idx < n_songs and session_active:
                # select song
                song_vec = E[idx]
                # Update session vector (fast)
                s_vec = update_session(s_vec, song_vec, LR_SESSION)
                # Persist updated session vector to Redis
                if session_id is not None:
                    db.set_session_vector(session_id, s_vec)
                # Keep raw embeddings for direction
                session_embeds.append(song_vec)
                # Store song ID in session history (Redis list)
                if session_id is not None:
                    db.add_to_session_history(session_id, idx2id[int(idx)])
                # Store selected song ID for UI
                session_song_ids.append(idx2id[int(idx)])
                # Increment playcount in Redis
                db.increment_playcount(db.USER_ID, idx2id[int(idx)])
                # Log listening event to Cassandra
                db.log_listening_event(
                    user_id=db.USER_ID,
                    song_id=idx2id[int(idx)],
                    session_id=session_id if session_id else "",
                    context="",  # we don't have context yet
                    played_at=datetime.now(),
                )
            # else ignore invalid input

        # Redraw UI
        draw_ui({
            'session_active': session_active,
            's_vec_norm': np.linalg.norm(s_vec),
            'u_vec_norm': np.linalg.norm(u_vec),
            'session_embeds': session_embeds,
            'session_song_ids': session_song_ids,
            'last_recs': last_recs,
            'metadata': metadata,
            'n_songs': n_songs
        })


if __name__ == "__main__":
    main()