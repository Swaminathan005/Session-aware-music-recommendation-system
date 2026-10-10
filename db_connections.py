#!/usr/bin/env python3
"""
db_connections.py

Handles connections to MongoDB, Redis, and Cassandra.
Provides helper functions to load metadata, embeddings, and to store/retrieve
vectors, counters, and logs.
"""

import os
import json
import base64
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# MongoDB
from pymongo import MongoClient

# Redis
import redis

# Cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement, PreparedStatement

# -------------------------- CONFIGURATION --------------------------
# Connection details (localhost, default ports)
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB_NAME = "music_rec"
MONGO_COLLECTION_SONGS = "songs"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

CASSANDRA_HOSTS = ["127.0.0.1"]
CASSANDRA_PORT = 9042
CASSANDRA_KEYSPACE = "music_rec"

# User ID (for simplicity, we use a single fixed user)
USER_ID = "user1"

# ---------------------------------------------------------------


def get_mongo_client():
    """Return a MongoDB client."""
    return MongoClient(host=MONGO_HOST, port=MONGO_PORT)


def get_mongo_db():
    """Return the MongoDB database."""
    client = get_mongo_client()
    return client[MONGO_DB_NAME]


def get_songs_collection():
    """Return the MongoDB collection for songs."""
    db = get_mongo_db()
    return db[MONGO_COLLECTION_SONGS]


def get_redis_client():
    """Return a Redis client."""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)


def get_cassandra_session():
    """Return a connected Cassandra session to the keyspace."""
    cluster = Cluster(contact_points=CASSANDRA_HOSTS, port=CASSANDRA_PORT)
    session = cluster.connect()
    # Create keyspace if not exists (simple replication for dev)
    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
        WITH REPLICATION = {{ 'class' : 'SimpleStrategy', 'replication_factor' : 1 }}
        AND DURABLE_WRITES = true;
        """
    )
    session.set_keyspace(CASSANDRA_KEYSPACE)
    return session


def _ensure_tables():
    """Create tables if they don't exist (idempotent)."""
    session = get_cassandra_session()
    # user_listening_events
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS user_listening_events (
            user_id text,
            event_time timeuuid,
            song_id text,
            session_id text,
            context text,
            track_name text,
            artist_name text,
            played_at timestamp,
            ingested_at timestamp,
            PRIMARY KEY ((user_id), event_time)
        ) WITH CLUSTERING ORDER BY (event_time DESC);
        """
    )
    # sessions
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id text PRIMARY KEY,
            user_id text,
            start_time timestamp,
            end_time timestamp,
            track_ids list<text>,
            event_count int,
            context_tags set<text>,
            vector_blob blob
        );
        """
    )
    # user_vector
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS user_vector (
            user_id text PRIMARY KEY,
            vector_blob blob,
            norm float,
            updated_at timestamp,
            version int
        );
        """
    )
    # song_vector (optional, we may keep embeddings in Redis only)
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS song_vector (
            song_id text PRIMARY KEY,
            vector_blob blob,
            norm float,
            source text,
            updated_at timestamp
        );
        """
    )
    # playcount_agg
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS playcount_agg (
            user_id text,
            song_id text,
            play_count counter,
            last_played timestamp,
            PRIMARY KEY ((user_id), song_id)
        );
        """
    )
    # daily_trends
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_trends (
            date date,
            genre text,
            play_count counter,
            unique_users counter,
            top_song_ids list<text>,
            PRIMARY KEY ((date), genre)
        );
        """
    )


# Initialize tables on import (best effort)
try:
    _ensure_tables()
except Exception:
    # If Cassandra is not reachable yet, we'll try later
    pass


# -------------------------- Helper functions for data conversion --------------------------
def _blob_to_vec(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Convert a blob (base64-encoded or raw bytes) to a numpy float64 vector."""
    if blob is None:
        return None
    try:
        # Try to decode as base64 string first
        if isinstance(blob, bytes):
            # Assume it's raw bytes of float64 little endian
            # But we stored as base64 string? We'll handle both.
            # If it's base64 encoded string, decode it.
            # We'll try to interpret as raw float64 array; if fails, try base64.
            # We'll check length: if length % 8 == 0, treat as raw float64.
            if len(blob) % 8 == 0:
                return np.frombuffer(blob, dtype=np.float64)
            else:
                # Assume base64
                decoded = base64.b64decode(blob)
                return np.frombuffer(decoded, dtype=np.float64)
        else:
            return np.frombuffer(blob, dtype=np.float64)
    except Exception:
        return None


def _vec_to_blob(vec: np.ndarray) -> bytes:
    """Convert a numpy float64 vector to raw bytes (little endian)."""
    return vec.astype(np.float64).tobytes()


# -------------------------- Metadata (MongoDB) --------------------------
def load_songs_metadata() -> Dict[str, str]:
    """
    Load song metadata from MongoDB.
    Returns a dict mapping song_id to display string "name - artist".
    Falls back to empty dict on error.
    """
    try:
        coll = get_songs_collection()
        cursor = coll.find({}, {"_id": 0, "song_id": 1, "name": 1, "artist": 1})
        metadata = {}
        for doc in cursor:
            song_id = doc.get("song_id")
            name = doc.get("name", "")
            artist = doc.get("artist", "")
            if song_id and name:
                display = f"{name} - {artist}" if artist else name
                metadata[song_id] = display
        return metadata
    except Exception as e:
        print(f"Warning: Failed to load metadata from MongoDB: {e}")
        return {}


# -------------------------- Embeddings (Redis) --------------------------
def load_embeddings_from_redis() -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]], Optional[Dict[int, str]]]:
    """
    Load all song embeddings from Redis.
    Returns (embeddings matrix, id2idx, idx2id) or (None, None, None) on failure.
    Assumes keys matching "song_emb:*".
    """
    try:
        r = get_redis_client()
        # Scan for keys
        cursor = 0
        id_list = []
        vec_list = []
        while True:
            cursor, keys = r.scan(cursor=cursor, match="song_emb:*", count=100)
            for key in keys:
                try:
                    key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                    # Extract song_id from key: song_emb:<song_id>
                    if key_str.startswith("song_emb:"):
                        song_id = key_str.split(":", 1)[1]
                    else:
                        continue
                    # Fetch hash
                    data = r.hgetall(key)
                    if not data:
                        continue
                    vec_blob = data.get(b'vec') if b'vec' in data else data.get('vec')
                    if vec_blob is None:
                        continue
                    vec = _blob_to_vec(vec_blob)
                    if vec is None:
                        continue
                    id_list.append(song_id)
                    vec_list.append(vec)
                except Exception:
                    continue
            if cursor == 0:
                break
        if not id_list:
            return None, None, None
        # Build matrix
        vec_array = np.stack(vec_list)  # (n_songs, 64)
        id2idx = {song_id: idx for idx, song_id in enumerate(id_list)}
        idx2id = {idx: song_id for idx, song_id in enumerate(id_list)}
        return vec_array, id2idx, idx2id
    except Exception as e:
        print(f"Warning: Failed to load embeddings from Redis: {e}")
        return None, None, None


# -------------------------- Vectors (Redis) --------------------------
def get_user_vector() -> Optional[np.ndarray]:
    """Retrieve the long-term user vector from Redis."""
    try:
        r = get_redis_client()
        data = r.hgetall(f"user_vec:{USER_ID}")
        if not data:
            return None
        vec_blob = data.get(b'vec') if b'vec' in data else data.get('vec')
        if vec_blob is None:
            return None
        return _blob_to_vec(vec_blob)
    except Exception:
        return None


def set_user_vector(vec: np.ndarray) -> None:
    """Store the long-term user vector in Redis."""
    try:
        r = get_redis_client()
        blob = _vec_to_blob(vec)
        r.hset(
            f"user_vec:{USER_ID}",
            mapping={
                "vec": blob,
                "norm": float(np.linalg.norm(vec)),
                "updated_at": int(datetime.now().timestamp() * 1000),
            },
        )
    except Exception as e:
        print(f"Warning: Failed to set user vector in Redis: {e}")


def get_session_vector(session_id: str) -> Optional[np.ndarray]:
    """Retrieve the session vector from Redis."""
    try:
        r = get_redis_client()
        data = r.hgetall(f"sess_vec:{session_id}")
        if not data:
            return None
        vec_blob = data.get(b'vec') if b'vec' in data else data.get('vec')
        if vec_blob is None:
            return None
        return _blob_to_vec(vec_blob)
    except Exception:
        return None


def set_session_vector(session_id: str, vec: np.ndarray) -> None:
    """Store the session vector in Redis."""
    try:
        r = get_redis_client()
        blob = _vec_to_blob(vec)
        r.hset(
            f"sess_vec:{session_id}",
            mapping={
                "vec": blob,
                "norm": float(np.linalg.norm(vec)),
                "updated_at": int(datetime.now().timestamp() * 1000),
            },
        )
    except Exception as e:
        print(f"Warning: Failed to set session vector in Redis: {e}")


# -------------------------- Play counts (Redis) --------------------------
def increment_playcount(user_id: str, song_id: str, amount: int = 1) -> None:
    """Increment the play count for a user-song pair in Redis."""
    try:
        r = get_redis_client()
        key = f"user_playcount:{user_id}:{song_id}"
        r.hincrby(key, "count", amount)
    except Exception as e:
        print(f"Warning: Failed to increment playcount in Redis: {e}")


def get_playcount(user_id: str, song_id: str) -> int:
    """Get the play count for a user-song pair from Redis."""
    try:
        r = get_redis_client()
        key = f"user_playcount:{user_id}:{song_id}"
        count = r.hget(key, "count")
        if count is None:
            return 0
        return int(count)
    except Exception:
        return 0


# -------------------------- Session history (Redis list) --------------------------
def add_to_session_history(session_id: str, song_id: str) -> None:
    """Append a song ID to the session history list (left push)."""
    try:
        r = get_redis_client()
        r.lpush(f"sess_hist:{session_id}", song_id)
    except Exception as e:
        print(f"Warning: Failed to add to session history in Redis: {e}")


def get_session_history(session_id: str) -> List[str]:
    """Retrieve the session history list (most recent first)."""
    try:
        r = get_redis_client()
        # LRANGE 0 -1 to get all
        items = r.lrange(f"sess_hist:{session_id}", 0, -1)
        # Decode if bytes
        result = []
        for item in items:
            if isinstance(item, bytes):
                result.append(item.decode("utf-8"))
            else:
                result.append(item)
        return result
    except Exception:
        return []


def clear_session_history(session_id: str) -> None:
    """Delete the session history list."""
    try:
        r = get_redis_client()
        r.delete(f"sess_hist:{session_id}")
    except Exception as e:
        print(f"Warning: Failed to clear session history in Redis: {e}")


# -------------------------- Listening event logging (Cassandra) --------------------------
def log_listening_event(
    user_id: str,
    song_id: str,
    session_id: str,
    context: str,
    played_at: datetime,
    track_name: Optional[str] = None,
    artist_name: Optional[str] = None,
) -> None:
    """
    Insert a listening event into the user_listening_events table.
    """
    try:
        session = get_cassandra_session()
        event_time = uuid.uuid1()  # timeuuid based on current time
        ingested_at = datetime.now()
        session.execute(
            """
            INSERT INTO user_listening_events
            (user_id, event_time, song_id, session_id, context, track_name, artist_name, played_at, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, event_time, song_id, session_id, context, track_name, artist_name, played_at, ingested_at),
        )
    except Exception as e:
        print(f"Warning: Failed to log listening event to Cassandra: {e}")


def log_session_summary(
    session_id: str,
    user_id: str,
    start_time: datetime,
    end_time: datetime,
    track_ids: List[str],
    event_count: int,
    context_tags: set,
    vector_blob: bytes,
) -> None:
    """
    Insert or update a session summary in the sessions table.
    """
    try:
        session = get_cassandra_session()
        session.execute(
            """
            INSERT INTO sessions
            (session_id, user_id, start_time, end_time, track_ids, event_count, context_tags, vector_blob)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                user_id,
                start_time,
                end_time,
                track_ids,
                event_count,
                context_tags,  # pass the set directly
                vector_blob,
            ),
        )
    except Exception as e:
        print(f"Warning: Failed to log session summary to Cassandra: {e}")


# -------------------------- User vector persistence (Cassandra) --------------------------
def get_user_vector_from_cassandra() -> Optional[np.ndarray]:
    """Retrieve the user vector from Cassandra."""
    try:
        session = get_cassandra_session()
        row = session.execute(
            "SELECT vector_blob FROM user_vector WHERE user_id = %s", (USER_ID,)
        ).one()
        if row and row.vector_blob:
            return _blob_to_vec(row.vector_blob)
        return None
    except Exception:
        return None


def set_user_vector_to_cassandra(vec: np.ndarray) -> None:
    """Store the user vector in Cassandra."""
    try:
        session = get_cassandra_session()
        blob = _vec_to_blob(vec)
        norm = float(np.linalg.norm(vec))
        updated_at = datetime.now()
        # We'll do a simple insert; if row exists, it will be replaced (since PRIMARY KEY)
        session.execute(
            """
            INSERT INTO user_vector (user_id, vector_blob, norm, updated_at, version)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (USER_ID, blob, norm, updated_at, 0),  # version not used for now
        )
    except Exception as e:
        print(f"Warning: Failed to set user vector in Cassandra: {e}")


# -------------------------- Cleanup (optional) --------------------------
def close_connections():
    """Close any open connections (if needed)."""
    # Redis and MongoDB clients don't need explicit close in most cases.
    # Cassandra cluster shutdown can be done if we kept a reference.
    pass


# If this module is run directly, run a simple test.
if __name__ == "__main__":
    print("Testing connections...")
    try:
        m = get_mongo_client()
        print("MongoDB client:", m)
    except Exception as e:
        print("MongoDB error:", e)
    try:
        r = get_redis_client()
        print("Redis ping:", r.ping())
    except Exception as e:
        print("Redis error:", e)
    try:
        c = get_cassandra_session()
        print("Cassandra session:", c)
    except Exception as e:
        print("Cassandra error:", e)