# Snippet showing runtime database updates from interactive_recommendation.py
# This code is executed when the user selects a song (idx) during an active session.

song_vec = E[idx]
# Update session vector (fast EMA)
s_vec = update_session(s_vec, song_vec, LR_SESSION)
# Persist updated session vector to Redis
if session_id is not None:
    db.set_session_vector(session_id, s_vec)
# Keep raw embeddings for direction vector computation
session_embeds.append(song_vec)
# Store song ID in session history (Redis LPUSH)
if session_id is not None:
    db.add_to_session_history(session_id, idx2id[int(idx)])
# Store selected song ID for UI display
session_song_ids.append(idx2id[int(idx)])
# Increment playcount in Redis
db.increment_playcount(db.USER_ID, idx2id[int(idx)])
# Log listening event to Cassandra (immutable event log)
db.log_listening_event(
    user_id=db.USER_ID,
    song_id=idx2id[int(idx)],
    session_id=session_id if session_id else "",   # empty string if no session
    context="",                                   # placeholder for context
    played_at=datetime.now(),
)