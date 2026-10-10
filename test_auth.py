import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

CLIENT_ID = "edc9f067830e4d83a12c5ae6e8dde46e"
CLIENT_SECRET = "7af2df8249734f9cb85191c2eaba770"
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPE = "user-read-recently-played"

auth_manager = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=".spotify_cache_test",
    show_dialog=True
)

sp = spotipy.Spotify(auth_manager=auth_manager)

try:
    user = sp.current_user()
    print("Success:", user["display_name"])
except Exception as e:
    print("Error:", e)