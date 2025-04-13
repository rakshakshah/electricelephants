import os
import requests
import base64
import json
from dotenv import load_dotenv
from io import BytesIO
import subprocess

load_dotenv()

##############################################################################
# 1) SPOTIFY: MANUAL CLIENT CREDENTIALS (to get track preview + metadata)
##############################################################################

def get_spotify_access_token():
    """
    Retrieves an access token from Spotify using the Client Credentials Flow.
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise Exception("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment.")

    token_url = "https://accounts.spotify.com/api/token"
    auth_str = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    resp = requests.post(token_url, headers=headers, data=data)
    if resp.status_code != 200:
        raise Exception(f"Failed to get Spotify token: {resp.text}")

    return resp.json()["access_token"]


def get_preview_from_node(track_name, artist_name):
    """
    Uses Node (and the npm package spotify-preview-finder) to fetch a preview URL.
    Requires that a Node script named 'get_preview.js' exists.
    
    This function calls:
        node get_preview.js "<track_name> - <artist_name>"
    and expects the Node script to output the preview URL (or an error message).
    """
    query = f"{track_name} - {artist_name}"

    try:
        result = subprocess.run(
            ["node", "get_preview.js", query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        preview_url = result.stdout.strip()
        if not preview_url:
            print("No preview URL found by Node script.")
            return None
        return preview_url
    except subprocess.CalledProcessError as e:
        print("Error running Node Script for preview URL:", e.stderr)
        return None

def get_track_metadata(track_query, token):
    """
    Searches Spotify for the track_query (song/artist) and returns track metadata:
      id, name, artists, album, preview_url, spotify_url
    """
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": track_query, "type": "track", "limit": 1}

    r = requests.get(search_url, headers=headers, params=params)
    if r.status_code != 200:
        print("Spotify search error:", r.text)
        return None

    data = r.json()
    items = data.get("tracks", {}).get("items", [])
    if not items:
        print(f"No tracks found for '{track_query}'.")
        return None

    first_track = items[0]
    # Convert duration from milliseconds to seconds
    duration_ms = first_track.get("duration_ms", 0)
    duration_sec = duration_ms // 1000
    metadata =  {
        "id": first_track["id"],
        "name": first_track["name"],
        "artists": [a["name"] for a in first_track["artists"]],
        "album": first_track["album"]["name"],
        "duration": duration_sec,
        "preview_url": first_track["preview_url"],  # May be None or only 30s
        "spotify_url": first_track["external_urls"].get("spotify"),
    }

    if not metadata["preview_url"]:
        preview = get_preview_from_node(metadata["name"], metadata["artists"][0])
        if preview:
            metadata["preview_url"] = preview

    return metadata

##############################################################################
# 2) RECCOBEATS: AUDIO FEATURE EXTRACTION (multipart/form-data)
##############################################################################

def get_reccobeats_features_from_mp3(mp3_bytes):
    """
    Uploads the in-memory MP3 (<= 5MB, <=30s) to Reccobeats' Audio Features endpoint.

    Docs: https://reccobeats.com/docs/documentation/Analysis/audio-features-extraction

    POST /v1/analysis/audio-features
    Content-Type: multipart/form-data
      audioFile=<binary>
    """
    url = "https://api.reccobeats.com/v1/analysis/audio-features"
    
    # Reccobeats docs say: audioFile is required, must be 'multipart/form-data'
    # We'll pass 'files' dict for requests to handle the form data:
    files = {
        "audioFile": ("preview.mp3", mp3_bytes, "audio/mpeg"),
    }

    resp = requests.post(url, files=files)
    if resp.status_code == 200:
        return resp.json()  # {acousticness, danceability, energy, ...}
    else:
        print("Reccobeats error:", resp.status_code, resp.text)
        return None

##############################################################################
# 3) Get lyrics with some API
##############################################################################
def get_lrclib_lyrics(track_name, artist_name, album_name, duration):
    url = "https://lrclib.net/api/get"
    params = {
        "track_name": track_name,
        "artist_name": artist_name,
        "album_name": album_name,
        "duration": duration
    }
    
    # It is recommended to include a User-Agent header when making requests to LRCLIB.
    headers = {
        "User-Agent": "LRCGET v0.2.0 (https://github.com/tranxuanthang/lrcget)"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        # Successfully found lyrics; return the JSON data.
        return response.json()
    elif response.status_code == 404:
        # Track not found: return an error message dictionary.
        return {
            "error": "TrackNotFound",
            "message": "Failed to find specified track"
        }
    else:
        # For any unexpected errors, raise an exception.
        response.raise_for_status()


##############################################################################
# MAIN: Putting It All Together
##############################################################################

def main():
    print("Welcome! This script will:")
    print("1) Get a Spotify track's preview audio.")
    print("2) Upload it to Reccobeats for audio feature analysis.")
    print("3) Use open-source API to get the lyrics.")
    print("4) Return all of the info.\n")

    # 1) Get Spotify token
    try:
        sp_token = get_spotify_access_token()
        print("Acquired Spotify token (first 20 chars):", sp_token[:20], "...")
    except Exception as e:
        print("Error retrieving Spotify token:", e)
        return

    # 2) Ask user for track
    song_query = input("Enter a song name: ").strip()
    if not song_query:
        print("No input provided. Exiting.")
        return

    # 3) Get track metadata from Spotify
    track_info = get_track_metadata(song_query, sp_token)
    if not track_info:
        print("Failed to retrieve Spotify track info. Exiting.")
        return

    print(f"\nTrack found: {track_info['name']} by {', '.join(track_info['artists'])}")
    print(f"Album: {track_info['album']}")
    print(f"Duration (sec): {track_info['duration']}")
    print(f"Preview URL: {track_info['preview_url']}")
    if not track_info["preview_url"]:
        print("No preview available for this track. Reccobeats needs an audio file. Exiting.")
        return
    
    # 3a) Get song lyrics using LRCLIB
    # LRCLIB requires the exact track signature: title, artist, album, and duration (in seconds)
    lyrics_response = get_lrclib_lyrics(
        track_name=track_info["name"],
        artist_name=track_info["artists"][0],
        album_name=track_info["album"],
        duration=track_info["duration"]
    )
    if "plainLyrics" in lyrics_response:
        lyrics = lyrics_response["plainLyrics"]
    else:
        lyrics = lyrics_response.get("message", "Lyrics not found.")
    
    print("\nLyrics:")
    print(lyrics)

    # 4) Download the preview MP3
    try:
        preview_resp = requests.get(track_info["preview_url"])
        preview_resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error downloading preview audio:", e)
        return

    # Store it in-memory as BytesIO
    mp3_bytes = BytesIO(preview_resp.content)

    # 5) Upload to Reccobeats
    recco_feats = get_reccobeats_features_from_mp3(mp3_bytes)
    if not recco_feats:
        print("Could not retrieve features from Reccobeats. Exiting.")
        return

    print("\nReccobeats features:", json.dumps(recco_feats, indent=4))

if __name__ == "__main__":
    main()
