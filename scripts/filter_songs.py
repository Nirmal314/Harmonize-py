import json
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import os
from dotenv import load_dotenv

load_dotenv()
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri='https://google.com'
                                               ))

with open('json/playlist_response.json', 'r') as file:
    data = json.load(file)

filtered_data = []

for item in data:
    song = {}
    track = item.get('track')

    song["name"] = track.get('name')
    song["url"] = track.get('external_urls').get('spotify')

    song_id = track.get('id')

    features = sp.audio_features(song_id)[0]

    features.pop('uri')
    features.pop('track_href')
    features.pop('analysis_url')
    features.pop('duration_ms')
    features.pop('time_signature')
    features.pop('type')
    features.pop('id')

    song["features"] = features

    # song["category"] = ""
    # song["reason"] = ""

    filtered_data.append(song)

# print(filtered_data)

with open('json/songs_with_features.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)