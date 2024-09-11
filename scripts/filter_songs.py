import json

with open('json/playlist_response.json', 'r') as file:
    data = json.load(file)

for item in data:
    track = item.get('track', {})
    song_name = track.get('name', 'Unknown')
    spotify_url = track.get('external_urls', {}).get('spotify', 'No URL available')
    
    print(f"Song Name: {song_name}")
    print(f"Spotify URL: {spotify_url}")