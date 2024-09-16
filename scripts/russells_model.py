import json


def categorize_song(song_features):
    danceability = song_features['danceability']
    energy = song_features['energy']
    valence = song_features['valence']
    tempo = song_features['tempo']
    loudness = song_features['loudness']
    instrumentalness = song_features['instrumentalness']
    mode = song_features['mode']
    
    # Instrumental songs
    if instrumentalness > 0.5:
        return 'instrumental'
    
    # High arousal (energy), High valence (positive)
    if energy > 0.7 and valence > 0.6:
        if tempo > 120 and danceability > 0.6:
            return 'happy'
        else:
            return 'energetic'
    
    # High arousal (energy), Low valence (negative)
    if energy > 0.7 and valence < 0.4:
        if mode == 0:  # Minor key
            return 'confident'
        else:
            return 'tense'
    
    # Low arousal (calm), High valence (positive)
    if energy < 0.5 and valence > 0.6:
        return 'calm'
    
    # Low arousal (energy), Low valence (negative)
    if energy < 0.5 and valence < 0.4:
        return 'sad'
    
    # Default categorization if no clear category
    if valence > 0.5:
        return 'happy'
    else:
        return 'sad'

# Example usage:

# song = {
#     "danceability": 0.765,
#     "energy": 0.473,
#     "key": 10,
#     "loudness": -5.829,
#     "mode": 1,
#     "speechiness": 0.0514,
#     "acousticness": 0.287,
#     "instrumentalness": 0,
#     "liveness": 0.391,
#     "valence": 0.34,
#     "tempo": 119.992
# }

with open('json/sound_data.json', 'r') as file:
    songs = json.load(file)

for song in songs:
    # print(song)
    print(f"Category: {categorize_song(song)}\nSong ID: {song.get("id", "no id")}\n\n")

