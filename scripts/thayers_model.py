import json

def categorize_song(song):
    energy = song['energy']
    valence = song['valence']
    instrumentalness = song['instrumentalness']
    danceability = song['danceability']

    # Categorize as Instrumental
    if instrumentalness >= 0.5:
        return 'Instrumental'
    
    # Thayer's model-based categorization
    if energy >= 0.6:
        if valence >= 0.5:
            if danceability >= 0.7:
                return 'Confident'
            return 'Happy'
        else:
            return 'Energetic'
    else:
        if valence >= 0.5:
            return 'Calm'
        else:
            return 'Sad'
        
with open('json/sound_data.json', 'r') as file:
    songs = json.load(file)

for song in songs:
    # print(song)
    print(f"Category: {categorize_song(song)}\nSong ID: {song.get("id", "no id")}\n\n")

