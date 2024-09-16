import spotipy
from dotenv import load_dotenv
import os
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import json

model = load_model('model/models_final.h5')

load_dotenv()
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri='https://google.com'
                                               ))

# Fetch playlist tracks
playlist_id = '2uORYX3pVmRBUJe8uXrK8H'
results = sp.playlist_tracks(playlist_id)
tracks = results['items']

# Extract features, names, and URLs
songs = []
for track in tracks:
    song = {}
    song_id = track['track']['id']
    features = sp.audio_features(song_id)[0]
    if features is not None:
        song["name"] = track['track']['name']
        song["url"] = track['track']['external_urls']['spotify']

        features.pop('type')
        features.pop('id')
        features.pop('uri')
        features.pop('track_href')
        features.pop('analysis_url')
        features.pop('duration_ms')
        features.pop('time_signature')

        song["features"] = features

        songs.append(song)

# Define the feature list
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

df = pd.read_csv('predicted_songs_with_categories_vs.csv')
X = df[features]

# Fit the scaler using the existing data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit the scaler here

y = df['predicted_category']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

def predict_category(song_data):
    song_data = song_data['features']
    # Ensure the song_data has all required features
    required_features = set(features)
    provided_features = set(song_data.keys())
    if not required_features.issubset(provided_features):
        missing_features = required_features - provided_features
        raise ValueError(f"Missing features: {missing_features}")
    
    # Extract and scale features
    song_features = np.array([[song_data[feature] for feature in features]])
    song_features_scaled = scaler.transform(song_features)  # Use the fitted scaler here
    
    # Make prediction
    prediction = model.predict(song_features_scaled)
    predicted_index = np.argmax(prediction)
    predicted_category = le.inverse_transform([predicted_index])[0]
    
    return predicted_category

# Make predictions for each song
for song in songs:
    song['category'] = predict_category(song)

with open('json/final.json', 'w') as file:
    json.dump(songs, file, indent=4)
