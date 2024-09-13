import spotipy
from dotenv import load_dotenv
import os
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Load environment variables and set up Spotify client
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
data = []
names = []
urls = []
for track in tracks:
    song_id = track['track']['id']
    features = sp.audio_features(song_id)[0]
    if features is not None:
        data.append(features)
        names.append(track['track']['name'])
        urls.append(track['track']['external_urls']['spotify'])

# Convert to DataFrame
df = pd.DataFrame(data)

# # Select relevant features (make sure these match the features used in training)
# features = ['valence', 'energy', 'danceability', 'acousticness', 'loudness', 'tempo', 'instrumentalness', 'liveness']
# X = df[features]

# # Load the trained model and related components
# model = tf.keras.models.load_model('model/full_model.h5')
# with open('model/label_encoder.pkl', 'rb') as f:
#     le = pickle.load(f)
# with open('model/feature_scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Preprocess the data
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)
# X_scaled = scaler.transform(X_imputed)

# # Make predictions
# predictions = model.predict(X_scaled)
# predicted_classes = np.argmax(predictions, axis=1)
# predicted_emotions = le.inverse_transform(predicted_classes)

# # Create a DataFrame with results
# results_df = pd.DataFrame({
#     'Song Name': names,
#     'Song URL': urls,
#     'Predicted Emotion': predicted_emotions
# })

# # Print the results
# print(results_df.to_string(index=False))

# # Optionally, save the results to a CSV file
# results_df.to_csv('playlist_emotions.csv', index=False)
# print("\nResults have been saved to 'playlist_emotions.csv'")