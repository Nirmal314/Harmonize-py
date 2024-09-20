import spotipy
from dotenv import load_dotenv
import os
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

model = load_model('model/song_category_model_T_v3.h5')

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
        features.pop('key')
        features.pop('loudness')
        features.pop('mode')
        features.pop('speechiness')
        features.pop('instrumentalness')
        features.pop('liveness')
        # features.pop('tempo')

        song["features"] = features

        songs.append(song)


with open('json/songs_with_categories.json', 'r') as file:
    songs_json = json.load(file)

df = pd.DataFrame([{**song, **song['features']} for song in songs_json])

# songs_json_filtered = [
#     {feature: song['features'][feature] for feature in selected_features}
#     for song in songs_json
# ]

selected_features = ["danceability", "energy", "acousticness", "valence", "tempo"]
X_new = df[selected_features]

# Normalize the features
scaler = MinMaxScaler()
X_new_normalized = scaler.fit_transform(X_new)

# Make predictions
predictions = model.predict(X_new_normalized)
predicted_classes = np.argmax(predictions, axis=1)

# If you want to convert the predicted class indices back to labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('model/label_encoders/label_encoder_classes_T_v3.npy', allow_pickle=True)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

df['predicted_category'] = predicted_labels

# Create a list of dictionaries with the results
results = df.to_dict('records')
count = 0
# for result in results:
#     print(f"Song: {result['name']}")
#     print(f"Original category: {result['category']}")
#     print(f"Predicted category: {result['predicted_category']}")
#     if result['category'] != result['predicted_category']:
#         count = count + 1

#     print("---")

# accuracy = 100 - (count / len(results) * 100)

# # Print the accuracy
# print(f"Accuracy: {accuracy:.2f}%")

# Initialize a dictionary to hold counts
category_counts = {category: {'correct': 0, 'total': 0} for category in df['category'].unique()}

# Count correct predictions per category
for result in results:
    print(f"Song: {result['name']}")
    print(f"Original category: {result['category']}")
    print(f"Predicted category: {result['predicted_category']}")
    print("--------")
    category = result['category']
    category_counts[category]['total'] += 1
    if category == result['predicted_category']:
        category_counts[category]['correct'] += 1

# Calculate accuracy for each category
accuracy_per_category = {
    category: (counts['correct'] / counts['total']) * 100 if counts['total'] > 0 else 0
    for category, counts in category_counts.items()
}

# Prepare data for plotting
accuracy_df = pd.DataFrame(list(accuracy_per_category.items()), columns=['Category', 'Accuracy'])

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(accuracy_df['Category'], accuracy_df['Accuracy'], color='skyblue')
plt.title('Model Prediction Accuracy by Category')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.show()