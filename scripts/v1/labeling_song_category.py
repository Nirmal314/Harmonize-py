import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

model = load_model('model/model_vs.h5')

data = pd.read_csv('cleaned_dataset.csv')

def prepare_data(data):
    features = data[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'mode']].values
    return features

X = prepare_data(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_pred_prob = model.predict(X_scaled)
y_pred = np.argmax(y_pred_prob, axis=-1)

label_map = {0: 'happy', 1: 'sad', 2: 'calm', 3: 'energetic', 4: 'confident'}
y_pred_labels = [label_map[pred] for pred in y_pred]

results = data.copy()
results['predicted_category'] = y_pred_labels
results.to_csv('predicted_songs_with_categories_vs.csv', index=False)
