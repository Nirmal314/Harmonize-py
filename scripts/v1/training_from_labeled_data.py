import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv('predicted_songs_with_categories_vs.csv')

features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[features]
y = df['predicted_category']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(len(features),)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('model/models_final.h5')

# Function to predict category for new songs
def predict_category(song_data):
    # Ensure the song_data has all required features
    required_features = set(features)
    provided_features = set(song_data.keys())
    if not required_features.issubset(provided_features):
        missing_features = required_features - provided_features
        raise ValueError(f"Missing features: {missing_features}")
    
    # Extract and scale features
    song_features = np.array([[song_data[feature] for feature in features]])
    song_features_scaled = scaler.transform(song_features)
    
    # Make prediction
    prediction = model.predict(song_features_scaled)
    predicted_index = np.argmax(prediction)
    predicted_category = le.inverse_transform([predicted_index])[0]
    
    return predicted_category

# Example usage
new_song = {
            "danceability": 0.512,
            "energy": 0.662,
            "key": 3,
            "loudness": -6.797,
            "mode": 1,
            "speechiness": 0.0439,
            "acousticness": 0.0275,
            "instrumentalness": 0,
            "liveness": 0.118,
            "valence": 0.472,
            "tempo": 180.114
        }




predicted_category = predict_category(new_song)
print(f"Predicted category: {predicted_category}")