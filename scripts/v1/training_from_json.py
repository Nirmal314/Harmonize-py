import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import History

def prepare_data(data):
    features = np.array([[
        song['features']['danceability'],
        song['features']['energy'],
        song['features']['loudness'],
        song['features']['speechiness'],
        song['features']['acousticness'],
        song['features']['instrumentalness'],
        song['features']['liveness'],
        song['features']['valence'],
        song['features']['tempo'],
        song['features']['key'],
        song['features']['mode']
    ] for song in data])

    labels = np.array([song['category'] for song in data])
    
    #! Labels to numerical values
    label_map = {'happy': 0, 'sad': 1, 'calm': 2, 'energetic': 3, 'confident': 4}
    numerical_labels = np.array([label_map[label] for label in labels])
    
    return features, numerical_labels

def create_model(neurons=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(11,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neurons//2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neurons//4, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

labeled_data = pd.read_json('json/songs_with_categories.json')

#? features, labels
X, y = prepare_data(labeled_data.to_dict('records'))

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#! TRAINING (prev epochs = 100)
model = create_model(neurons=64, dropout_rate=0.2, learning_rate=0.001)
history = model.fit(X_train, y_train, epochs=162, batch_size=32, validation_split=0.2, verbose=1)

#? Evaluation
y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# model.save('model/model_vs.h5')

# Training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
