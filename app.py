from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and label encoder
model = load_model('model/song_category_model_T_v3.h5')
scaler = joblib.load('model/scalers/scaler_T_v3.joblib')  # Load the pre-fitted scaler
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('model/label_encoders/label_encoder_classes_T_v3.npy', allow_pickle=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
   
    # Extract features from request data
    features = np.array([[data['danceability'], data['energy'], data['acousticness'], data['valence'], data['tempo']]])
   
    # Normalize the features using the pre-fitted scaler
    normalized_features = scaler.transform(features)
   
    # Make predictions
    prediction = model.predict(normalized_features)
    predicted_class = np.argmax(prediction, axis=1)
   
    # Get the predicted label
    predicted_label = label_encoder.inverse_transform([predicted_class])
    print(predicted_label)
    return jsonify({'predicted_category': predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True)