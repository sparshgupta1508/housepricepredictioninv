from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import tensorflow as tf

app = Flask(__name__)

# Load your model and scaler
model = tf.keras.models.load_model(
    r'D:\Github_inv\housepricepredictioninv\house_price_prediction_model.h5',
    compile=False
)
scaler = pickle.load(open(r'D:\Github_inv\housepricepredictioninv\scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Extract features
        features = np.array([[
            float(data['bedrooms']),
            float(data['bathrooms']),
            float(data['sqft_living']),
            float(data['sqft_lot']),
            float(data['floors']),
            float(data['waterfront']),
            float(data['view']),
            float(data['condition']),
            float(data['grade']),
            float(data['sqft_above']),
            float(data['sqft_basement']),
            float(data['yr_built']),
            float(data['yr_renovated']),
            float(data['lat']),
            float(data['long'])
        ]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)
        price = float(prediction[0][0])

        # Return JSON response
        return jsonify({'predicted_price': price})

    except Exception as e:
        # Return error as JSON
        return jsonify({'error': str(e), 'predicted_price': 0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
