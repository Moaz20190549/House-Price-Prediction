from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

model = tf.keras.models.load_model('housing_price_model.h5')
scaler = joblib.load('scaler.save')

def preprocess_input(data):
    input_data = pd.DataFrame([data], columns=['Square_Feet', 'Bedrooms', 'Age'])

    features_to_normalize = ['Square_Feet', 'Age']
    input_data[features_to_normalize] = scaler.transform(input_data[features_to_normalize])

    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_data = preprocess_input(data)

    prediction = model.predict(input_data)
    
    predicted_price = np.exp(prediction[0][0])

    predicted_price = float(predicted_price)

    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
