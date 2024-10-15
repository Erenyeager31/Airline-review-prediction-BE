from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define the models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Helper function to load a pickle file
def load_pickle(filename):
    return pickle.load(open(os.path.join(models_dir, filename), 'rb'))

# Load models
business_model = load_pickle('business_model.pkl')
personal_model = load_pickle('personal_model.pkl')

# Load scalers
business_scaler = load_pickle('business_scaler.pkl')
personal_scaler = load_pickle('personal_scaler.pkl')

# Load encoders for business and personal
def load_encoders(prefix):
    encoders = {}
    for feature in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        encoders[feature] = load_pickle(f'{prefix}_{feature}_encoder.pkl')
    return encoders

business_encoders = load_encoders('business')
personal_encoders = load_encoders('personal')

# Preprocessing function for business model
def preprocess_business_features(features):
    # Create DataFrame from input features
    df_business = pd.DataFrame([features])

    # Apply label encoding to categorical features
    for feature in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        df_business[feature] = business_encoders[feature].transform(df_business[feature])

    # Apply scaling to numerical features
    numerical_features = ['Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay', 
                          'Departure and Arrival Time Convenience', 'On-board Service', 'Seat Comfort', 
                          'Leg Room Service', 'Cleanliness', 'Food and Drink', 'In-flight Service', 
                          'In-flight Wifi Service', 'In-flight Entertainment', 'Baggage Handling']
    df_business[numerical_features] = business_scaler.transform(df_business[numerical_features])

    return df_business

# Preprocessing function for personal model
def preprocess_personal_features(features):
    # Create DataFrame from input features
    df_personal = pd.DataFrame([features])

    # Apply label encoding to categorical features
    for feature in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
        df_personal[feature] = personal_encoders[feature].transform(df_personal[feature])

    # Apply scaling to numerical features
    numerical_features = ['Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay', 
                          'Departure and Arrival Time Convenience', 'On-board Service', 'Seat Comfort', 
                          'Leg Room Service', 'Cleanliness', 'Food and Drink', 'In-flight Service', 
                          'In-flight Wifi Service', 'In-flight Entertainment', 'Baggage Handling']
    df_personal[numerical_features] = personal_scaler.transform(df_personal[numerical_features])

    return df_personal

# Route for business model prediction
@app.route('/predict/business', methods=['POST'])
def predict_business():
    try:
        # Get the features from the request
        data = request.get_json()

        # Preprocess the features
        processed_features = preprocess_business_features(data)

        # Make prediction
        prediction = business_model.predict(processed_features)

        # Convert prediction to a readable format
        result = 'Satisfied' if prediction[0] == 1 else 'Neutral or Dissatisfied'

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

# Route for personal model prediction
@app.route('/predict/personal', methods=['POST'])
def predict_personal():
    try:
        # Get the features from the request
        data = request.get_json()

        # Preprocess the features
        processed_features = preprocess_personal_features(data)

        # Make prediction
        prediction = personal_model.predict(processed_features)

        # Convert prediction to a readable format
        result = 'Satisfied' if prediction[0] == 1 else 'Neutral or Dissatisfied'

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
