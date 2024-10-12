from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models, encoders, and scalers
business_model = pickle.load(open('models/business_model.pkl', 'rb'))
personal_model = pickle.load(open('models/personal_model.pkl', 'rb'))

business_scaler = pickle.load(open('models/business_scaler.pkl', 'rb'))
personal_scaler = pickle.load(open('models/personal_scaler.pkl', 'rb'))

# Load encoders for business
business_encoders = {}
for feature in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    business_encoders[feature] = pickle.load(open(f'models/business_{feature}_encoder.pkl', 'rb'))

# Load encoders for personal
personal_encoders = {}
for feature in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    personal_encoders[feature] = pickle.load(open(f'models/personal_{feature}_encoder.pkl', 'rb'))

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
