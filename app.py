from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the corrected model (use the correct file names)
try:
    rf_model = joblib.load('heart_model.joblib')
    scaler = joblib.load('heart_scaler.joblib')
    app.logger.info("Corrected model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    rf_model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if rf_model is None or scaler is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
            
        # Extract features in correct order (original 13 features only)
        feature_order = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        
        if request.is_json:
            data = request.get_json()
            features = [float(data[feature]) for feature in feature_order]
        else:
            features = []
            for feature_name in feature_order:
                if feature_name not in request.form:
                    return jsonify({'error': f'Missing field: {feature_name}'}), 400
                try:
                    value = float(request.form[feature_name])
                    features.append(value)
                except ValueError:
                    return jsonify({'error': f'Invalid value for {feature_name}'}), 400
        
        # Create DataFrame and predict (no enhanced features - keep it simple)
        input_df = pd.DataFrame([features], columns=feature_order)
        scaled_features = scaler.transform(input_df)
        
        prediction = rf_model.predict(scaled_features)[0]
        risk_probability = rf_model.predict_proba(scaled_features)[0][1]  # 1 = disease (corrected)
        
        # Determine risk level
        if risk_probability >= 0.7:
            result = 'High risk of heart disease. Please consult a cardiologist immediately.'
            risk_level = 'High'
        elif risk_probability >= 0.5:
            result = 'Moderate-high risk of heart disease. Medical consultation recommended.'
            risk_level = 'Moderate-High'
        elif risk_probability >= 0.3:
            result = 'Moderate risk of heart disease. Regular monitoring recommended.'
            risk_level = 'Moderate'
        else:
            result = 'Low risk of heart disease. Maintain healthy lifestyle.'
            risk_level = 'Low'

        response_data = {
            'prediction': int(prediction),
            'risk_percentage': float(risk_probability * 100),
            'risk_level': risk_level,
            'result': result
        }

        if request.is_json:
            return jsonify(response_data)
        else:
            return render_template('index.html', 
                                 prediction_text=result, 
                                 risk_percentage=f'Heart Disease Risk: {risk_probability*100:.1f}%')
            
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        error_msg = "An error occurred during prediction. Please check your input."
        
        if request.is_json:
            return jsonify({'error': error_msg}), 500
        else:
            return render_template('index.html', error=error_msg)

if __name__ == "__main__":
    app.run(debug=True)