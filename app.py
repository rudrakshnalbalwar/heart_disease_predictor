from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the enhanced model and scaler
try:
    rf_model = joblib.load('rf_model.joblib')
    scaler = joblib.load('rf_scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    app.logger.info("Enhanced model loaded successfully")
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
            
        if request.is_json:
            # Handle API requests (JSON)
            data = request.get_json()
            input_data = pd.DataFrame(data, index=[0])
            # Add enhanced features
            input_data_enhanced = add_enhanced_features(input_data)
            final_features_scaled = scaler.transform(input_data_enhanced)
        else:
            # Handle form submissions (HTML form)
            feature_order = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
            
            # Extract features in the correct order
            features = []
            for feature_name in feature_order:
                if feature_name not in request.form:
                    return jsonify({'error': f'Missing field: {feature_name}'}), 400
                try:
                    value = float(request.form[feature_name])
                    features.append(value)
                except ValueError:
                    return jsonify({'error': f'Invalid value for {feature_name}'}), 400
            
            # Create DataFrame and add enhanced features
            input_df = pd.DataFrame([features], columns=feature_order)
            input_df_enhanced = add_enhanced_features(input_df)
            final_features_scaled = scaler.transform(input_df_enhanced)

        # Make predictions
        prediction = rf_model.predict(final_features_scaled)[0]
        risk_probability = rf_model.predict_proba(final_features_scaled)[0][1]
        
        # Enhanced risk determination
        if risk_probability >= 0.7:
            result = 'High risk of heart attack detected. Immediate medical consultation recommended.'
        elif risk_probability >= 0.5:
            result = 'Moderate-high risk of heart attack. Medical consultation recommended.'
        elif risk_probability >= 0.3:
            result = 'Moderate risk of heart attack. Regular monitoring advised.'
        else:
            result = 'Low risk of heart attack. Maintain healthy lifestyle.'

        risk = f'Risk percentage: {risk_probability*100:.1f}%'

        # Return response
        if request.is_json:
            return jsonify({
                'prediction': int(prediction),
                'risk_percentage': float(risk_probability*100),
                'result': result
            })
        else:
            return render_template('index.html', prediction_text=result, risk_percentage=risk)
            
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        error_msg = "An error occurred during prediction. Please check your input."
        
        if request.is_json:
            return jsonify({'error': error_msg}), 500
        else:
            return render_template('index.html', error=error_msg)

def add_enhanced_features(df):
    """Add the same enhanced features used during training."""
    df_enhanced = df.copy()
    
    # High cholesterol indicator (>250)
    df_enhanced['high_chol'] = (df['chol'] > 250).astype(int)
    
    # High blood pressure indicator (>140)
    df_enhanced['high_bp'] = (df['trestbps'] > 140).astype(int)
    
    # Low heart rate indicator (<130)
    df_enhanced['low_hr'] = (df['thalach'] < 130).astype(int)
    
    # High ST depression indicator (>2.0)
    df_enhanced['high_oldpeak'] = (df['oldpeak'] > 2.0).astype(int)
    
    # Combine risk factors
    df_enhanced['risk_score'] = (
        df_enhanced['high_chol'] + 
        df_enhanced['high_bp'] + 
        df_enhanced['low_hr'] + 
        df_enhanced['high_oldpeak'] +
        (df['cp'] == 3).astype(int) +  # Asymptomatic chest pain
        df['exang']  # Exercise induced angina
    )
    
    return df_enhanced

if __name__ == "__main__":
    app.run(debug=True)