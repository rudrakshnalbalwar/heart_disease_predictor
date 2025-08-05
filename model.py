import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_simple_robust_model():
    """Create a simple but robust model that generalizes well."""
    
    # Load data
    data = pd.read_csv('heart.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    # INVERT THE TARGET LABELS TO MATCH MEDICAL INTUITION
    # Original: 0=disease, 1=no disease
    # Fixed: 0=no disease, 1=disease
    y_corrected = 1 - y
    
    print("Creating simple but robust model with CORRECTED targets...")
    print("Fixed target meaning: 0=no disease, 1=disease")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_corrected, test_size=0.3, random_state=42, stratify=y_corrected
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple Random Forest with regularization to prevent overfitting
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("="*50)
    print("CORRECTED MODEL RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_names = list(X.columns)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model
    joblib.dump(rf_model, 'corrected_model.joblib')
    joblib.dump(scaler, 'corrected_scaler.joblib')
    joblib.dump(feature_names, 'corrected_features.joblib')
    
    print("\nCorrected model saved!")
    return rf_model, scaler

def predict_corrected(input_data, feature_names):
    """Make predictions using the corrected model."""
    
    # Load models
    model = joblib.load('corrected_model.joblib')
    scaler = joblib.load('corrected_scaler.joblib')
    
    # Create DataFrame
    df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale and predict
    scaled_features = scaler.transform(df)
    risk_probability = model.predict_proba(scaled_features)[0][1]  # Now 1 = disease
    
    return risk_probability

if __name__ == "__main__":
    model, scaler = create_simple_robust_model()