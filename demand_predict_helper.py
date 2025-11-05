"""
Helper function for making demand predictions with XGBoost model.
This function automatically handles scaling/unscaling so predictions are in original MW units.
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

def predict_demand(Temperature, Humidity, Timestamp, model_path='demand_forecast_xgboost.pkl', 
                   feature_scaler_path='demand_feature_scaler_xgb.pkl',
                   target_scaler_path='demand_target_scaler_xgb.pkl'):
    """
    Predict electricity demand in MW (original scale).
    
    Parameters:
    -----------
    Temperature : float or array-like
        Temperature value(s)
    Humidity : float or array-like
        Humidity value(s)
    Timestamp : str, datetime, or array-like
        Timestamp(s) - can be datetime object or string
    model_path : str
        Path to saved XGBoost model
    feature_scaler_path : str
        Path to feature scaler
    target_scaler_path : str
        Path to target scaler
    
    Returns:
    --------
    predictions : float or array
        Predicted demand in MW (original scale, no rescaling needed)
    """
    # Load model and scalers
    model = joblib.load(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    
    # Convert to pandas if needed
    if not isinstance(Timestamp, pd.Series):
        if isinstance(Timestamp, str):
            ts = pd.to_datetime(Timestamp)
        else:
            ts = pd.to_datetime(Timestamp)
        timestamps = pd.Series([ts] * len(np.array(Temperature).flatten()))
    else:
        timestamps = pd.to_datetime(Timestamp)
    
    # Extract time features
    hour = timestamps.dt.hour.values
    dayofweek = timestamps.dt.dayofweek.values
    month = timestamps.dt.month.values
    year = timestamps.dt.year.values
    dayofyear = timestamps.dt.dayofyear.values
    
    # Convert inputs to arrays
    temp = np.array(Temperature).flatten()
    hum = np.array(Humidity).flatten()
    
    # Create feature matrix
    features = np.column_stack([
        temp,  # Temperature
        hum,   # Humidity
        np.sin(2 * np.pi * hour / 24),  # hour_sin
        np.cos(2 * np.pi * hour / 24),  # hour_cos
        np.sin(2 * np.pi * dayofweek / 7),  # dayofweek_sin
        np.cos(2 * np.pi * dayofweek / 7),  # dayofweek_cos
        np.sin(2 * np.pi * month / 12),  # month_sin
        np.cos(2 * np.pi * month / 12),  # month_cos
        np.sin(2 * np.pi * dayofyear / 365.25),  # dayofyear_sin
        np.cos(2 * np.pi * dayofyear / 365.25),  # dayofyear_cos
    ])
    
    # Normalize year using saved parameters from training
    import os
    import json
    year_params_path = 'demand_year_params.json'
    if os.path.exists(year_params_path):
        with open(year_params_path, 'r') as f:
            year_params = json.load(f)
        year_min = year_params['year_min']
        year_max = year_params['year_max']
    else:
        # Fallback if file doesn't exist
        year_min = 2020
        year_max = 2025
    year_normalized = (year - year_min) / (year_max - year_min) if year_max > year_min else np.zeros_like(year)
    features = np.column_stack([features, year_normalized])
    
    # Scale features
    features_scaled = feature_scaler.transform(features)
    
    # Predict (model outputs scaled values)
    predictions_scaled = model.predict(features_scaled)
    
    # Inverse transform to get original MW scale
    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    
    # Return as float or array
    if len(predictions) == 1:
        return float(predictions[0])
    return predictions.flatten()

# Example usage:
# prediction = predict_demand(25.0, 65.0, '2024-01-15 14:00:00')
# print(f"Predicted demand: {prediction:.2f} MW")
