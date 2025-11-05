# ============================================================
# Demand Forecasting with XGBoost (with Time Features)
# ============================================================
#
# IMPORTANT: This script trains a model that outputs predictions
# in ORIGINAL SCALE (MW units, same as your dataset).
# No additional rescaling needed when using the model!
#
# The helper function (demand_predict_helper.py) automatically
# handles all scaling/unscaling internally, so predictions are
# directly in MW and ready to display in Streamlit.
#
# ============================================================
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = 'demand_data_texas.csv'
TARGET_COL = 'Demand'
TEST_SIZE = 0.2  # 80% train, 20% test

# Model parameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}
# Note: eval_metric removed - XGBoost will use default RMSE for regression

MODEL_OUT = 'demand_forecast_xgboost.pkl'
FEATURE_SCALER_OUT = 'demand_feature_scaler_xgb.pkl'
TARGET_SCALER_OUT = 'demand_target_scaler_xgb.pkl'

# -----------------------------
# Load and preprocess data
# -----------------------------
print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Build timestamp if not already complete
if 'hour' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%b-%y', errors='coerce') + pd.to_timedelta(df['hour'], unit='h')
else:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Ensure time features exist (extract from timestamp if not in CSV)
if 'hour' not in df.columns:
    df['hour'] = df['Timestamp'].dt.hour
if 'dayofweek' not in df.columns:
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
if 'month' not in df.columns:
    df['month'] = df['Timestamp'].dt.month
if 'year' not in df.columns:
    df['year'] = df['Timestamp'].dt.year
if 'dayofyear' not in df.columns:
    df['dayofyear'] = df['Timestamp'].dt.dayofyear

# Force numeric for features and target
numeric_cols = ['Temperature', 'Humidity', 'hour', 'dayofweek', 'month', 'year', 'dayofyear', TARGET_COL]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Sort and clean
df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.dropna(subset=['Timestamp', 'Temperature', 'Humidity', TARGET_COL])

# Remove infinite values
df = df[np.isfinite(df[['Temperature', 'Humidity', TARGET_COL]]).all(axis=1)]

print(f"Data loaded: {len(df)} rows")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# -----------------------------
# Feature Engineering
# -----------------------------
print("\nEngineering features...")

# Base features
features = ['Temperature', 'Humidity']

# Add time features (cyclic encoding for periodic patterns)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

# Add cyclic features to feature list
features.extend([
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'dayofyear_sin', 'dayofyear_cos'
])

# Optional: Add year as numeric (or normalize it)
# For year, we can use it as-is or normalize relative to base year
YEAR_MIN = df['year'].min()
YEAR_MAX = df['year'].max()
if 'year' in df.columns:
    df['year_normalized'] = (df['year'] - YEAR_MIN) / (YEAR_MAX - YEAR_MIN) if YEAR_MAX > YEAR_MIN else 0
    features.append('year_normalized')

print(f"Features: {features}")
print(f"Total features: {len(features)}")

# Prepare data
X = df[features].values.astype('float32')
y = df[TARGET_COL].values.astype('float32').reshape(-1, 1)

# -----------------------------
# Train/Test Split (time-ordered)
# -----------------------------
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_timestamps = df['Timestamp'].iloc[:split_idx].values
test_timestamps = df['Timestamp'].iloc[split_idx:].values

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# -----------------------------
# Scaling
# -----------------------------
print("\nScaling features and target...")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train)
y_test_scaled = target_scaler.transform(y_test)

# -----------------------------
# Train XGBoost Model
# -----------------------------
print("\nTraining XGBoost model...")
print(f"Parameters: {XGBOOST_PARAMS}")

model = xgb.XGBRegressor(**XGBOOST_PARAMS)

# Train with early stopping
model.fit(
    X_train_scaled, y_train_scaled.ravel(),
    eval_set=[(X_train_scaled, y_train_scaled.ravel()), 
              (X_test_scaled, y_test_scaled.ravel())],
    verbose=True,
)

print("✅ Model training completed!")

# -----------------------------
# Evaluate
# -----------------------------
print("\nEvaluating model...")
y_pred_scaled = model.predict(X_test_scaled)
# IMPORTANT: Inverse transform predictions to get original MW scale
# All predictions from this model will be in MW (original dataset units)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_true = target_scaler.inverse_transform(y_test_scaled)

# Flatten for metrics
y_pred = y_pred.flatten()
y_true = y_true.flatten()

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print("\n" + "="*50)
print("Model Performance Metrics:")
print("="*50)
print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print("="*50)

# -----------------------------
# Feature Importance
# -----------------------------
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# -----------------------------
# Save Model and Scalers
# -----------------------------
print("\nSaving model and scalers...")
joblib.dump(model, MODEL_OUT)
joblib.dump(feature_scaler, FEATURE_SCALER_OUT)
joblib.dump(target_scaler, TARGET_SCALER_OUT)

print(f"✅ Model saved: {MODEL_OUT}")
print(f"✅ Feature scaler saved: {FEATURE_SCALER_OUT}")
print(f"✅ Target scaler saved: {TARGET_SCALER_OUT}")

# Save year normalization parameters for helper function
import json
year_params = {'year_min': int(YEAR_MIN), 'year_max': int(YEAR_MAX)}
with open('demand_year_params.json', 'w') as f:
    json.dump(year_params, f)
print(f"✅ Year normalization params saved: demand_year_params.json")

# -----------------------------
# Save Helper Function for Easy Predictions (for Streamlit use)
# -----------------------------
# Create a helper module content for easy predictions
helper_code = '''"""
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
'''

# Save helper function to a separate file
with open('demand_predict_helper.py', 'w') as f:
    f.write(helper_code)

print(f"✅ Helper function saved: demand_predict_helper.py")
print("\n" + "="*60)
print("IMPORTANT: Predictions are in ORIGINAL SCALE (MW)")
print("="*60)
print("The model outputs values in the same units as your dataset.")
print("No additional rescaling needed when using in Streamlit!")
print("="*60)

# -----------------------------
# Visualization: Training Loss
# -----------------------------
print("\nGenerating plots...")
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test RMSE')
ax.legend()
ax.set_xlabel('Boosting Round')
ax.set_ylabel('RMSE')
ax.set_title('XGBoost Training Progress')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('demand_xgboost_training_curve.png', dpi=300, bbox_inches='tight')
print("✅ Training curve saved: demand_xgboost_training_curve.png")
plt.close()

# -----------------------------
# Visualization: Feature Importance
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance.head(15)
ax.barh(range(len(top_features)), top_features['importance'].values)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values)
ax.set_xlabel('Feature Importance')
ax.set_title('Top 15 Feature Importance (XGBoost)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('demand_xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance plot saved: demand_xgboost_feature_importance.png")
plt.close()

# -----------------------------
# Visualization: Actual vs Predicted (Sample)
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Time series comparison
sample_size = min(500, len(y_test))
axes[0].plot(test_timestamps[:sample_size], y_true[:sample_size], 
             label='Actual', linewidth=2, alpha=0.8, color='#2E86AB')
axes[0].plot(test_timestamps[:sample_size], y_pred[:sample_size], 
             label='Predicted', linewidth=2, alpha=0.8, color='#A23B72', linestyle='--')
axes[0].set_title(f'Demand Forecast: Actual vs Predicted (First {sample_size} samples)', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Timestamp', fontsize=12)
axes[0].set_ylabel('Demand (MW)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Scatter plot
axes[1].scatter(y_true, y_pred, alpha=0.5, s=10, color='#A23B72')
axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Demand (MW)', fontsize=12)
axes[1].set_ylabel('Predicted Demand (MW)', fontsize=12)
axes[1].set_title('Actual vs Predicted: Scatter Plot', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f} MW', 
             transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('demand_xgboost_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Predictions plot saved: demand_xgboost_predictions.png")
plt.close()

print("\n" + "="*50)
print("Training completed successfully!")
print("="*50)

