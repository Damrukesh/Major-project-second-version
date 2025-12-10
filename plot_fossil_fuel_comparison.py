"""
Script to plot comparison of fossil fuel energy:
1) Actual fossil fuel energies needed hourly (Fossil_Required_MW)
2) Forecasted fossil fuels needed (Demand - Wind - Hydro - Solar)
3) Actual fossil fuels produced (Fossil_Actual_MW)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle

# Load the data
df = pd.read_csv('datasets/final_portfolio.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Load the trained models for forecasting
print("Loading models...")
with open('demand_forecast_xgboost.pkl', 'rb') as f:
    demand_model = pickle.load(f)

with open('wind forecast/wind_forecast_model.pkl', 'rb') as f:
    wind_model = pickle.load(f)

# Load scalers
with open('demand_feature_scaler_xgb.pkl', 'rb') as f:
    demand_feature_scaler = pickle.load(f)

with open('demand_target_scaler_xgb.pkl', 'rb') as f:
    demand_target_scaler = pickle.load(f)

with open('wind forecast/scaler.pkl', 'rb') as f:
    wind_scaler = pickle.load(f)

# Prepare features for demand forecasting
def prepare_demand_features(df):
    """Prepare features for demand forecasting"""
    features = pd.DataFrame()
    features['hour'] = df['Timestamp'].dt.hour
    features['day_of_week'] = df['Timestamp'].dt.dayofweek
    features['month'] = df['Timestamp'].dt.month
    features['day_of_year'] = df['Timestamp'].dt.dayofyear
    
    # Add lag features if needed (assuming we have historical data)
    if 'Demand_MW' in df.columns:
        features['demand_lag_1'] = df['Demand_MW'].shift(1)
        features['demand_lag_24'] = df['Demand_MW'].shift(24)
        features['demand_lag_168'] = df['Demand_MW'].shift(168)
    
    return features

# Prepare features for wind forecasting
def prepare_wind_features(df_wind):
    """Prepare features for wind forecasting"""
    features = pd.DataFrame()
    features['hour'] = df_wind['Timestamp'].dt.hour
    features['day_of_week'] = df_wind['Timestamp'].dt.dayofweek
    features['month'] = df_wind['Timestamp'].dt.month
    
    # Add wind-specific features if available
    if 'Wind_MW' in df_wind.columns:
        features['wind_lag_1'] = df_wind['Wind_MW'].shift(1)
        features['wind_lag_24'] = df_wind['Wind_MW'].shift(24)
    
    return features

# Make predictions
print("Generating forecasts...")

# Forecast demand
demand_features = prepare_demand_features(df)
demand_features = demand_features.fillna(method='bfill').fillna(0)  # Handle NaN values
demand_features_scaled = demand_feature_scaler.transform(demand_features)
demand_forecast_scaled = demand_model.predict(demand_features_scaled)
demand_forecast = demand_target_scaler.inverse_transform(demand_forecast_scaled.reshape(-1, 1)).flatten()

# Forecast wind
wind_features = prepare_wind_features(df)
wind_features = wind_features.fillna(method='bfill').fillna(0)  # Handle NaN values
wind_features_scaled = wind_scaler.transform(wind_features)
wind_forecast = wind_model.predict(wind_features_scaled)

# Use actual values for hydro and solar (assuming they are predictable or use forecasts if available)
hydro_values = df['Hydro_MW'].values
solar_values = df['Solar_MW'].values

# Calculate forecasted fossil fuel needs
# Forecasted fossil fuels needed = forecasted demand - forecasted wind - hydro - solar
forecasted_fossil_needed = demand_forecast - wind_forecast - hydro_values - solar_values

# Extract actual values
actual_fossil_required = df['Fossil_Required_MW'].values
actual_fossil_produced = df['Fossil_Actual_MW'].values
timestamps = df['Timestamp'].values

# Select a subset for better visualization (e.g., first week of January)
start_idx = 0
end_idx = 24 * 7  # One week of hourly data

timestamps_subset = timestamps[start_idx:end_idx]
actual_fossil_required_subset = actual_fossil_required[start_idx:end_idx]
forecasted_fossil_needed_subset = forecasted_fossil_needed[start_idx:end_idx]
actual_fossil_produced_subset = actual_fossil_produced[start_idx:end_idx]

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the three lines
ax.plot(timestamps_subset, actual_fossil_required_subset, 
        label='1) Actual Fossil Fuel Required (MW)', 
        color='blue', linewidth=2, alpha=0.8)

ax.plot(timestamps_subset, forecasted_fossil_needed_subset, 
        label='2) Forecasted Fossil Fuel Needed (Demand - Wind - Hydro - Solar)', 
        color='orange', linewidth=2, alpha=0.8, linestyle='--')

ax.plot(timestamps_subset, actual_fossil_produced_subset, 
        label='3) Actual Fossil Fuel Produced (MW)', 
        color='green', linewidth=2, alpha=0.8)

# Formatting
ax.set_xlabel('Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Power (MW)', fontsize=14, fontweight='bold')
ax.set_title('Fossil Fuel Energy Comparison - First Week of January 2024', 
             fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Tight layout
plt.tight_layout()

# Save the plot
output_filename = 'fossil_fuel_comparison_plot.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{output_filename}'")

# Show the plot
plt.show()

# Print some statistics
print("\n" + "="*60)
print("STATISTICS FOR THE PLOTTED PERIOD")
print("="*60)
print(f"\nActual Fossil Fuel Required:")
print(f"  Mean: {np.mean(actual_fossil_required_subset):.2f} MW")
print(f"  Min:  {np.min(actual_fossil_required_subset):.2f} MW")
print(f"  Max:  {np.max(actual_fossil_required_subset):.2f} MW")

print(f"\nForecasted Fossil Fuel Needed:")
print(f"  Mean: {np.mean(forecasted_fossil_needed_subset):.2f} MW")
print(f"  Min:  {np.min(forecasted_fossil_needed_subset):.2f} MW")
print(f"  Max:  {np.max(forecasted_fossil_needed_subset):.2f} MW")

print(f"\nActual Fossil Fuel Produced:")
print(f"  Mean: {np.mean(actual_fossil_produced_subset):.2f} MW")
print(f"  Min:  {np.min(actual_fossil_produced_subset):.2f} MW")
print(f"  Max:  {np.max(actual_fossil_produced_subset):.2f} MW")

# Calculate forecast error
forecast_error = forecasted_fossil_needed_subset - actual_fossil_required_subset
mae = np.mean(np.abs(forecast_error))
rmse = np.sqrt(np.mean(forecast_error**2))

print(f"\nForecast Performance:")
print(f"  MAE (Mean Absolute Error): {mae:.2f} MW")
print(f"  RMSE (Root Mean Square Error): {rmse:.2f} MW")
print("="*60)
