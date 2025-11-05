# ============================================================
# Demand Model Comparison Plot: Actual vs Predicted
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Load Model and Scalers
# -----------------------------
print("Loading model and scalers...")
MODEL_PATH = "demand_forecast_lstm.keras"
TARGET_SCALER_PATH = "target_scaler.pkl"
FEATURE_SCALER_PATH = "feature_scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
target_scaler = joblib.load(TARGET_SCALER_PATH)
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
print("✅ Model and scalers loaded successfully!")

# -----------------------------
# Load and preprocess data (same as training)
# -----------------------------
print("Loading and preprocessing data...")
CSV_PATH = 'demand_data_texas.csv'
FEATURE_COLS = ['Temperature', 'Humidity']
TARGET_COL = 'Demand'
LOOKBACK = 24
TEST_SIZE = 0.2

df = pd.read_csv(CSV_PATH)

# Build timestamp (same as training script)
if 'hour' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%b-%y', errors='coerce') + pd.to_timedelta(df['hour'], unit='h')
else:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Force numeric for features and target
for col in FEATURE_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Sort and drop rows with missing core fields
df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.dropna(subset=['Timestamp'] + FEATURE_COLS + [TARGET_COL])

# Remove inf/-inf if any
df = df[np.isfinite(df[FEATURE_COLS + [TARGET_COL]]).all(axis=1)]

print(f"Data loaded: {len(df)} rows")

# -----------------------------
# Build sequences (same as training)
# -----------------------------
features_all = df[FEATURE_COLS].values.astype('float32')
target_all = df[[TARGET_COL]].values.astype('float32')

def create_dataset(features, target, lookback):
    X, y = [], []
    indices = []  # Store original indices for timestamp mapping
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])  # shape (1,)
        indices.append(i)  # Store the index of the target
    return np.array(X, dtype='float32'), np.array(y, dtype='float32'), indices

X_all, y_all, indices = create_dataset(features_all, target_all, LOOKBACK)

# Time-ordered split (same as training: 80/20)
split = int(len(X_all) * (1 - TEST_SIZE))
X_test = X_all[split:]
y_test = y_all[split:]
test_indices = indices[split:]

print(f"Test set size: {len(X_test)} samples")

# -----------------------------
# Scale test data (same as training)
# -----------------------------
n_steps, n_feats = X_test.shape[1], X_test.shape[2]

# Flatten time dimension to fit scaler
X_test_2d = X_test.reshape(-1, n_feats)

# Scale features using previously fitted scaler
X_test_scaled_2d = feature_scaler.transform(X_test_2d)
X_test_scaled = X_test_scaled_2d.reshape(-1, n_steps, n_feats)

# Scale target using previously fitted scaler
y_test_scaled = target_scaler.transform(y_test)

# -----------------------------
# Make predictions
# -----------------------------
print("Making predictions on test set...")
y_pred_scaled = model.predict(X_test_scaled, verbose=0)

# -----------------------------
# Inverse transform to get actual values
# -----------------------------
y_test_actual = target_scaler.inverse_transform(y_test_scaled)
y_pred_actual = target_scaler.inverse_transform(y_pred_scaled)

# Flatten if needed
y_test_actual = y_test_actual.flatten()
y_pred_actual = y_pred_actual.flatten()

# Get timestamps for test set
test_timestamps = df["Timestamp"].iloc[test_indices].values

# -----------------------------
# Calculate metrics
# -----------------------------
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n" + "="*50)
print("Model Performance Metrics:")
print("="*50)
print(f"RMSE: {rmse:.2f} MW")
print(f"MAE:  {mae:.2f} MW")
print(f"R²:   {r2:.4f}")
print("="*50)

# -----------------------------
# Create comparison plots
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Actual vs Predicted (overlapped)
axes[0].plot(test_timestamps, y_test_actual, label='Actual', linewidth=2, alpha=0.8, color='#2E86AB')
axes[0].plot(test_timestamps, y_pred_actual, label='Predicted', linewidth=2, alpha=0.8, color='#A23B72', linestyle='--')
axes[0].set_title('Electricity Demand: Actual vs Predicted (Test Set)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Timestamp', fontsize=12)
axes[0].set_ylabel('Demand (MW)', fontsize=12)
axes[0].legend(fontsize=11, loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Side-by-side comparison
x_pos = np.arange(len(test_timestamps))
width = 0.35

# Limit to first 200 points for better visibility (or use a subset)
plot_limit = min(200, len(test_timestamps))
axes[1].bar(x_pos[:plot_limit] - width/2, y_test_actual[:plot_limit], 
            width, label='Actual', alpha=0.7, color='#2E86AB')
axes[1].bar(x_pos[:plot_limit] + width/2, y_pred_actual[:plot_limit], 
            width, label='Predicted', alpha=0.7, color='#A23B72')

axes[1].set_title(f'Electricity Demand: Actual vs Predicted Comparison (First {plot_limit} samples)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sample Index', fontsize=12)
axes[1].set_ylabel('Demand (MW)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

# Add metrics text box
metrics_text = f'RMSE: {rmse:.2f} MW\nMAE: {mae:.2f} MW\nR²: {r2:.4f}'
axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('demand_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Plot saved as 'demand_model_comparison.png'")
plt.show()

# -----------------------------
# Optional: Create a scatter plot for correlation
# -----------------------------
fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(y_test_actual, y_pred_actual, alpha=0.5, s=20, color='#A23B72')
ax.plot([y_test_actual.min(), y_test_actual.max()], 
        [y_test_actual.min(), y_test_actual.max()], 
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Demand (MW)', fontsize=12)
ax.set_ylabel('Predicted Demand (MW)', fontsize=12)
ax.set_title('Actual vs Predicted: Scatter Plot', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('demand_model_scatter.png', dpi=300, bbox_inches='tight')
print("✅ Scatter plot saved as 'demand_model_scatter.png'")
plt.show()

