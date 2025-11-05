import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Load Model and Scalers
# ----------------------------
MODEL_PATH = "demand_forecast_lstm.keras"
TARGET_SCALER_PATH = "target_scaler.pkl"
FEATURE_SCALER_PATH = "feature_scaler.pkl"

@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    return model, target_scaler, feature_scaler

try:
    model, target_scaler, feature_scaler = load_model_and_scalers()
except FileNotFoundError as e:
    st.error(f"Missing required files: {e}. Please run the training script first.")
    st.stop()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("⚡ Electricity Demand Forecasting (Batch Mode)")
st.write("Upload a CSV file containing Temperature and Humidity for multiple hours to forecast electricity demand for each row.")

st.markdown("""
**Expected CSV format:**  
`Temperature, Humidity`
""")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        required_cols = ["Temperature", "Humidity"]
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain 'Temperature' and 'Humidity' columns.")
        else:
            # Clean and scale
            df = df.dropna(subset=required_cols)
            df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna()

            # Scale features
            features_scaled = feature_scaler.transform(df[required_cols].astype('float32'))

            # Predict for each row (independent inference)
            X_input = features_scaled.reshape(features_scaled.shape[0], 1, len(required_cols))
            y_pred_scaled = model.predict(X_input, verbose=0)

            # Inverse transform predictions
            y_pred_mw = target_scaler.inverse_transform(y_pred_scaled)

            # Add predictions to dataframe
            df["Predicted_Demand_MW"] = y_pred_mw
            st.subheader("📈 Forecast Results")
            st.dataframe(df)

            # Plot chart
            st.line_chart(df["Predicted_Demand_MW"])

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "predicted_demand.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.exception(e)

st.markdown("---")
st.caption("Developed for Final Year Project — Wind Power & Energy Management System")
