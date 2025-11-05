import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Load Model and Scaler
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("wind_forecast_model.keras", compile=False)
    scaler = joblib.load("wind_scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("💨 Wind Power Forecasting (Next 24 Hours)")
st.write("Upload your latest **wind turbine data CSV** to forecast the next 24 hours of generated power.")

st.markdown("""
**Expected CSV Columns:**  
`Time stamp`, `System power generated(kW)`, `Wind speed(m/s)`, `Wind direction(deg)`, `Pressure(atm)`, `Air temperature 'C`, `windfarm power(MW)`
""")

uploaded_file = st.file_uploader("📁 Upload your wind dataset", type=["csv"])

if uploaded_file:
    try:
        # ------------------------------------------------------------
        # Load and preprocess
        # ------------------------------------------------------------
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
        df["Timestamp"] = pd.to_datetime(df["Time_stamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)

        features = ["Wind_speed", "Wind_direction", "Pressure", "Air_temperature"]
        target = "windfarm_power"

        # Scale using previously fitted scaler
        scaled = scaler.transform(df[features + [target]])
        scaled_df = pd.DataFrame(scaled, columns=features + [target])

        st.subheader("✅ Data loaded successfully!")
        st.dataframe(df.head())

        # ------------------------------------------------------------
        # Forecast next 24 hours
        # ------------------------------------------------------------
        SEQ_LEN = 24
        last_seq = scaled_df[features].values[-SEQ_LEN:]
        X_input = np.expand_dims(last_seq, axis=0)

        future_preds_scaled = []
        input_seq = X_input.copy()

        for _ in range(24):
            next_pred = model.predict(input_seq)[0][0]
            future_preds_scaled.append(next_pred)

            next_row = input_seq[0, -1, :].copy()
            input_seq = np.append(input_seq[:, 1:, :], [[next_row]], axis=1)

        # Convert back to actual kW scale
        max_kw = scaler.data_max_[-1]  # last feature is target
        future_preds = np.array(future_preds_scaled) * max_kw

        # Build forecast dataframe
        future_timestamps = pd.date_range(df["Timestamp"].iloc[-1] + pd.Timedelta(hours=1),
                                          periods=24, freq="H")
        forecast_df = pd.DataFrame({"Timestamp": future_timestamps,
                                    "Predicted_Power_kW": future_preds.flatten()})

        # ------------------------------------------------------------
        # Display & download results
        # ------------------------------------------------------------
        st.subheader("🔮 24-Hour Forecast")
        st.line_chart(forecast_df.set_index("Timestamp"))
        st.dataframe(forecast_df)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV",
                           data=csv,
                           file_name="wind_forecast_24h.csv",
                           mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.caption("Developed for Final Year Project — Wind Energy Forecasting & Carbon-Aware Management")
