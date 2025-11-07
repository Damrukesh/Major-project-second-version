import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from demand_predict_helper import predict_demand

# Set page config
st.set_page_config(
    page_title="Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; color: #42A5F5; margin-top: 1.5rem;}
    .prediction-box {background-color: #E3F2FD; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;}
    .prediction-value {font-size: 2rem; font-weight: bold; color: #0D47A1; text-align: center;}
    .metric-box {background-color: #E8F5E9; padding: 1rem; border-radius: 8px; text-align: center;}
    .stSlider > div > div > div > div {background-color: #1E88E5;}
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>⚡ Electricity Demand Forecasting</h1>", unsafe_allow_html=True)
st.write("""
This application uses an XGBoost model to predict electricity demand based on weather conditions and time features.
Adjust the sliders and select a date to see the predicted demand.
""")

# Create two columns for the input form
col1, col2 = st.columns(2)

with col1:
    # Date and time input
    st.markdown("### 📅 Select Date and Time")
    selected_date = st.date_input(
        "Date",
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31),
        value=datetime.now()
    )
    
    selected_time = st.time_input("Time", value=datetime.now().replace(hour=12, minute=0))
    
    # Combine date and time
    selected_datetime = datetime.combine(selected_date, selected_time)
    
    # Weather inputs
    st.markdown("### 🌡️ Weather Conditions")
    temperature = st.slider(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=50.0,
        value=25.0,
        step=0.5,
        help="Adjust the temperature in degrees Celsius"
    )
    
    humidity = st.slider(
        "Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
        help="Adjust the relative humidity percentage"
    )

with col2:
    # Prediction display
    st.markdown("### 📊 Prediction")
    
    # Add a placeholder for the prediction
    prediction_placeholder = st.empty()
    
    # Make prediction when inputs change
    try:
        prediction = predict_demand(
            Temperature=temperature,
            Humidity=humidity,
            Timestamp=selected_datetime
        )
        
        # Display prediction in a nice box
        with prediction_placeholder.container():
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-header'>Predicted Demand</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-value'>{prediction:,.0f} MW</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add some metrics
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Temperature", f"{temperature} °C")
            with metrics_col2:
                st.metric("Humidity", f"{humidity}%")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add a section for time series forecast
st.markdown("---")
st.markdown("## 📈 24-Hour Forecast")
st.write("See how the demand is expected to change over the next 24 hours with the current weather conditions:")

# Generate 24 hours of data
hours = 24
future_times = [selected_datetime + timedelta(hours=i) for i in range(hours)]
temps = [temperature for _ in range(hours)]  # Keep temperature constant
humids = [humidity for _ in range(hours)]    # Keep humidity constant

# Make predictions for each hour
try:
    predictions = []
    for i in range(hours):
        pred = predict_demand(
            Temperature=temps[i],
            Humidity=humids[i],
            Timestamp=future_times[i]
        )
        predictions.append(float(pred))
    
    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame({
        'Time': [t.strftime('%H:%M') for t in future_times],
        'Demand (MW)': predictions,
        'Temperature (°C)': temps,
        'Humidity (%)': humids
    })
    
    # Plot the forecast
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot demand on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Demand (MW)', color=color)
    ax1.plot(forecast_df['Time'], forecast_df['Demand (MW)'], color=color, marker='o', label='Demand')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create secondary y-axis for temperature
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Temperature (°C)', color=color)
    ax2.plot(forecast_df['Time'], forecast_df['Temperature (°C)'], color=color, linestyle='--', label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('24-Hour Demand Forecast')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Display the plot
    st.pyplot(fig)
    
    # Show the data table
    st.write("### Forecast Data")
    st.dataframe(forecast_df, hide_index=True, use_container_width=True)
    
    # Add download button for the forecast data
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast Data (CSV)",
        data=csv,
        file_name=f"demand_forecast_{selected_date.strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
    
except Exception as e:
    st.error(f"Error generating forecast: {str(e)}")

# Add some helpful information
st.markdown("---")
with st.expander("ℹ️ About this model"):
    st.write("""
    This demand forecasting model is built using XGBoost and considers:
    - Temperature and humidity as weather inputs
    - Time-based features (hour of day, day of week, month, etc.)
    - Cyclic encoding for periodic patterns
    
    The model is trained on historical demand data and provides predictions in megawatts (MW).
    """)

# Add footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
            "Electricity Demand Forecasting App | Built with Streamlit and XGBoost</div>", 
            unsafe_allow_html=True)
