
Wind Forecasting Dashboard - MVP
================================

How to run:
1. Install dependencies:
   pip install flask pandas numpy scikit-learn tensorflow joblib

2. Place your trained models in /models/:
   - wind_forecast_model.keras
   - demand_forecast_model.keras
   And optionally scalers:
   - wind_scaler.pkl
   - demand_scaler.pkl
   Also you can place historical_energy_mix_2024.csv in /models/ for real historic baseline.

3. Run the app:
   python app.py

4. Open http://127.0.0.1:5000 in your browser.

Notes:
- The app will fallback to simple predictors if models are not present.
- Upload two CSVs (wind and demand) each containing at least 24 rows of hourly features.
- The templates expect reasonable column names; the app tries to be flexible.
