
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os, io, joblib, numpy as np, pandas as pd, json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.secret_key = "dev-secret-key"  # change for production

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper: safe model loader
def load_keras_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print("Model load error:", e)
        return None

# Simple fallback predictor if model not available: repeat last value or mean
def fallback_predict_series(values, steps=24):
    # values: pandas Series of recent target values
    last = float(values.iloc[-24:].mean()) if len(values) >= 24 else float(values.mean())
    return [last]*steps

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Expect two files: wind_csv and demand_csv
    wind_file = request.files.get("wind_csv")
    demand_file = request.files.get("demand_csv")
    if not wind_file or not demand_file:
        flash("Please upload both Wind and Demand CSV files.", "danger")
        return redirect(url_for("index"))

    wind_path = os.path.join(UPLOAD_FOLDER, "uploaded_wind.csv")
    demand_path = os.path.join(UPLOAD_FOLDER, "uploaded_demand.csv")
    wind_file.save(wind_path)
    demand_file.save(demand_path)

    # Load as DataFrames (robust parsing)
    try:
        wind_df = pd.read_csv(wind_path)
        demand_df = pd.read_csv(demand_path)
    except Exception as e:
        flash(f"Error reading CSVs: {e}", "danger")
        return redirect(url_for("index"))

    # Basic column normalization: replace spaces with underscores and lowercase
    wind_df.columns = [c.strip().replace(" ", "_").lower() for c in wind_df.columns]
    demand_df.columns = [c.strip().replace(" ", "_").lower() for c in demand_df.columns]

    # Ensure there are at least 24 rows for inference
    if len(wind_df) < 24 or len(demand_df) < 24:
        flash("Each uploaded CSV must contain at least 24 hourly records (past 24 hours).", "danger")
        return redirect(url_for("index"))

    # Try to load models & scalers if present
    wind_model = None
    demand_model = None
    wind_scaler = None
    demand_scaler = None

    wind_model_path = os.path.join(MODELS_FOLDER, "wind_forecast_model.keras")
    demand_model_path = os.path.join(MODELS_FOLDER, "demand_forecast_model.keras")
    wind_scaler_path = os.path.join(MODELS_FOLDER, "wind_scaler.pkl")
    demand_scaler_path = os.path.join(MODELS_FOLDER, "demand_scaler.pkl")

    if os.path.exists(wind_model_path):
        wind_model = load_keras_model(wind_model_path)
    if os.path.exists(demand_model_path):
        demand_model = load_keras_model(demand_model_path)
    if os.path.exists(wind_scaler_path):
        try:
            wind_scaler = joblib.load(wind_scaler_path)
        except: wind_scaler = None
    if os.path.exists(demand_scaler_path):
        try:
            demand_scaler = joblib.load(demand_scaler_path)
        except: demand_scaler = None

    # Determine target column names heuristically
    def pick_target(df, candidates):
        cols = df.columns.tolist()
        for c in candidates:
            if c in cols:
                return c
        return cols[-1]  # fallback to last column

    wind_target = pick_target(wind_df, ["windfarm_power", "windfarm_power_mw", "system_power_generated", "system_power_generated_kw", "power"])
    demand_target = pick_target(demand_df, ["demand", "total_demand_mw", "load", "system_load", "demand_mw"])

    # If models available, use them; else fallback
    steps = 24

    # WIND prediction
    if wind_model is not None and wind_scaler is not None:
        try:
            # select numeric features (last 24 rows)
            Xw = wind_df.select_dtypes(include=[float, int]).iloc[-24:]
            # scale - assume scaler fit on same column order; otherwise fallback
            Xw_scaled = wind_scaler.transform(Xw)
            Xw_seq = Xw_scaled.reshape(1, Xw_scaled.shape[0], Xw_scaled.shape[1])
            preds_w_scaled = wind_model.predict(Xw_seq)[0]
            # inverse transform target: assume scaler had target as last col
            # If scaler has attributes, try to invert robustly
            # Here we assume target was the last column of scaler input
            max_target = wind_scaler.data_max_[-1] if hasattr(wind_scaler, "data_max_") else Xw.values[:, -1].max()
            preds_w = np.array(preds_w_scaled).reshape(-1,1) * max_target
            wind_forecast = [float(x) for x in preds_w.flatten()]
        except Exception as e:
            print("Wind model predict error:", e)
            wind_forecast = fallback_predict_series(wind_df[wind_target], steps=steps)
    else:
        wind_forecast = fallback_predict_series(wind_df[wind_target], steps=steps)

    # DEMAND prediction
    if demand_model is not None and demand_scaler is not None:
        try:
            Xd = demand_df.select_dtypes(include=[float, int]).iloc[-24:]
            Xd_scaled = demand_scaler.transform(Xd)
            Xd_seq = Xd_scaled.reshape(1, Xd_scaled.shape[0], Xd_scaled.shape[1])
            preds_d_scaled = demand_model.predict(Xd_seq)[0]
            max_target_d = demand_scaler.data_max_[-1] if hasattr(demand_scaler, "data_max_") else Xd.values[:, -1].max()
            preds_d = np.array(preds_d_scaled).reshape(-1,1) * max_target_d
            demand_forecast = [float(x) for x in preds_d.flatten()]
        except Exception as e:
            print("Demand model predict error:", e)
            demand_forecast = fallback_predict_series(demand_df[demand_target], steps=steps)
    else:
        demand_forecast = fallback_predict_series(demand_df[demand_target], steps=steps)

    # Build results table
    hours = list(range(1, steps+1))
    fossil_needed = [max(0, d - w) for d,w in zip(demand_forecast, wind_forecast)]

    results = []
    for h, w, d, f in zip(hours, wind_forecast, demand_forecast, fossil_needed):
        results.append({"hour": h, "wind_mw": round(w,2), "demand_mw": round(d,2), "fossil_needed_mw": round(f,2)})

    # Save results temporarily as JSON for analysis page
    out_json_path = os.path.join(UPLOAD_FOLDER, "latest_results.json")
    with open(out_json_path, "w") as fh:
        json.dump({"results": results}, fh)

    return render_template("results.html", results=results)

@app.route("/analysis")
def analysis():
    out_json_path = os.path.join(UPLOAD_FOLDER, "latest_results.json")
    if not os.path.exists(out_json_path):
        flash("No results available. Please upload CSVs and run prediction first.", "warning")
        return redirect(url_for("index"))
    with open(out_json_path) as fh:
        data = json.load(fh)
    results = data["results"]

    # Build time series for plotting f1 (historical fossil) and f2 (after forecast)
    # Load historical dataset if present at models/historical_energy_mix_2024.csv, else synthesize
    hist_path = os.path.join(MODELS_FOLDER, "historical_energy_mix_2024.csv")
    if os.path.exists(hist_path):
        hist_df = pd.read_csv(hist_path)
        # try to use last 24 fossil values
        if "Fossil_MW" in hist_df.columns:
            f1 = list(hist_df["Fossil_MW"].iloc[-24:].round(2))
        else:
            f1 = [float(x["fossil_needed_mw"]) for x in results]  # fallback
    else:
        # synthesize simple baseline from results mean
        avg = np.mean([r["fossil_needed_mw"] for r in results])
        f1 = [round(avg * (1 + 0.05*np.sin(i/3)),2) for i in range(24)]

    f2 = [r["fossil_needed_mw"] for r in results]

    # energy saved per hour (MWh) = f1 - f2 (in MW for 1 hour = MWh)
    energy_saved = [max(0, round(a - b,2)) for a,b in zip(f1,f2)]
    total_energy_saved_mwh = round(sum(energy_saved),2)
    co2_saved_kg = round(total_energy_saved_mwh * 1000 * 0.95,2)
    recs = round(total_energy_saved_mwh,2)

    return render_template("analysis.html", f1=f1, f2=f2, energy_saved=energy_saved,
                           total_energy_saved_mwh=total_energy_saved_mwh, co2_saved_kg=co2_saved_kg, recs=recs)

@app.route("/download_results")
def download_results():
    path = os.path.join(UPLOAD_FOLDER, "latest_results.json")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="latest_results.json")
    else:
        flash("No results to download.", "warning")
        return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
