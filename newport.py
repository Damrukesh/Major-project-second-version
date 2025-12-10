import pandas as pd
import numpy as np

# Load your datasets
demand = pd.read_csv("datasets/demand.csv")
wind = pd.read_csv("datasets/Wind.csv")
portfolio = pd.read_csv("datasets/portfolio.csv")
# --- STEP 1: Clean timestamps ---
# Convert to datetime formats
demand["Timestamp"] = pd.to_datetime(demand["Timestamp"], format="%d-%b-%y")
# Wind data has format like "Jan 1, 12:00 am" - need to add year and specify format
wind["Timestamp"] = wind["Timestamp"].apply(lambda x: pd.to_datetime(x + ', 2024', format='%b %d, %I:%M %p, %Y'))
portfolio["Timestamp"] = pd.to_datetime(portfolio["Timestamp"])

# --- STEP 2: Sort all datasets by timestamp ---
demand = demand.sort_values("Timestamp").reset_index(drop=True)
wind = wind.sort_values("Timestamp").reset_index(drop=True)
portfolio = portfolio.sort_values("Timestamp").reset_index(drop=True)

# Ensure equal length (trim to the smallest)
min_len = min(len(demand), len(wind), len(portfolio))
demand = demand.iloc[:min_len]
wind = wind.iloc[:min_len]
portfolio = portfolio.iloc[:min_len]

# --- STEP 3: Replace portfolio demand and wind with real values ---
portfolio["Demand_MW"] = demand["Demand"]
portfolio["Wind_MW"] = wind["windfarm power"]

# --- STEP 4: Create synthetic Solar (0 at night, peak midday) ---
solar = []
for ts in portfolio["Timestamp"]:
    hour = ts.hour
    if hour < 6 or hour > 20:
        solar.append(0)
    else:
        solar.append(250 * np.sin((np.pi / 14) * (hour - 6)))
portfolio["Solar_MW"] = np.maximum(0, solar)

# --- STEP 5: Create Hydro (small 3–6% of demand) ---
portfolio["Hydro_MW"] = np.round(portfolio["Demand_MW"] * np.random.uniform(0.03, 0.06, min_len), 2)

# --- STEP 6: Create Nuclear (constant ~7% of demand average) ---
avg_nuclear = portfolio["Demand_MW"].mean() * 0.07
portfolio["Nuclear_MW"] = np.repeat(round(avg_nuclear, 2), min_len)

# --- STEP 7: Compute REQUIRED Fossil Fuel (what is ACTUALLY needed) ---
portfolio["Fossil_Required_MW"] = (
    portfolio["Demand_MW"] 
    - portfolio["Wind_MW"] 
    - portfolio["Solar_MW"] 
    - portfolio["Hydro_MW"] 
    - portfolio["Nuclear_MW"]
)

portfolio["Fossil_Required_MW"] = portfolio["Fossil_Required_MW"].clip(lower=0)

# --- STEP 8: Compute ACTUAL Fossil Production (baseline overproduction) ---
# Assume grid overproduces fossil by 10–25% due to lack of forecasting
portfolio["Fossil_Actual_MW"] = portfolio["Fossil_Required_MW"] * np.random.uniform(1.10, 1.25, min_len)

# --- STEP 9: Total Non-Renewables ---
portfolio["Total_NonRenewable_MW"] = portfolio["Fossil_Actual_MW"] + portfolio["Nuclear_MW"]

# --- STEP 10: Remove CO2 column if exists ---
if "CO2_kg" in portfolio.columns:
    portfolio = portfolio.drop(columns=["CO2_kg"])

# --- STEP 11: Save final dataset ---
portfolio.to_csv("final_portfolio.csv", index=False)

print("Dataset created → final_portfolio.csv")
