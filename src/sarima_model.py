import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "model_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "sarima_results.csv"

print("Loading model features...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["date", "demand"]
)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")

# Aggregate demand by date.
# This creates the same daily demand series used by ARIMA and Prophet.
daily_demand = (
    df.groupby("date", as_index=False)["demand"]
    .sum()
    .sort_values("date")
    .reset_index(drop=True)
)

print("\nDaily demand series created:")
print(f"Observations: {len(daily_demand):,}")
print(
    f"Date range: "
    f"{daily_demand['date'].min().date()} to "
    f"{daily_demand['date'].max().date()}"
)

# Same 80/20 time-based split
split_index = int(len(daily_demand) * 0.80)

train = daily_demand.iloc[:split_index].copy()
test = daily_demand.iloc[split_index:].copy()

print("\nTime-based split:")
print(
    f"Training period: "
    f"{train['date'].min().date()} to "
    f"{train['date'].max().date()}"
)
print(
    f"Testing period:  "
    f"{test['date'].min().date()} to "
    f"{test['date'].max().date()}"
)
print(f"Training observations: {len(train):,}")
print(f"Testing observations:  {len(test):,}")

print("\nTraining SARIMA model...")

# SARIMA with weekly seasonality
model = SARIMAX(
    train["demand"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fitted_model = model.fit(
    disp=False
)

print("SARIMA training completed.")

print("\nGenerating forecasts...")

forecast = fitted_model.forecast(
    steps=len(test)
)

forecast = np.maximum(
    np.asarray(forecast),
    0
)

actual = test["demand"].values

# Evaluation
mae = mean_absolute_error(
    actual,
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        forecast
    )
)

non_zero = actual != 0

mape = (
    np.mean(
        np.abs(
            (actual[non_zero] - forecast[non_zero])
            / actual[non_zero]
        )
    )
    * 100
)

print("\nSARIMA Model Results")
print("--------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

results = pd.DataFrame({
    "model": ["SARIMA"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE_percent": [mape],
    "data_scope": ["All stores and all SKUs aggregated daily"]
})

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSARIMA evaluation completed.")
print(f"Results saved to: {OUTPUT_FILE}")