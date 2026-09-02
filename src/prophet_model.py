import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "model_features.csv"
RESULTS_FILE = BASE_DIR / "outputs" / "prophet_results.csv"
FORECAST_FILE = BASE_DIR / "outputs" / "prophet_forecast_predictions.csv"

print("Loading model features...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=["date", "demand"]
)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")

# Aggregate all stores and SKUs into one daily demand series
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

# Same 80/20 time-based split used for model comparison
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

# Convert training data to Prophet format
prophet_train = train.rename(
    columns={
        "date": "ds",
        "demand": "y"
    }
)

print("\nTraining Prophet model...")

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive",
    interval_width=0.95
)

model.fit(prophet_train)

print("Prophet training completed.")

print("\nGenerating forecasts...")

future = test[["date"]].rename(
    columns={"date": "ds"}
)

forecast_df = model.predict(future)

# Keep actual test demand together with predictions
predictions = pd.DataFrame({
    "date": test["date"].values,
    "actual": test["demand"].values,
    "forecast": np.maximum(forecast_df["yhat"].values, 0),
    "lower_bound": np.maximum(
        forecast_df["yhat_lower"].values,
        0
    ),
    "upper_bound": np.maximum(
        forecast_df["yhat_upper"].values,
        0
    )
})

# Evaluation
actual = predictions["actual"].values
forecast = predictions["forecast"].values

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

print("\nProphet Model Results")
print("---------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# Save model metrics
results = pd.DataFrame({
    "model": ["Prophet"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE_percent": [mape],
    "data_scope": [
        "All stores and all SKUs aggregated daily"
    ]
})

results.to_csv(
    RESULTS_FILE,
    index=False
)

# Save individual forecast predictions
predictions.to_csv(
    FORECAST_FILE,
    index=False
)

print("\nProphet evaluation completed.")

print(f"Results saved to: {RESULTS_FILE}")
print(f"Forecast predictions saved to: {FORECAST_FILE}")

print("\nForecast prediction sample:")
print(predictions.head())