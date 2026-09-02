import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "model_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "baseline_results.csv"


# --------------------------------------------------
# 2. Load model features
# --------------------------------------------------

print("Loading model features...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")
print()


# --------------------------------------------------
# 3. Sort data chronologically
# --------------------------------------------------

df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# 4. Time-based train/test split
# --------------------------------------------------

# Use the last 20% of dates as the test period.
unique_dates = df["date"].sort_values().unique()

split_index = int(len(unique_dates) * 0.80)

train_end_date = unique_dates[split_index - 1]
test_start_date = unique_dates[split_index]

train = df[df["date"] <= train_end_date].copy()
test = df[df["date"] >= test_start_date].copy()

print("Time-based split:")
print(f"Training period: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Testing period:  {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Training rows: {len(train):,}")
print(f"Testing rows:  {len(test):,}")
print()


# --------------------------------------------------
# 5. Baseline prediction
# --------------------------------------------------

# Use the previous observed demand as the baseline prediction.
# lag_1 is calculated only from previous observations,
# so it does not use the current target value.

test = test.dropna(subset=["lag_1"]).copy()

actual = test["demand"]
prediction = test["lag_1"]


# --------------------------------------------------
# 6. Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(actual, prediction)

rmse = np.sqrt(
    mean_squared_error(actual, prediction)
)


# --------------------------------------------------
# 7. Calculate MAPE safely
# --------------------------------------------------

non_zero = actual != 0

mape = (
    np.mean(
        np.abs(
            (actual[non_zero] - prediction[non_zero])
            / actual[non_zero]
        )
    )
    * 100
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("Baseline Model Results")
print("----------------------")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print()


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

results = pd.DataFrame({
    "model": ["Baseline - Previous Demand"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE_percent": [mape]
})

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Baseline evaluation completed.")
print(f"Results saved to: {OUTPUT_FILE}")