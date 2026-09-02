import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "model_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "random_forest_results.csv"


print("Loading model features...")

columns = [
    "date",
    "store_id",
    "sku_id",
    "demand",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "unit_price",
    "cost_price",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "promo_active",
    "promo_discount_pct",
    "promo_count",
    "margin_pct"
]

df = pd.read_csv(INPUT_FILE, usecols=columns)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")


# Convert IDs to numeric values
df["store_code"] = df["store_id"].str.replace("ST", "").astype(int)
df["sku_code"] = df["sku_id"].str.replace("SKU", "").astype(int)


# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)


# Time-based train/test split
unique_dates = df["date"].sort_values().unique()

split_index = int(len(unique_dates) * 0.80)

train_end_date = unique_dates[split_index - 1]
test_start_date = unique_dates[split_index]

train = df[df["date"] <= train_end_date].copy()
test = df[df["date"] >= test_start_date].copy()


print("\nTime-based split:")
print(
    f"Training period: {train['date'].min().date()} "
    f"to {train['date'].max().date()}"
)
print(
    f"Testing period:  {test['date'].min().date()} "
    f"to {test['date'].max().date()}"
)

print(f"Training rows: {len(train):,}")
print(f"Testing rows:  {len(test):,}")


# Model features
feature_columns = [
    "store_code",
    "sku_code",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "unit_price",
    "cost_price",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "promo_active",
    "promo_discount_pct",
    "promo_count",
    "margin_pct"
]


train = train.dropna(
    subset=feature_columns + ["demand"]
)

test = test.dropna(
    subset=feature_columns + ["demand"]
)


# Use a representative training sample
sample_size = min(300_000, len(train))

train_sample = train.sample(
    n=sample_size,
    random_state=42
)


X_train = train_sample[feature_columns]
y_train = train_sample["demand"]

X_test = test[feature_columns]
y_test = test["demand"]


print(
    f"\nRows used for Random Forest training: "
    f"{len(X_train):,}"
)


# Random Forest model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=4
)


print("\nTraining Random Forest model...")

model.fit(
    X_train,
    y_train
)

print("Random Forest training completed.")


# Predictions
print("\nGenerating predictions...")

predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)


# Evaluation
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

non_zero = y_test != 0

mape = np.mean(
    np.abs(
        (y_test[non_zero] - predictions[non_zero])
        / y_test[non_zero]
    )
) * 100


print("\nRandom Forest Model Results")
print("---------------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")


# Save results
results = pd.DataFrame({
    "model": ["Random Forest"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE_percent": [mape]
})


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nRandom Forest evaluation completed.")
print(f"Results saved to: {OUTPUT_FILE}")