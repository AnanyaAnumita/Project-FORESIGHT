import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "model_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "hist_gradient_boosting_results.csv"


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

df = pd.read_csv(
    INPUT_FILE,
    usecols=columns
)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")


# Convert identifiers to numeric values
df["store_code"] = (
    df["store_id"]
    .str.replace("ST", "")
    .astype(int)
)

df["sku_code"] = (
    df["sku_id"]
    .str.replace("SKU", "")
    .astype(int)
)


# Sort by date
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


# Features
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


# Remove missing feature rows
train = train.dropna(
    subset=feature_columns + ["demand"]
)

test = test.dropna(
    subset=feature_columns + ["demand"]
)


# Representative training sample
sample_size = min(500_000, len(train))

train_sample = train.sample(
    n=sample_size,
    random_state=42
)


X_train = train_sample[feature_columns]
y_train = train_sample["demand"]

X_test = test[feature_columns]
y_test = test["demand"]


print(
    f"\nRows used for HistGradientBoosting training: "
    f"{len(X_train):,}"
)


# HistGradientBoosting model
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=63,
    l2_regularization=0.1,
    random_state=42
)


print("\nTraining HistGradientBoosting model...")

model.fit(
    X_train,
    y_train
)

print("HistGradientBoosting training completed.")


# Predictions
print("\nGenerating predictions...")

predictions = model.predict(X_test)

# Demand cannot be negative
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


print("\nHistGradientBoosting Model Results")
print("----------------------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")


# Save results
results = pd.DataFrame({
    "model": ["HistGradientBoosting"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE_percent": [mape]
})


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nHistGradientBoosting evaluation completed.")
print(f"Results saved to: {OUTPUT_FILE}")