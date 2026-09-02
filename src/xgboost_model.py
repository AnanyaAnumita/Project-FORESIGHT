import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "model_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "xgboost_results.csv"


print("Loading model features...")

# Load only the columns required by the model
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


# Convert store and SKU IDs to numeric values
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


# Remove rows where lag/rolling features are unavailable
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

train = train.dropna(subset=feature_columns + ["demand"])
test = test.dropna(subset=feature_columns + ["demand"])


X_train = train[feature_columns]
y_train = train["demand"]

X_test = test[feature_columns]
y_test = test["demand"]


# To keep training practical on a normal computer,
# use a representative sample of the training data.
sample_size = min(500_000, len(X_train))

train_sample = train.sample(
    n=sample_size,
    random_state=42
)

X_train_sample = train_sample[feature_columns]
y_train_sample = train_sample["demand"]


print(f"\nRows used for XGBoost training: {len(X_train_sample):,}")


# XGBoost model
model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4
)


print("\nTraining XGBoost model...")

model.fit(
    X_train_sample,
    y_train_sample,
    verbose=False
)


print("XGBoost training completed.")


# Predictions
print("\nGenerating predictions...")

predictions = model.predict(X_test)

# Demand cannot be negative
predictions = np.maximum(predictions, 0)


# Evaluation
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

non_zero = y_test != 0

mape = np.mean(
    np.abs(
        (y_test[non_zero] - predictions[non_zero])
        / y_test[non_zero]
    )
) * 100


print("\nXGBoost Model Results")
print("---------------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")


# Save model results
results = pd.DataFrame({
    "model": ["XGBoost"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE_percent": [mape]
})

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nXGBoost evaluation completed.")
print(f"Results saved to: {OUTPUT_FILE}")
