import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "model_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "final_ml_results.csv"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading model features...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "date",
        "demand",
        "unit_price",
        "cost_price",
        "promo_active",
        "promo_discount_pct",
        "promo_count",
        "margin_pct"
    ]
)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")


# ---------------------------------------------------------
# Aggregate to daily demand
# ---------------------------------------------------------

daily = (
    df.groupby("date")
    .agg(
        demand=("demand", "sum"),
        avg_unit_price=("unit_price", "mean"),
        avg_cost_price=("cost_price", "mean"),
        promo_active=("promo_active", "max"),
        promo_discount_pct=("promo_discount_pct", "mean"),
        promo_count=("promo_count", "sum"),
        avg_margin_pct=("margin_pct", "mean")
    )
    .reset_index()
    .sort_values("date")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Create calendar features
# ---------------------------------------------------------

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)


# ---------------------------------------------------------
# Create lag and rolling features
# ---------------------------------------------------------

daily["lag_1"] = daily["demand"].shift(1)
daily["lag_7"] = daily["demand"].shift(7)
daily["lag_14"] = daily["demand"].shift(14)

daily["rolling_mean_7"] = (
    daily["demand"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily["rolling_mean_28"] = (
    daily["demand"]
    .shift(1)
    .rolling(28)
    .mean()
)


# ---------------------------------------------------------
# IMPORTANT:
# Use the EXACT same train/test dates as ARIMA/SARIMA/Prophet
# ---------------------------------------------------------

TRAIN_START = "2022-01-29"
TRAIN_END = "2025-03-19"

TEST_START = "2025-03-20"
TEST_END = "2025-12-31"


train = daily[
    (daily["date"] >= TRAIN_START) &
    (daily["date"] <= TRAIN_END)
].copy()

test = daily[
    (daily["date"] >= TEST_START) &
    (daily["date"] <= TEST_END)
].copy()


# Remove rows with missing ML features
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)


print("\nDaily modeling dataset:")
print(f"Total observations: {len(daily):,}")

print("\nExact common time-based split:")
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
print(f"Testing observations: {len(test):,}")


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

features = [
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "avg_unit_price",
    "avg_cost_price",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "promo_active",
    "promo_discount_pct",
    "promo_count",
    "avg_margin_pct"
]

X_train = train[features]
y_train = train["demand"]

X_test = test[features]
y_test = test["demand"]


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

def evaluate_model(name, model):

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    predictions = np.maximum(
        predictions,
        0
    )

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

    mape = (
        np.mean(
            np.abs(
                (y_test[non_zero] - predictions[non_zero])
                / y_test[non_zero]
            )
        )
        * 100
    )

    print(f"{name} Results")
    print("----------------")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "data_scope": "All stores and all SKUs aggregated daily",
        "test_start": TEST_START,
        "test_end": TEST_END
    }


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

results = []


# Random Forest
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=4
)

results.append(
    evaluate_model(
        "Random Forest",
        rf
    )
)


# XGBoost
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4,
    objective="reg:squarederror"
)

results.append(
    evaluate_model(
        "XGBoost",
        xgb
    )
)


# LightGBM
lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

results.append(
    evaluate_model(
        "LightGBM",
        lgbm
    )
)


# HistGradientBoosting
hist = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=63,
    l2_regularization=0.1,
    random_state=42
)

results.append(
    evaluate_model(
        "HistGradientBoosting",
        hist
    )
)


# ---------------------------------------------------------
# Final comparison
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)


print("\nFINAL ML MODEL COMPARISON")
print("=========================")

print(
    results_df.to_string(index=False)
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)