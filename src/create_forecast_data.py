import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "daily_demand_features.csv"
OUTPUT_FILE = DATA_DIR / "forecast_data.csv"


def create_forecast_data():

    print("Loading feature dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    print(f"Rows loaded: {len(df):,}")

    # Sort by store, SKU and date
    print("Sorting data...")

    df = df.sort_values(
        ["store_id", "sku_id", "date"]
    ).reset_index(drop=True)

    print("Creating lag features...")

    # Previous demand for each store-SKU combination
    df["lag_1"] = (
        df.groupby(["store_id", "sku_id"])["demand"]
        .shift(1)
    )

    df["lag_7"] = (
        df.groupby(["store_id", "sku_id"])["demand"]
        .shift(7)
    )

    df["lag_14"] = (
        df.groupby(["store_id", "sku_id"])["demand"]
        .shift(14)
    )

    print("Creating rolling demand features...")

    # Rolling averages using only previous observations
    df["rolling_mean_7"] = (
        df.groupby(["store_id", "sku_id"])["demand"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["rolling_mean_28"] = (
        df.groupby(["store_id", "sku_id"])["demand"]
        .transform(
            lambda x: x.shift(1).rolling(28).mean()
        )
    )

    print("Removing rows without enough history...")

    df = df.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_mean_7",
            "rolling_mean_28"
        ]
    )

    print("Saving forecasting dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nForecasting dataset created successfully!")
    print(f"Rows remaining: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_forecast_data()