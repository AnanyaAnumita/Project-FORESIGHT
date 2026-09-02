import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = PROJECT_ROOT / "data raw" / "retail_clean_dataset"

DEMAND_FILE = DATA_DIR / "daily_demand.csv"
SKU_FILE = RAW_DATA / "sku_master.csv"
OUTPUT_FILE = DATA_DIR / "daily_demand_features.csv"


def build_features():
    print("Loading daily demand data...")

    demand = pd.read_csv(
        DEMAND_FILE,
        parse_dates=["date"]
    )

    print("Loading SKU information...")

    sku = pd.read_csv(SKU_FILE)

    # Keep only the product information we need
    sku = sku[
        [
            "sku_id",
            "sku_name",
            "category",
            "subcategory",
            "brand",
            "unit_price",
            "cost_price"
        ]
    ]

    print("Combining demand with product information...")

    features = demand.merge(
        sku,
        on="sku_id",
        how="left"
    )

    # Add useful time-based features
    features["year"] = features["date"].dt.year
    features["month"] = features["date"].dt.month
    features["day_of_week"] = features["date"].dt.dayofweek
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype(int)

    # Check whether any SKU information failed to match
    missing_sku = features["sku_name"].isna().sum()

    print(f"Rows: {len(features):,}")
    print(f"Missing SKU matches: {missing_sku:,}")

    # Save
    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature dataset created successfully!")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_features()