import pandas as pd
from pathlib import Path

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data raw" / "retail_clean_dataset"
OUTPUT_DATA = PROJECT_ROOT / "data"

SALES_FILE = RAW_DATA / "sales_transactions.csv"
OUTPUT_FILE = OUTPUT_DATA / "daily_demand.csv"


def prepare_daily_demand():
    print("Starting sales data processing...")

    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)

    # Store daily demand in a list of smaller results
    daily_parts = []

    # Read the large sales file in chunks
    for chunk in pd.read_csv(
        SALES_FILE,
        usecols=["date", "store_id", "sku_id", "quantity"],
        chunksize=200_000
    ):
        chunk["date"] = pd.to_datetime(chunk["date"])

        daily = (
            chunk.groupby(
                ["date", "store_id", "sku_id"],
                as_index=False
            )["quantity"]
            .sum()
            .rename(columns={"quantity": "demand"})
        )

        daily_parts.append(daily)

        print(f"Processed {len(daily_parts)} chunks...")

    # Combine all processed chunks
    daily_demand = pd.concat(daily_parts, ignore_index=True)

    # Aggregate again because the same date/store/SKU
    # may have appeared in different chunks
    daily_demand = (
        daily_demand
        .groupby(
            ["date", "store_id", "sku_id"],
            as_index=False
        )["demand"]
        .sum()
    )

    # Sort the final dataset
    daily_demand = daily_demand.sort_values(
        ["store_id", "sku_id", "date"]
    )

    # Save the analysis-ready dataset
    daily_demand.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessing complete!")
    print(f"Rows created: {len(daily_demand):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_daily_demand()