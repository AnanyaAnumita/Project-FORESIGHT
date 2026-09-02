import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_FILE = DATA_DIR / "daily_demand_features.csv"


# ---------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------

def run_eda():

    print("Loading data for EDA...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 1. Basic dataset information
    # -----------------------------------------------------

    print("\n========== DATASET INFORMATION ==========")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDate range:")
    print(df["date"].min())
    print(df["date"].max())

    print("\nNumber of stores:")
    print(df["store_id"].nunique())

    print("\nNumber of SKUs:")
    print(df["sku_id"].nunique())

    print("\nNumber of categories:")
    print(df["category"].nunique())

    print("\nMissing values:")
    print(df.isna().sum())

    # -----------------------------------------------------
    # 2. Overall demand
    # -----------------------------------------------------

    total_demand = df["demand"].sum()
    average_daily_demand = df["demand"].mean()

    print("\n========== DEMAND SUMMARY ==========")

    print(f"Total demand: {total_demand:,.0f}")
    print(f"Average demand per record: {average_daily_demand:.2f}")

    # -----------------------------------------------------
    # 3. Demand by category
    # -----------------------------------------------------

    category_demand = (
        df.groupby("category")["demand"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== DEMAND BY CATEGORY ==========")
    print(category_demand)

    plt.figure(figsize=(10, 6))

    category_demand.plot(
        kind="bar"
    )

    plt.title("Total Demand by Category")
    plt.xlabel("Category")
    plt.ylabel("Demand")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "demand_by_category.png"
    )

    plt.close()

    # -----------------------------------------------------
    # 4. Demand by store
    # -----------------------------------------------------

    store_demand = (
        df.groupby("store_id")["demand"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== TOP 10 STORES BY DEMAND ==========")
    print(store_demand.head(10))

    plt.figure(figsize=(10, 6))

    store_demand.head(10).plot(
        kind="bar"
    )

    plt.title("Top 10 Stores by Demand")
    plt.xlabel("Store")
    plt.ylabel("Demand")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_10_stores_demand.png"
    )

    plt.close()

    # -----------------------------------------------------
    # 5. Top 10 SKUs
    # -----------------------------------------------------

    sku_demand = (
        df.groupby(
            ["sku_id", "sku_name"]
        )["demand"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== TOP 10 PRODUCTS ==========")
    print(sku_demand.head(10))

    # -----------------------------------------------------
    # 6. Monthly demand trend
    # -----------------------------------------------------

    monthly_demand = (
        df.set_index("date")
        .resample("ME")["demand"]
        .sum()
    )

    print("\n========== MONTHLY DEMAND ==========")
    print(monthly_demand)

    plt.figure(figsize=(12, 6))

    monthly_demand.plot()

    plt.title("Monthly Demand Trend")
    plt.xlabel("Month")
    plt.ylabel("Demand")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_demand_trend.png"
    )

    plt.close()

    # -----------------------------------------------------
    # 7. Weekly demand pattern
    # -----------------------------------------------------

    weekday_demand = (
        df.groupby("day_of_week")["demand"]
        .mean()
    )

    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_demand.index = [
        weekday_names[i]
        for i in weekday_demand.index
    ]

    print("\n========== DEMAND BY DAY OF WEEK ==========")
    print(weekday_demand)

    plt.figure(figsize=(10, 6))

    weekday_demand.plot(
        kind="bar"
    )

    plt.title("Average Demand by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Demand")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "demand_by_weekday.png"
    )

    plt.close()

    # -----------------------------------------------------
    # 8. Weekend vs weekday
    # -----------------------------------------------------

    weekend_demand = (
        df.groupby("is_weekend")["demand"]
        .mean()
    )

    print("\n========== WEEKDAY VS WEEKEND ==========")

    print(
        "Weekday average:",
        weekend_demand.get(0, 0)
    )

    print(
        "Weekend average:",
        weekend_demand.get(1, 0)
    )

    # -----------------------------------------------------
    # 9. Save summary
    # -----------------------------------------------------

    summary = pd.DataFrame({
        "metric": [
            "total_demand",
            "average_demand_per_record",
            "number_of_stores",
            "number_of_skus",
            "number_of_categories",
            "start_date",
            "end_date"
        ],
        "value": [
            total_demand,
            average_daily_demand,
            df["store_id"].nunique(),
            df["sku_id"].nunique(),
            df["category"].nunique(),
            df["date"].min(),
            df["date"].max()
        ]
    })

    summary.to_csv(
        OUTPUT_DIR / "eda_summary.csv",
        index=False
    )

    print("\n========== EDA COMPLETE ==========")

    print(
        f"Charts and summary saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    run_eda()