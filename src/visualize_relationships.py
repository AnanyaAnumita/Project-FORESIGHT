import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "daily_demand_features.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print()


# --------------------------------------------------
# 3. Category vs Demand
# --------------------------------------------------

print("Creating category vs demand chart...")

category_demand = (
    df.groupby("category")["demand"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
category_demand.plot(kind="bar")
plt.title("Total Demand by Category")
plt.xlabel("Category")
plt.ylabel("Total Demand")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "relationship_category_demand.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 4. Store vs Demand
# --------------------------------------------------

print("Creating store vs demand chart...")

store_demand = (
    df.groupby("store_id")["demand"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
store_demand.plot(kind="bar")
plt.title("Total Demand by Store")
plt.xlabel("Store")
plt.ylabel("Total Demand")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "relationship_store_demand.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 5. Monthly Demand Trend
# --------------------------------------------------

print("Creating monthly demand trend...")

df["date"] = pd.to_datetime(df["date"])

monthly_demand = (
    df.groupby(df["date"].dt.to_period("M"))["demand"]
    .sum()
)

monthly_demand.index = monthly_demand.index.astype(str)

plt.figure(figsize=(14, 6))
monthly_demand.plot(kind="line")

plt.title("Monthly Demand Trend")
plt.xlabel("Month")
plt.ylabel("Total Demand")
plt.xticks(
    range(0, len(monthly_demand), 3),
    monthly_demand.index[::3],
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "relationship_monthly_demand.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 6. Price vs Demand
# --------------------------------------------------

print("Creating price vs demand relationship...")

# Aggregate demand by SKU so that each point represents a product
sku_analysis = (
    df.groupby("sku_id")
    .agg(
        unit_price=("unit_price", "first"),
        total_demand=("demand", "sum")
    )
    .reset_index()
)

plt.figure(figsize=(10, 6))

plt.scatter(
    sku_analysis["unit_price"],
    sku_analysis["total_demand"],
    alpha=0.5
)

plt.title("Unit Price vs Total Product Demand")
plt.xlabel("Unit Price")
plt.ylabel("Total Demand")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "relationship_price_demand.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 7. Display summary
# --------------------------------------------------

print()
print("Visualization completed successfully.")
print()
print("Charts created:")

print("- relationship_category_demand.png")
print("- relationship_store_demand.png")
print("- relationship_monthly_demand.png")
print("- relationship_price_demand.png")

print()
print(f"Charts saved in: {OUTPUT_DIR}")