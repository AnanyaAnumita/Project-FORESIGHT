import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "daily_demand_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "transformation_summary.csv"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print()


# --------------------------------------------------
# 3. Check original data types
# --------------------------------------------------

print("Original data types:")
print(df.dtypes)
print()


# --------------------------------------------------
# 4. Convert date column
# --------------------------------------------------

print("Converting date column...")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print(f"Date data type: {df['date'].dtype}")
print()


# --------------------------------------------------
# 5. Create time-based features
# --------------------------------------------------

print("Creating time-based features...")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.day_name()
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

print("Time features created:")
print("- year")
print("- month")
print("- day")
print("- day_of_week")
print("- is_weekend")
print()


# --------------------------------------------------
# 6. Check numerical columns
# --------------------------------------------------

numeric_columns = [
    "demand",
    "unit_price",
    "cost_price"
]

print("Checking numerical columns...")

for column in numeric_columns:

    if column in df.columns:

        # Convert to numeric
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        print(
            f"{column}: "
            f"min={df[column].min():.2f}, "
            f"max={df[column].max():.2f}, "
            f"mean={df[column].mean():.2f}"
        )

print()


# --------------------------------------------------
# 7. Check invalid values
# --------------------------------------------------

print("Checking invalid values...")

negative_demand = (df["demand"] < 0).sum()
zero_demand = (df["demand"] == 0).sum()
negative_price = (df["unit_price"] < 0).sum()
negative_cost = (df["cost_price"] < 0).sum()

print(f"Negative demand: {negative_demand:,}")
print(f"Zero demand: {zero_demand:,}")
print(f"Negative unit price: {negative_price:,}")
print(f"Negative cost price: {negative_cost:,}")
print()


# --------------------------------------------------
# 8. Check missing values after transformation
# --------------------------------------------------

print("Checking missing values after transformation...")

missing_values = df.isnull().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) == 0:
    print("No missing values found.")
else:
    print(missing_values)

print()


# --------------------------------------------------
# 9. Create transformation summary
# --------------------------------------------------

summary = pd.DataFrame({
    "check": [
        "Rows processed",
        "Date converted to datetime",
        "Time features created",
        "Negative demand",
        "Zero demand",
        "Negative unit price",
        "Negative cost price",
        "Missing values after transformation"
    ],
    "result": [
        len(df),
        str(df["date"].dtype),
        "year, month, day, day_of_week, is_weekend",
        negative_demand,
        zero_demand,
        negative_price,
        negative_cost,
        int(df.isnull().sum().sum())
    ]
})


# --------------------------------------------------
# 10. Save summary
# --------------------------------------------------

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Data transformation completed.")
print(f"Summary saved to: {OUTPUT_FILE}")