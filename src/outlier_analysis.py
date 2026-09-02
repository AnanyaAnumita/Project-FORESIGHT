import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "daily_demand_features.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "outlier_summary.csv"


# --------------------------------------------------
# 2. Load the cleaned demand data
# --------------------------------------------------

print("Loading data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print()


# --------------------------------------------------
# 3. Function to calculate outliers using IQR
# --------------------------------------------------

def outlier_summary(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    return {
        "column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "outlier_count": len(outliers),
        "outlier_percentage": (len(outliers) / len(data)) * 100
    }


# --------------------------------------------------
# 4. Check demand for outliers
# --------------------------------------------------

print("Checking demand for outliers...")

result = outlier_summary(df, "demand")

print(f"Q1: {result['Q1']:.2f}")
print(f"Q3: {result['Q3']:.2f}")
print(f"IQR: {result['IQR']:.2f}")
print(f"Lower limit: {result['lower_limit']:.2f}")
print(f"Upper limit: {result['upper_limit']:.2f}")
print(f"Outliers: {result['outlier_count']:,}")
print(f"Outlier percentage: {result['outlier_percentage']:.2f}%")
print()


# --------------------------------------------------
# 5. Check important numerical columns
# --------------------------------------------------

columns_to_check = [
    "demand",
    "unit_price",
    "cost_price"
]

results = []

for column in columns_to_check:

    if column in df.columns:
        results.append(outlier_summary(df, column))


# --------------------------------------------------
# 6. Save summary
# --------------------------------------------------

summary = pd.DataFrame(results)

summary.to_csv(OUTPUT_FILE, index=False)

print("Outlier analysis completed.")
print(f"Summary saved to: {OUTPUT_FILE}")
print()

print(summary.to_string(index=False))