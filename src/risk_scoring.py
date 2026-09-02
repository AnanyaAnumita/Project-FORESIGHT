import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data raw"
    / "retail_clean_dataset"
    / "inventory_snapshot.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading inventory data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")

# ---------------------------------------------------------
# 1. Calculate stock gap relative to reorder point
# ---------------------------------------------------------

df["reorder_gap"] = (
    df["stock_on_hand"] - df["reorder_point"]
)

# ---------------------------------------------------------
# 2. Calculate stock gap relative to safety stock
# ---------------------------------------------------------

df["safety_stock_gap"] = (
    df["stock_on_hand"] - df["safety_stock"]
)

# ---------------------------------------------------------
# 3. Calculate stock coverage ratio
# ---------------------------------------------------------

df["stock_to_reorder_ratio"] = np.where(
    df["reorder_point"] > 0,
    df["stock_on_hand"] / df["reorder_point"],
    np.nan
)

# ---------------------------------------------------------
# 4. Create risk category
# ---------------------------------------------------------

def assign_risk(row):

    stock = row["stock_on_hand"]
    reorder = row["reorder_point"]
    safety = row["safety_stock"]

    # Critical stockout risk
    if stock <= 0:
        return "Critical"

    # High risk: below safety stock
    elif stock < safety:
        return "High"

    # Medium risk: below reorder point
    elif stock < reorder:
        return "Medium"

    # Low risk: sufficient stock
    else:
        return "Low"


df["risk_category"] = df.apply(
    assign_risk,
    axis=1
)

# ---------------------------------------------------------
# 5. Convert risk category to numerical score
# ---------------------------------------------------------

risk_score_mapping = {
    "Critical": 100,
    "High": 75,
    "Medium": 50,
    "Low": 10
}

df["risk_score"] = df["risk_category"].map(
    risk_score_mapping
)

# ---------------------------------------------------------
# 6. Convert restock date
# ---------------------------------------------------------

df["last_restock_date"] = pd.to_datetime(
    df["last_restock_date"],
    errors="coerce"
)

# ---------------------------------------------------------
# 7. Save risk-scored inventory
# ---------------------------------------------------------

output_file = OUTPUT_DIR / "inventory_risk_scoring.csv"

df.to_csv(
    output_file,
    index=False
)

# ---------------------------------------------------------
# 8. Print summary
# ---------------------------------------------------------

print("\nRISK SCORING SUMMARY")
print("====================")

print("\nRisk categories:")
print(
    df["risk_category"]
    .value_counts()
    .sort_index()
)

print("\nAverage risk score:")
print(
    round(df["risk_score"].mean(), 2)
)

print("\nRisk percentage:")
print(
    (df["risk_category"]
     .value_counts(normalize=True)
     * 100)
    .round(2)
)

print("\nOutput saved to:")
print(output_file)
