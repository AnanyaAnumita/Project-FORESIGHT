import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "forecast_data.csv"
PROMO_FILE = (
    BASE_DIR
    / "data raw"
    / "retail_clean_dataset"
    / "promotions.csv"
)

OUTPUT_FILE = BASE_DIR / "data" / "model_features.csv"


# --------------------------------------------------
# 2. Load forecasting data
# --------------------------------------------------

print("Loading forecasting data...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")
print()


# --------------------------------------------------
# 3. Load promotion data
# --------------------------------------------------

print("Loading promotion data...")

promotions = pd.read_csv(PROMO_FILE)

promotions["start_date"] = pd.to_datetime(
    promotions["start_date"]
)

promotions["end_date"] = pd.to_datetime(
    promotions["end_date"]
)

print(f"Promotions loaded: {len(promotions):,}")
print()


# --------------------------------------------------
# 4. Create daily promotion features
# --------------------------------------------------

print("Creating daily promotion features...")

date_range = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="D"
)

promo_daily = pd.DataFrame({
    "date": date_range
})

promo_daily["promo_active"] = 0
promo_daily["promo_discount_pct"] = 0.0
promo_daily["promo_count"] = 0


# Add each promotion to its active dates
for _, promo in promotions.iterrows():

    mask = (
        (promo_daily["date"] >= promo["start_date"])
        &
        (promo_daily["date"] <= promo["end_date"])
    )

    promo_daily.loc[mask, "promo_active"] = 1

    promo_daily.loc[
        mask,
        "promo_discount_pct"
    ] = promo_daily.loc[
        mask,
        "promo_discount_pct"
    ].clip(lower=promo["discount_pct"])

    promo_daily.loc[
        mask,
        "promo_count"
    ] += 1


# --------------------------------------------------
# 5. Merge promotion features
# --------------------------------------------------

print("Merging promotion features...")

df = df.merge(
    promo_daily,
    on="date",
    how="left"
)


# --------------------------------------------------
# 6. Create product margin feature
# --------------------------------------------------

print("Creating margin feature...")

df["margin_pct"] = (
    (df["unit_price"] - df["cost_price"])
    / df["unit_price"]
) * 100


# --------------------------------------------------
# 7. Handle possible infinite values
# --------------------------------------------------

df["margin_pct"] = df["margin_pct"].replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# --------------------------------------------------
# 8. Check missing values
# --------------------------------------------------

print("Checking missing values...")

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

print()


# --------------------------------------------------
# 9. Display final feature information
# --------------------------------------------------

print("Final feature columns:")

for column in df.columns:
    print("-", column)

print()


# --------------------------------------------------
# 10. Save model-ready dataset
# --------------------------------------------------

print("Saving model-ready feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Feature engineering completed successfully.")
print(f"Rows saved: {len(df):,}")
print(f"Columns saved: {len(df.columns):,}")
print(f"Output file: {OUTPUT_FILE}")