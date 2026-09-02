import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "data raw" / "retail_clean_dataset"

DAILY_DEMAND_FILE = DATA_DIR / "daily_demand.csv"
SKU_MASTER_FILE = DATA_DIR / "sku_master.csv"

OUTPUT_DIR = DATA_DIR / "dashboard_summaries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# FIND INPUT FILES
# ---------------------------------------------------------
if not DAILY_DEMAND_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{DAILY_DEMAND_FILE}\n\n"
        "Make sure data/daily_demand.csv exists."
    )

if SKU_MASTER_FILE.exists():
    sku_master_path = SKU_MASTER_FILE
else:
    sku_master_path = RAW_DIR / "sku_master.csv"

if not sku_master_path.exists():
    raise FileNotFoundError(
        f"Could not find sku_master.csv at:\n{sku_master_path}"
    )


# ---------------------------------------------------------
# LOAD SKU MASTER
# ---------------------------------------------------------
print("Loading SKU master...")

sku_master = pd.read_csv(
    sku_master_path,
    usecols=["sku_id", "sku_name", "category"],
    low_memory=False
)

sku_master["sku_id"] = sku_master["sku_id"].astype(str)

sku_category = (
    sku_master
    .drop_duplicates("sku_id")
    .set_index("sku_id")["category"]
    .to_dict()
)


# ---------------------------------------------------------
# STORAGE FOR PARTIAL SUMMARIES
# ---------------------------------------------------------
daily_parts = []
store_parts = []
sku_parts = []
month_parts = []
category_parts = []
weekday_parts = []


# ---------------------------------------------------------
# PROCESS DAILY DEMAND
# ---------------------------------------------------------
print("\nProcessing daily_demand.csv...")
print(f"Input file: {DAILY_DEMAND_FILE}")
print("This step may take a few minutes because the source file is large.\n")

chunk_size = 250_000

reader = pd.read_csv(
    DAILY_DEMAND_FILE,
    usecols=["date", "store_id", "sku_id", "demand"],
    chunksize=chunk_size,
    low_memory=False
)

total_rows = 0
chunk_number = 0

for chunk in reader:

    chunk_number += 1
    total_rows += len(chunk)

    print(
        f"Processing chunk {chunk_number} | "
        f"Rows processed: {total_rows:,}"
    )

    # -----------------------------------------------------
    # CLEAN TYPES
    # -----------------------------------------------------
    chunk["date"] = pd.to_datetime(
        chunk["date"],
        errors="coerce"
    )

    chunk["store_id"] = chunk["store_id"].astype(str)
    chunk["sku_id"] = chunk["sku_id"].astype(str)

    chunk["demand"] = pd.to_numeric(
        chunk["demand"],
        errors="coerce"
    ).fillna(0)

    # Remove invalid dates
    chunk = chunk.dropna(subset=["date"])

    # -----------------------------------------------------
    # TIME FIELDS
    # -----------------------------------------------------
    chunk["year"] = chunk["date"].dt.year
    chunk["month_num"] = chunk["date"].dt.month
    chunk["month"] = chunk["date"].dt.strftime("%b")
    chunk["weekday"] = chunk["date"].dt.day_name()

    # -----------------------------------------------------
    # DAILY SUMMARY
    # -----------------------------------------------------
    daily = (
        chunk.groupby("date", as_index=False)["demand"]
        .sum()
    )

    daily_parts.append(daily)

    # -----------------------------------------------------
    # STORE SUMMARY
    # -----------------------------------------------------
    store = (
        chunk.groupby("store_id", as_index=False)["demand"]
        .sum()
    )

    store_parts.append(store)

    # -----------------------------------------------------
    # SKU SUMMARY
    # -----------------------------------------------------
    sku = (
        chunk.groupby("sku_id", as_index=False)["demand"]
        .sum()
    )

    sku_parts.append(sku)

    # -----------------------------------------------------
    # MONTHLY SUMMARY
    # -----------------------------------------------------
    monthly = (
        chunk.groupby(
            ["year", "month_num", "month"],
            as_index=False
        )["demand"]
        .sum()
    )

    month_parts.append(monthly)

    # -----------------------------------------------------
    # WEEKDAY SUMMARY
    # -----------------------------------------------------
    weekday = (
        chunk.groupby("weekday", as_index=False)["demand"]
        .sum()
    )

    weekday_parts.append(weekday)

    # -----------------------------------------------------
    # CATEGORY SUMMARY
    # -----------------------------------------------------
    chunk["category"] = chunk["sku_id"].map(sku_category)

    category = (
        chunk.groupby("category", dropna=False, as_index=False)["demand"]
        .sum()
    )

    category_parts.append(category)


# ---------------------------------------------------------
# COMBINE PARTIAL RESULTS
# ---------------------------------------------------------
print("\nCombining summaries...")


# DAILY
daily_df = (
    pd.concat(daily_parts, ignore_index=True)
    .groupby("date", as_index=False)["demand"]
    .sum()
    .sort_values("date")
)

# STORE
store_df = (
    pd.concat(store_parts, ignore_index=True)
    .groupby("store_id", as_index=False)["demand"]
    .sum()
    .sort_values("demand", ascending=False)
)

# SKU
sku_df = (
    pd.concat(sku_parts, ignore_index=True)
    .groupby("sku_id", as_index=False)["demand"]
    .sum()
    .sort_values("demand", ascending=False)
)

# MONTH
month_df = (
    pd.concat(month_parts, ignore_index=True)
    .groupby(
        ["year", "month_num", "month"],
        as_index=False
    )["demand"]
    .sum()
    .sort_values(["year", "month_num"])
)

# CATEGORY
category_df = (
    pd.concat(category_parts, ignore_index=True)
    .groupby("category", dropna=False, as_index=False)["demand"]
    .sum()
    .sort_values("demand", ascending=False)
)

# WEEKDAY
weekday_df = (
    pd.concat(weekday_parts, ignore_index=True)
    .groupby("weekday", as_index=False)["demand"]
    .sum()
)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_df["weekday"] = pd.Categorical(
    weekday_df["weekday"],
    categories=weekday_order,
    ordered=True
)

weekday_df = (
    weekday_df
    .sort_values("weekday")
    .reset_index(drop=True)
)

weekday_df["weekday"] = weekday_df["weekday"].astype(str)


# ---------------------------------------------------------
# SAVE SUMMARY FILES
# ---------------------------------------------------------
print("\nSaving dashboard summary files...")


files = {
    "dashboard_daily.csv": daily_df,
    "dashboard_store.csv": store_df,
    "dashboard_sku.csv": sku_df,
    "dashboard_monthly.csv": month_df,
    "dashboard_category.csv": category_df,
    "dashboard_weekday.csv": weekday_df,
}

for filename, dataframe in files.items():

    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path.name} "
        f"({len(dataframe):,} rows)"
    )


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("DASHBOARD DATA PREPARATION COMPLETE")
print("=" * 60)

print(f"\nSource rows processed: {total_rows:,}")

print("\nSummary files created in:")
print(OUTPUT_DIR)

print("\nYou can now run the Streamlit dashboard.")
print("The dashboard will no longer process the 8.5M-row")
print("daily_demand.csv during normal startup.")