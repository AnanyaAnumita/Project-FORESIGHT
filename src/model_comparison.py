import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------
# Load corrected ML results
# ---------------------------------------------------------

ml_file = OUTPUT_DIR / "final_ml_results.csv"

ml_results = pd.read_csv(ml_file)


# ---------------------------------------------------------
# Load classical model results
# ---------------------------------------------------------

classical_files = [
    "arima_results.csv",
    "sarima_results.csv",
    "prophet_results.csv"
]

classical_results = []

for file_name in classical_files:

    file_path = OUTPUT_DIR / file_name

    if file_path.exists():

        result = pd.read_csv(file_path)

        classical_results.append(result)

    else:

        print(f"Warning: {file_name} not found.")


if not classical_results:
    print("No classical model result files found.")
    raise SystemExit


classical_results = pd.concat(
    classical_results,
    ignore_index=True
)


# ---------------------------------------------------------
# Make classical results use the same test period
# ---------------------------------------------------------

classical_results["test_start"] = "2025-03-20"
classical_results["test_end"] = "2025-12-31"

classical_results["data_scope"] = (
    "All stores and all SKUs aggregated daily"
)


# ---------------------------------------------------------
# Keep common columns
# ---------------------------------------------------------

columns = [
    "model",
    "MAE",
    "RMSE",
    "MAPE_percent",
    "data_scope",
    "test_start",
    "test_end"
]

ml_results = ml_results[columns]
classical_results = classical_results[columns]


# ---------------------------------------------------------
# Combine all models
# ---------------------------------------------------------

comparison = pd.concat(
    [
        ml_results,
        classical_results
    ],
    ignore_index=True
)


# ---------------------------------------------------------
# Rank models by MAE
# ---------------------------------------------------------

comparison = comparison.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

comparison.insert(
    0,
    "Rank",
    range(1, len(comparison) + 1)
)


# ---------------------------------------------------------
# Display final comparison
# ---------------------------------------------------------

print("\nFINAL MODEL COMPARISON")
print("======================")

print(
    comparison.to_string(index=False)
)


# ---------------------------------------------------------
# Save final comparison
# ---------------------------------------------------------

output_file = OUTPUT_DIR / "final_model_comparison.csv"

comparison.to_csv(
    output_file,
    index=False
)

print("\nFinal comparison saved to:")
print(output_file)


# ---------------------------------------------------------
# Best model
# ---------------------------------------------------------

best_model = comparison.iloc[0]

print("\nBEST FORECASTING MODEL")
print("======================")

print(f"Model: {best_model['model']}")
print(f"MAE:  {best_model['MAE']:.4f}")
print(f"RMSE: {best_model['RMSE']:.4f}")
print(f"MAPE: {best_model['MAPE_percent']:.2f}%")