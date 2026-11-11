
import os
import pandas as pd
import wooldridge as woo

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "wage1.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load Wooldridge wage1 dataset
# --------------------------------------------------

print("=" * 60)
print("DOWNLOADING WOOLDRIDGE WAGE1 DATASET")
print("=" * 60)

wage1 = woo.data("wage1")

# --------------------------------------------------
# 3. Basic validation
# --------------------------------------------------

print("\nDataset successfully loaded.")

print("\nDataset dimensions:")
print(f"Observations: {wage1.shape[0]}")
print(f"Variables: {wage1.shape[1]}")

print("\nVariable names:")
print(wage1.columns.tolist())

print("\nFirst five observations:")
print(wage1.head())

# --------------------------------------------------
# 4. Save dataset
# --------------------------------------------------

wage1.to_csv(OUTPUT_FILE, index=False)

print("\nDataset saved successfully.")
print(f"Location: {os.path.abspath(OUTPUT_FILE)}")

# --------------------------------------------------
# 5. Initial data quality checks
# --------------------------------------------------

print("\nMissing values:")
print(wage1.isnull().sum())

print("\nData types:")
print(wage1.dtypes)

print("\nSummary statistics:")
print(wage1.describe().T)

print("\n" + "=" * 60)
print("DATA ACQUISITION COMPLETED")
print("=" * 60)
