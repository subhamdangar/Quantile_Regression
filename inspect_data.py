
import os
import numpy as np
import pandas as pd

# ==========================================================
# 1. PATHS
# ==========================================================

DATA_FILE = "data/wage1.csv"
RESULTS_DIR = "results/tables"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# 2. LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)

print("=" * 75)
print("DETAILED WOOLDRIDGE WAGE1 DATA INSPECTION")
print("=" * 75)

# ==========================================================
# 3. DATASET STRUCTURE
# ==========================================================

print("\n1. DATASET STRUCTURE")

print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# ==========================================================
# 4. MISSING VALUE ANALYSIS
# ==========================================================

print("\n2. MISSING VALUE ANALYSIS")

missing = df.isnull().sum()

print(missing)

print("\nTotal missing values:", missing.sum())

# ==========================================================
# 5. DESCRIPTIVE STATISTICS
# ==========================================================

print("\n3. DESCRIPTIVE STATISTICS")

summary = df.describe().T

print(summary.round(4))

summary.to_csv(
    os.path.join(RESULTS_DIR, "descriptive_statistics.csv")
)

# ==========================================================
# 6. CHECK LOG AND SQUARED VARIABLES
# ==========================================================

print("\n4. TRANSFORMATION VALIDATION")

log_error = np.abs(
    df["lwage"] - np.log(df["wage"])
).max()

exper_error = np.abs(
    df["expersq"] - df["exper"] ** 2
).max()

tenure_error = np.abs(
    df["tenursq"] - df["tenure"] ** 2
).max()

print("Maximum log wage difference:", log_error)
print("Maximum experience square difference:", exper_error)
print("Maximum tenure square difference:", tenure_error)

print(
    "\nLog wage verified:",
    np.allclose(df["lwage"], np.log(df["wage"]))
)

print(
    "Experience square verified:",
    np.allclose(df["expersq"], df["exper"] ** 2)
)

print(
    "Tenure square verified:",
    np.allclose(df["tenursq"], df["tenure"] ** 2)
)

# ==========================================================
# 7. BINARY VARIABLE INSPECTION
# ==========================================================

print("\n5. BINARY VARIABLE INSPECTION")

binary_variables = [
    "nonwhite", "female", "married", "smsa",
    "northcen", "south", "west",
    "construc", "ndurman", "trcommpu",
    "trade", "services", "profserv",
    "profocc", "clerocc", "servocc"
]

for variable in binary_variables:

    print(f"\n{variable}")

    print("Unique values:", sorted(df[variable].unique()))

    print("Frequency:")
    print(df[variable].value_counts().sort_index())

# ==========================================================
# 8. KEY VARIABLES: DISTRIBUTION
# ==========================================================

print("\n6. IMPORTANT VARIABLE DISTRIBUTIONS")

key_variables = [
    "wage", "lwage", "educ",
    "exper", "tenure", "female"
]

for variable in key_variables:

    print(f"\n{variable}")

    print("Mean:", round(df[variable].mean(), 4))
    print("Median:", round(df[variable].median(), 4))
    print("Standard deviation:", round(df[variable].std(), 4))
    print("Minimum:", round(df[variable].min(), 4))
    print("Maximum:", round(df[variable].max(), 4))

# ==========================================================
# 9. GENDER COMPARISON
# ==========================================================

print("\n7. DESCRIPTIVE GENDER COMPARISON")

gender_summary = df.groupby("female").agg(
    observations=("wage", "count"),
    mean_wage=("wage", "mean"),
    median_wage=("wage", "median"),
    mean_log_wage=("lwage", "mean"),
    mean_education=("educ", "mean"),
    mean_experience=("exper", "mean")
)

gender_summary.index = ["Male (0)", "Female (1)"]

print(gender_summary.round(4))

gender_summary.to_csv(
    os.path.join(RESULTS_DIR, "gender_descriptive_comparison.csv")
)

# ==========================================================
# 10. CORRELATION ANALYSIS
# ==========================================================

print("\n8. CORRELATION MATRIX")

correlation_variables = [
    "lwage", "educ", "exper",
    "expersq", "tenure", "tenursq", "female"
]

correlation_matrix = df[
    correlation_variables
].corr()

print(correlation_matrix.round(4))

correlation_matrix.to_csv(
    os.path.join(RESULTS_DIR, "correlation_matrix.csv")
)

# ==========================================================
# 11. FINAL VALIDATION
# ==========================================================

print("\n9. FINAL VALIDATION")

assert df.shape == (526, 24)
assert missing.sum() == 0
assert (df["wage"] > 0).all()

print("Dataset dimensions: PASS")
print("Missing value check: PASS")
print("Positive wage check: PASS")

print("\n" + "=" * 75)
print("DATA INSPECTION COMPLETED")
print("=" * 75)
