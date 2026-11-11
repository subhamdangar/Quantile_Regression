
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "wage1.csv"
TABLE_DIR = BASE_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("OLS ESTIMATION: WOOLDRIDGE WAGE1")
print("=" * 70)

# ---------------------------------------------------------
# 3. MODEL SPECIFICATION
# ---------------------------------------------------------

formula = (
    "lwage ~ educ + exper + expersq + tenure + tenursq + female"
)

# ---------------------------------------------------------
# 4. ESTIMATE OLS
# ---------------------------------------------------------

# Conventional OLS estimation
ols_classic = smf.ols(formula=formula, data=df).fit()

# HC3 heteroscedasticity-robust inference
ols_robust = smf.ols(formula=formula, data=df).fit(cov_type="HC3")

print("\n1. MODEL INFORMATION")
print("-" * 70)
print(f"Observations: {int(ols_robust.nobs)}")
print(f"R-squared: {ols_robust.rsquared:.4f}")
print(f"Adjusted R-squared: {ols_robust.rsquared_adj:.4f}")
print(f"F-statistic: {ols_robust.fvalue:.4f}")
print(f"F-test p-value: {ols_robust.f_pvalue:.6f}")

# ---------------------------------------------------------
# 5. COEFFICIENT TABLE
# ---------------------------------------------------------

conf_int = ols_robust.conf_int()

results = pd.DataFrame({
    "Coefficient": ols_robust.params,
    "Robust_Std_Error": ols_robust.bse,
    "t_statistic": ols_robust.tvalues,
    "p_value": ols_robust.pvalues,
    "CI_Lower": conf_int.iloc[:, 0],
    "CI_Upper": conf_int.iloc[:, 1]
})

print("\n2. OLS COEFFICIENT ESTIMATES")
print("-" * 70)
print(results.round(5).to_string())

results.to_csv(
    TABLE_DIR / "ols_results.csv",
    index_label="Variable"
)

# ---------------------------------------------------------
# 6. COMPARE CONVENTIONAL AND ROBUST STANDARD ERRORS
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Coefficient": ols_classic.params,
    "Conventional_SE": ols_classic.bse,
    "HC3_Robust_SE": ols_robust.bse,
    "Conventional_p_value": ols_classic.pvalues,
    "Robust_p_value": ols_robust.pvalues
})

comparison.to_csv(
    TABLE_DIR / "ols_standard_error_comparison.csv",
    index_label="Variable"
)

print("\n3. STANDARD ERROR COMPARISON")
print("-" * 70)
print(comparison.round(5).to_string())

# ---------------------------------------------------------
# 7. SAVE FULL MODEL SUMMARY
# ---------------------------------------------------------

summary_path = TABLE_DIR / "ols_full_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write(ols_robust.summary().as_text())

print("\n4. OUTPUT FILES")
print("-" * 70)
print("ols_results.csv")
print("ols_standard_error_comparison.csv")
print("ols_full_summary.txt")

print("\nOLS ESTIMATION COMPLETED SUCCESSFULLY.")
print("=" * 70)
