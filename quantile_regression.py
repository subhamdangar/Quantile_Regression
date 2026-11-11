
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from statsmodels.regression.quantile_regression import QuantReg


# ================================================================
# 1. PROJECT CONFIGURATION
# ================================================================

DATA_PATH = os.path.join("data", "wage1.csv")

TABLE_DIR = os.path.join("results", "tables")
FIGURE_DIR = os.path.join("results", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

FORMULA = (
    "lwage ~ educ + exper + expersq "
    "+ tenure + tenursq + female"
)

MAX_ITER = 10000
P_TOL = 1e-6


# ================================================================
# 2. LOAD AND VALIDATE DATA
# ================================================================

print("=" * 75)
print("QUANTILE REGRESSION: WOOLDRIDGE WAGE1")
print("=" * 75)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = [
    "lwage", "educ", "exper", "expersq",
    "tenure", "tenursq", "female"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df[required_columns].copy()

if df.isnull().any().any():
    raise ValueError("Missing observations detected.")

if not np.isfinite(df.to_numpy()).all():
    raise ValueError("Non-finite values detected.")

print("\n1. DATA INFORMATION")
print("-" * 75)
print(f"Observations: {len(df)}")
print(f"Dependent variable: lwage")
print(f"Explanatory variables: {len(required_columns) - 1}")
print(f"Quantiles: {QUANTILES}")


# ================================================================
# 3. ESTIMATE OLS BENCHMARK
# ================================================================

ols_model = smf.ols(
    formula=FORMULA,
    data=df
).fit(cov_type="HC3")

coefficient_names = ols_model.model.exog_names

ols_coefficients = ols_model.params

print("\n2. OLS BENCHMARK")
print("-" * 75)
print(f"R-squared: {ols_model.rsquared:.4f}")
print(f"Adjusted R-squared: {ols_model.rsquared_adj:.4f}")


# ================================================================
# 4. ESTIMATE QUANTILE REGRESSION MODELS
# ================================================================

exog_matrix = pd.DataFrame(
    ols_model.model.exog,
    columns=coefficient_names,
    index=df.index
)

qr_model = QuantReg(
    endog=df["lwage"],
    exog=exog_matrix
)

all_results = []
summary_text = []
fitted_models = {}

print("\n3. QUANTILE REGRESSION ESTIMATION")
print("-" * 75)

for tau in QUANTILES:

    print(f"\nEstimating quantile: {tau:.2f}")

    with warnings.catch_warnings(record=True) as caught_warnings:

        warnings.simplefilter("always")

        result = qr_model.fit(
            q=tau,
            vcov="robust",
            kernel="epa",
            bandwidth="hsheather",
            max_iter=MAX_ITER,
            p_tol=P_TOL
        )

    fitted_models[tau] = result

    if caught_warnings:
        print("Warnings:")
        for warning in caught_warnings:
            print(f"  {warning.message}")

    print(f"Convergence iterations: {result.iterations}")

    confidence_intervals = result.conf_int()

    for variable in coefficient_names:

        all_results.append({
            "Quantile": tau,
            "Variable": variable,
            "Coefficient": result.params[variable],
            "Robust_Std_Error": result.bse[variable],
            "t_statistic": result.tvalues[variable],
            "p_value": result.pvalues[variable],
            "CI_Lower": confidence_intervals.loc[variable, 0],
            "CI_Upper": confidence_intervals.loc[variable, 1],
            "Iterations": result.iterations
        })

    summary_text.append(
        f"\n{'=' * 75}\n"
        f"QUANTILE REGRESSION SUMMARY: TAU = {tau:.2f}\n"
        f"{'=' * 75}\n"
        + result.summary().as_text()
        + "\n"
    )


# ================================================================
# 5. SAVE DETAILED QR RESULTS
# ================================================================

qr_results = pd.DataFrame(all_results)

qr_results.to_csv(
    os.path.join(TABLE_DIR, "quantile_regression_results.csv"),
    index=False
)

with open(
    os.path.join(TABLE_DIR, "quantile_regression_summaries.txt"),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "QUANTILE REGRESSION RESULTS\n"
        "WOOLDRIDGE WAGE1 DATASET\n"
    )

    for summary in summary_text:
        file.write(summary)


# ================================================================
# 6. OLS VS QR COEFFICIENT COMPARISON
# ================================================================

comparison = pd.DataFrame({
    "Variable": coefficient_names,
    "OLS_HC3": [
        ols_coefficients[variable]
        for variable in coefficient_names
    ]
})

for tau in QUANTILES:

    quantile_coefficients = fitted_models[tau].params

    comparison[f"QR_{tau:.2f}"] = [
        quantile_coefficients[variable]
        for variable in coefficient_names
    ]

comparison.to_csv(
    os.path.join(TABLE_DIR, "ols_qr_coefficient_comparison.csv"),
    index=False
)


# ================================================================
# 7. DISPLAY COEFFICIENT RESULTS
# ================================================================

print("\n4. QUANTILE REGRESSION COEFFICIENTS")
print("-" * 75)

coefficient_table = qr_results.pivot(
    index="Variable",
    columns="Quantile",
    values="Coefficient"
)

print(coefficient_table.round(5).to_string())

print("\n5. OLS VS QR COEFFICIENT COMPARISON")
print("-" * 75)

print(comparison.round(5).to_string(index=False))


# ================================================================
# 8. PLOT COEFFICIENT PROFILES
# ================================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(18, 9)
)

axes = axes.flatten()

for index, variable in enumerate(coefficient_names):

    ax = axes[index]

    variable_results = qr_results[
        qr_results["Variable"] == variable
    ].sort_values("Quantile")

    x = variable_results["Quantile"].to_numpy()
    y = variable_results["Coefficient"].to_numpy()

    lower = variable_results["CI_Lower"].to_numpy()
    upper = variable_results["CI_Upper"].to_numpy()

    ax.errorbar(
        x,
        y,
        yerr=[y - lower, upper - y],
        fmt="o-",
        capsize=4,
        linewidth=1.5,
        label="Quantile Regression"
    )

    ax.axhline(
        y=ols_coefficients[variable],
        linestyle="--",
        linewidth=1.5,
        label="OLS"
    )

    ax.set_title(variable)
    ax.set_xlabel("Conditional Quantile (tau)")
    ax.set_ylabel("Coefficient")
    ax.set_xticks(QUANTILES)
    ax.grid(alpha=0.3)

    if index == 0:
        ax.legend()

axes[-1].axis("off")

fig.suptitle(
    "OLS vs Quantile Regression: Coefficient Profiles",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "07_ols_vs_qr_coefficient_profiles.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 9. SAVE MODEL FIT INFORMATION
# ================================================================

fit_information = []

for tau in QUANTILES:

    result = fitted_models[tau]

    fit_information.append({
        "Quantile": tau,
        "Pseudo_R_squared": result.prsquared,
        "Iterations": result.iterations
    })

fit_df = pd.DataFrame(fit_information)

fit_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "quantile_regression_fit_information.csv"
    ),
    index=False
)


# ================================================================
# 10. FINAL OUTPUT
# ================================================================

print("\n6. MODEL FIT INFORMATION")
print("-" * 75)
print(fit_df.round(5).to_string(index=False))

print("\n7. OUTPUT FILES")
print("-" * 75)

for filename in [
    "quantile_regression_results.csv",
    "quantile_regression_summaries.txt",
    "ols_qr_coefficient_comparison.csv",
    "quantile_regression_fit_information.csv"
]:
    print(filename)

print("\nFIGURE")
print("07_ols_vs_qr_coefficient_profiles.png")

print("\nQUANTILE REGRESSION COMPLETED SUCCESSFULLY.")
print("=" * 75)
