
"""
Gender-based heterogeneous effects using OLS and Quantile Regression.

Dataset: Wooldridge wage1
Dependent variable: lwage
Main variable of interest: educ
Heterogeneity variable: female

The model allows the education-wage association to differ by gender
and across conditional wage quantiles.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import norm


# ---------------------------------------------------------
# 1. Project paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "wage1.csv"

RESULTS_DIR = BASE_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
CONFIDENCE_LEVEL = 0.95

FORMULA = (
    "lwage ~ educ * female + exper + expersq + tenure + tenursq"
)


# ---------------------------------------------------------
# 2. Load and validate the data
# ---------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = [
    "lwage",
    "educ",
    "female",
    "exper",
    "expersq",
    "tenure",
    "tenursq",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isna().any().any():
    raise ValueError("Missing values found in model variables.")

if not np.isfinite(df[required_columns].to_numpy()).all():
    raise ValueError("Non-finite values found in model variables.")

if not set(df["female"].unique()).issubset({0, 1}):
    raise ValueError("female must be coded as 0 for men and 1 for women.")

print("=" * 65)
print("GENDER-BASED HETEROGENEOUS EFFECTS")
print("=" * 65)
print(f"Observations: {len(df)}")
print(f"Formula: {FORMULA}")


# ---------------------------------------------------------
# 3. Estimate OLS interaction model with HC3 standard errors
# ---------------------------------------------------------

ols_result = smf.ols(FORMULA, data=df).fit(cov_type="HC3")

print("\nOLS interaction model estimated successfully.")


# ---------------------------------------------------------
# 4. Helper function for coefficient inference
# ---------------------------------------------------------

def get_covariance_matrix(result):
    """Return a labelled covariance matrix."""

    covariance = result.cov_params()

    if isinstance(covariance, pd.DataFrame):
        return covariance

    names = list(result.params.index)

    return pd.DataFrame(
        covariance,
        index=names,
        columns=names
    )


def calculate_slope(
    result,
    covariance,
    terms,
    weights,
    quantile,
    gender,
    model_name
):
    """
    Calculate a linear combination of regression coefficients
    and its delta-method standard error.
    """

    params = result.params

    estimate = sum(
        weights[i] * params[terms[i]]
        for i in range(len(terms))
    )

    variance = 0.0

    for i in range(len(terms)):
        for j in range(len(terms)):
            variance += (
                weights[i]
                * weights[j]
                * covariance.loc[terms[i], terms[j]]
            )

    # Guard against tiny negative values from numerical rounding.
    variance = max(float(variance), 0.0)
    standard_error = np.sqrt(variance)

    z_statistic = (
        estimate / standard_error
        if standard_error > 0 else np.nan
    )

    p_value = (
        2 * norm.sf(abs(z_statistic))
        if np.isfinite(z_statistic) else np.nan
    )

    critical_value = norm.ppf(
        1 - (1 - CONFIDENCE_LEVEL) / 2
    )

    lower_ci = estimate - critical_value * standard_error
    upper_ci = estimate + critical_value * standard_error

    return {
        "model": model_name,
        "quantile": quantile,
        "gender": gender,
        "education_slope": estimate,
        "standard_error": standard_error,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "ci_lower": lower_ci,
        "ci_upper": upper_ci,
        "approx_percent_association": 100 * estimate,
        "exact_percent_association": 100 * np.expm1(estimate),
    }


# ---------------------------------------------------------
# 5. Estimate gender-specific education slopes in OLS
# ---------------------------------------------------------

ols_covariance = get_covariance_matrix(ols_result)

ols_slopes = []

for gender, terms, weights in [
    ("Men", ["educ"], [1.0]),
    ("Women", ["educ", "educ:female"], [1.0, 1.0]),
]:
    ols_slopes.append(
        calculate_slope(
            result=ols_result,
            covariance=ols_covariance,
            terms=terms,
            weights=weights,
            quantile="Mean",
            gender=gender,
            model_name="OLS"
        )
    )


# ---------------------------------------------------------
# 6. Quantile regression with gender interaction
# ---------------------------------------------------------

qr_model = smf.quantreg(FORMULA, data=df)

coefficient_records = []
slope_records = []
summary_file = TABLE_DIR / "gender_interaction_qr_summaries.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("GENDER INTERACTION QUANTILE REGRESSION\n")
    file.write("=" * 75 + "\n")
    file.write(f"Formula: {FORMULA}\n")
    file.write(f"Observations: {len(df)}\n")

    for tau in QUANTILES:

        print(f"\nEstimating quantile: {tau:.2f}")

        result = qr_model.fit(
            q=tau,
            vcov="robust",
            kernel="epa",
            bandwidth="hsheather",
            max_iter=10000,
            p_tol=1e-6
        )

        covariance = get_covariance_matrix(result)

        print(f"Converged in {result.iterations} iterations.")

        file.write("\n" + "=" * 75 + "\n")
        file.write(f"QUANTILE: {tau:.2f}\n")
        file.write("=" * 75 + "\n")
        file.write(result.summary().as_text())
        file.write("\n")

        # Save all regression coefficients and inference statistics.
        for variable in result.params.index:

            coefficient_records.append({
                "quantile": tau,
                "variable": variable,
                "coefficient": result.params[variable],
                "standard_error": result.bse[variable],
                "z_statistic": result.tvalues[variable],
                "p_value": result.pvalues[variable],
                "ci_lower": result.conf_int().loc[variable, 0],
                "ci_upper": result.conf_int().loc[variable, 1],
                "pseudo_r_squared": result.prsquared,
                "iterations": result.iterations,
            })

        # Education slope for men:
        # beta_educ(tau)
        men_slope = calculate_slope(
            result,
            covariance,
            ["educ"],
            [1.0],
            tau,
            "Men",
            "Quantile Regression"
        )

        # Education slope for women:
        # beta_educ(tau) + beta_educ:female(tau)
        women_slope = calculate_slope(
            result,
            covariance,
            ["educ", "educ:female"],
            [1.0, 1.0],
            tau,
            "Women",
            "Quantile Regression"
        )

        slope_records.extend([men_slope, women_slope])

        # Test whether the education slopes differ by gender.
        interaction_p_value = result.pvalues["educ:female"]

        print(
            f"Men's education slope: "
            f"{men_slope['education_slope']:.5f}"
        )
        print(
            f"Women's education slope: "
            f"{women_slope['education_slope']:.5f}"
        )
        print(
            f"Gender interaction p-value: "
            f"{interaction_p_value:.5f}"
        )


# ---------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------

coefficient_df = pd.DataFrame(coefficient_records)
slope_df = pd.DataFrame(slope_records)

coefficient_df.to_csv(
    TABLE_DIR / "gender_interaction_qr_coefficients.csv",
    index=False
)

slope_df.to_csv(
    TABLE_DIR / "gender_specific_education_slopes.csv",
    index=False
)

ols_slopes_df = pd.DataFrame(ols_slopes)

comparison_df = pd.concat(
    [ols_slopes_df, slope_df],
    ignore_index=True
)

comparison_df.to_csv(
    TABLE_DIR / "gender_interaction_ols_qr_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# 8. Plot gender-specific education slopes across quantiles
# ---------------------------------------------------------

qr_slopes = slope_df[
    slope_df["model"] == "Quantile Regression"
].copy()

fig, ax = plt.subplots(figsize=(10, 6))

for gender in ["Men", "Women"]:

    subset = qr_slopes[
        qr_slopes["gender"] == gender
    ].sort_values("quantile")

    ax.plot(
        subset["quantile"],
        subset["education_slope"],
        marker="o",
        linewidth=2,
        label=gender
    )

    ax.fill_between(
        subset["quantile"],
        subset["ci_lower"],
        subset["ci_upper"],
        alpha=0.15
    )

ax.set_title(
    "Education-Wage Association by Gender and Wage Quantile"
)

ax.set_xlabel("Conditional Wage Quantile")
ax.set_ylabel("Education Coefficient")
ax.set_xticks(QUANTILES)
ax.axhline(0, linewidth=0.8, linestyle="--")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "08_gender_education_slopes_by_quantile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# 9. Final output
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("GENDER INTERACTION ANALYSIS COMPLETED")
print("=" * 65)

print("\nGender-specific education associations:")
print(
    slope_df[
        [
            "quantile",
            "gender",
            "education_slope",
            "standard_error",
            "p_value",
            "ci_lower",
            "ci_upper",
            "exact_percent_association",
        ]
    ].to_string(index=False)
)

print("\nGenerated files:")
print("- results/tables/gender_interaction_qr_coefficients.csv")
print("- results/tables/gender_specific_education_slopes.csv")
print("- results/tables/gender_interaction_ols_qr_comparison.csv")
print("- results/tables/gender_interaction_qr_summaries.txt")
print("- results/figures/08_gender_education_slopes_by_quantile.png")

print("\nAll estimations and outputs completed.")
