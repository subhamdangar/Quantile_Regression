
"""
Bootstrap inference for heterogeneous effects in quantile regression.

Dataset: Wooldridge wage1
Dependent variable: lwage
Main explanatory variable: educ
Heterogeneity variable: female

Bootstrap method: Pairs bootstrap
Number of replications: 500

Tests:
1. Differences in men's education slopes across quantiles.
2. Differences in women's education slopes across quantiles.
3. Differences in gender gaps across quantiles.
4. Gender differences at each quantile.

Multiple testing correction: Holm procedure.
"""

from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "wage1.csv"

TABLE_DIR = BASE_DIR / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

FORMULA = (
    "lwage ~ educ * female + exper + expersq + tenure + tenursq"
)

N_BOOTSTRAP = 500
RANDOM_SEED = 42

MAX_ITER = 10000
P_TOL = 1e-6

CONFIDENCE_LEVEL = 0.95


# =========================================================
# 2. LOAD AND VALIDATE DATA
# =========================================================

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

missing = [
    col for col in required_columns if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required_columns].isna().any().any():
    raise ValueError("Missing values found in model variables.")

if not np.isfinite(df[required_columns].to_numpy()).all():
    raise ValueError("Non-finite values found in model variables.")

if not set(df["female"].unique()).issubset({0, 1}):
    raise ValueError("female must be coded 0 and 1.")

n = len(df)

print("=" * 70)
print("BOOTSTRAP TESTS FOR HETEROGENEOUS EFFECTS")
print("=" * 70)

print(f"Observations: {n}")
print(f"Bootstrap replications requested: {N_BOOTSTRAP}")
print(f"Quantiles: {QUANTILES}")
print(f"Random seed: {RANDOM_SEED}")


# =========================================================
# 3. ESTIMATE THE ORIGINAL SAMPLE MODEL
# =========================================================

print("\nEstimating original-sample quantile regressions...")

original_model = smf.quantreg(FORMULA, data=df)

original_estimates = {}

for tau in QUANTILES:

    result = original_model.fit(
        q=tau,
        vcov="iid",
        max_iter=MAX_ITER,
        p_tol=P_TOL
    )

    original_estimates[tau] = result.params

    print(
        f"Quantile {tau:.2f}: "
        f"converged in {result.iterations} iterations"
    )


# =========================================================
# 4. PAIRS BOOTSTRAP
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_records = []
successful_replications = 0
failed_replications = 0

print("\nStarting pairs bootstrap...")
print("Each replication estimates all five quantile regressions.")

for b in range(N_BOOTSTRAP):

    # Resample complete observations with replacement.
    sampled_indices = rng.integers(0, n, size=n)

    bootstrap_sample = (
        df.iloc[sampled_indices]
        .reset_index(drop=True)
    )

    bootstrap_model = smf.quantreg(
        FORMULA,
        data=bootstrap_sample
    )

    replication_estimates = {}

    try:

        for tau in QUANTILES:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = bootstrap_model.fit(
                    q=tau,
                    vcov="iid",
                    max_iter=MAX_ITER,
                    p_tol=P_TOL
                )

            if not np.isfinite(result.params.to_numpy()).all():
                raise ValueError(
                    f"Non-finite estimates at quantile {tau}"
                )

            replication_estimates[tau] = result.params

        # Save the estimates only when all five quantiles
        # have been successfully estimated.
        for tau in QUANTILES:

            params = replication_estimates[tau]

            men_slope = params["educ"]

            women_slope = (
                params["educ"] + params["educ:female"]
            )

            gender_gap = params["educ:female"]

            bootstrap_records.append({
                "replication": b + 1,
                "quantile": tau,
                "men_slope": men_slope,
                "women_slope": women_slope,
                "gender_gap": gender_gap,
            })

        successful_replications += 1

    except Exception as error:

        failed_replications += 1

        if failed_replications <= 5:
            print(
                f"Replication {b + 1} failed: {error}"
            )

    if (b + 1) % 25 == 0:

        print(
            f"Completed {b + 1}/{N_BOOTSTRAP} | "
            f"Successful: {successful_replications} | "
            f"Failed: {failed_replications}"
        )


# =========================================================
# 5. CHECK BOOTSTRAP COMPLETION
# =========================================================

print("\nBootstrap estimation finished.")

print(f"Successful replications: {successful_replications}")
print(f"Failed replications: {failed_replications}")

if successful_replications < 0.90 * N_BOOTSTRAP:
    raise RuntimeError(
        "Fewer than 90% of bootstrap replications succeeded. "
        "Inspect the errors before interpreting the results."
    )

if successful_replications < N_BOOTSTRAP:
    print(
        "WARNING: Some replications failed. "
        "Inference will use successful replications only."
    )

bootstrap_df = pd.DataFrame(bootstrap_records)

bootstrap_df.to_csv(
    TABLE_DIR / "bootstrap_coefficient_draws.csv",
    index=False
)


# =========================================================
# 6. ORIGINAL-SAMPLE ESTIMATES
# =========================================================

original_rows = []

for tau in QUANTILES:

    params = original_estimates[tau]

    original_rows.append({
        "quantile": tau,
        "men_slope": params["educ"],
        "women_slope": (
            params["educ"] + params["educ:female"]
        ),
        "gender_gap": params["educ:female"],
    })

original_df = pd.DataFrame(original_rows)


# =========================================================
# 7. HELPER FUNCTION FOR BOOTSTRAP INFERENCE
# =========================================================

def bootstrap_inference(
    bootstrap_values,
    original_estimate
):
    """
    Calculate percentile confidence intervals and
    a two-sided centered-bootstrap p-value.

    The p-value compares the bootstrap deviations from
    the original estimate with the observed distance
    from the null value zero.
    """

    values = np.asarray(bootstrap_values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) < 2:
        raise ValueError(
            "Insufficient valid bootstrap estimates."
        )

    alpha = 1 - CONFIDENCE_LEVEL

    lower_ci = np.quantile(values, alpha / 2)
    upper_ci = np.quantile(values, 1 - alpha / 2)

    centered_values = values - original_estimate

    # Add-one correction avoids reporting a zero p-value.
    extreme_count = np.sum(
        np.abs(centered_values) >= abs(original_estimate)
    )

    p_value = (
        extreme_count + 1
    ) / (
        len(values) + 1
    )

    bootstrap_se = np.std(values, ddof=1)

    return {
        "estimate": original_estimate,
        "bootstrap_se": bootstrap_se,
        "ci_lower": lower_ci,
        "ci_upper": upper_ci,
        "p_value": p_value,
        "bootstrap_replications": len(values),
    }


# =========================================================
# 8. TEST GENDER-SPECIFIC EFFECTS AT EACH QUANTILE
# =========================================================

gender_test_records = []

for tau in QUANTILES:

    point = original_df.loc[
        original_df["quantile"] == tau
    ].iloc[0]

    draws = bootstrap_df.loc[
        bootstrap_df["quantile"] == tau,
        "gender_gap"
    ].to_numpy()

    inference = bootstrap_inference(
        draws,
        point["gender_gap"]
    )

    gender_test_records.append({
        "test_family": "Gender difference within quantile",
        "quantile": tau,
        **inference,
    })

gender_tests_df = pd.DataFrame(gender_test_records)

gender_tests_df["holm_adjusted_p"] = multipletests(
    gender_tests_df["p_value"],
    method="holm"
)[1]

gender_tests_df["significant_5_percent"] = (
    gender_tests_df["holm_adjusted_p"] < 0.05
)


# =========================================================
# 9. PAIRWISE TESTS ACROSS QUANTILES
# =========================================================

comparison_records = []

effect_columns = {
    "Men's education association": "men_slope",
    "Women's education association": "women_slope",
    "Gender difference": "gender_gap",
}

for effect_name, column in effect_columns.items():

    for tau1, tau2 in combinations(QUANTILES, 2):

        original_value_1 = original_df.loc[
            original_df["quantile"] == tau1,
            column
        ].iloc[0]

        original_value_2 = original_df.loc[
            original_df["quantile"] == tau2,
            column
        ].iloc[0]

        original_difference = (
            original_value_1 - original_value_2
        )

        bootstrap_values_1 = (
            bootstrap_df.loc[
                bootstrap_df["quantile"] == tau1,
                column
            ].to_numpy()
        )

        bootstrap_values_2 = (
            bootstrap_df.loc[
                bootstrap_df["quantile"] == tau2,
                column
            ].to_numpy()
        )

        if len(bootstrap_values_1) != len(bootstrap_values_2):
            raise RuntimeError(
                "Bootstrap samples are not aligned across quantiles."
            )

        bootstrap_differences = (
            bootstrap_values_1 - bootstrap_values_2
        )

        inference = bootstrap_inference(
            bootstrap_differences,
            original_difference
        )

        comparison_records.append({
            "test_family": effect_name,
            "quantile_1": tau1,
            "quantile_2": tau2,
            **inference,
        })

comparisons_df = pd.DataFrame(comparison_records)


# =========================================================
# 10. HOLM MULTIPLE-TESTING CORRECTION
# =========================================================

comparisons_df["holm_adjusted_p"] = np.nan

for family in comparisons_df["test_family"].unique():

    mask = (
        comparisons_df["test_family"] == family
    )

    adjusted = multipletests(
        comparisons_df.loc[mask, "p_value"],
        method="holm"
    )[1]

    comparisons_df.loc[
        mask, "holm_adjusted_p"
    ] = adjusted

comparisons_df["significant_5_percent"] = (
    comparisons_df["holm_adjusted_p"] < 0.05
)


# =========================================================
# 11. SAVE RESULTS
# =========================================================

gender_tests_df.to_csv(
    TABLE_DIR / "bootstrap_gender_difference_by_quantile.csv",
    index=False
)

comparisons_df.to_csv(
    TABLE_DIR / "bootstrap_cross_quantile_tests.csv",
    index=False
)

original_df.to_csv(
    TABLE_DIR / "bootstrap_original_sample_estimates.csv",
    index=False
)


# =========================================================
# 12. DISPLAY RESULTS
# =========================================================

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 20)

print("\n" + "=" * 70)
print("GENDER DIFFERENCE AT EACH QUANTILE")
print("=" * 70)

print(
    gender_tests_df[
        [
            "quantile",
            "estimate",
            "bootstrap_se",
            "ci_lower",
            "ci_upper",
            "p_value",
            "holm_adjusted_p",
            "significant_5_percent",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("CROSS-QUANTILE HETEROGENEITY TESTS")
print("=" * 70)

for family in comparisons_df["test_family"].unique():

    print(f"\n{family}")
    print("-" * 70)

    subset = comparisons_df[
        comparisons_df["test_family"] == family
    ]

    print(
        subset[
            [
                "quantile_1",
                "quantile_2",
                "estimate",
                "bootstrap_se",
                "ci_lower",
                "ci_upper",
                "p_value",
                "holm_adjusted_p",
                "significant_5_percent",
            ]
        ].to_string(index=False)
    )


# =========================================================
# 13. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("BOOTSTRAP ANALYSIS COMPLETED")
print("=" * 70)

print(f"Successful replications: {successful_replications}")
print(f"Failed replications: {failed_replications}")

print("\nGenerated files:")

print(
    "- results/tables/bootstrap_coefficient_draws.csv"
)
print(
    "- results/tables/bootstrap_gender_difference_by_quantile.csv"
)
print(
    "- results/tables/bootstrap_cross_quantile_tests.csv"
)
print(
    "- results/tables/bootstrap_original_sample_estimates.csv"
)

print("\nInterpretation rule:")
print(
    "A Holm-adjusted p-value below 0.05 provides evidence "
    "against the corresponding null hypothesis."
)

print(
    "A 95% bootstrap confidence interval excluding zero "
    "also indicates a statistically significant difference "
    "at the unadjusted 5% level."
)
