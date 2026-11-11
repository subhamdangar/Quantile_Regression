
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATA_FILE = "data/wage1.csv"
FIGURE_DIR = "results/figures"
TABLE_DIR = "results/tables"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# ==========================================================
# 2. LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS: WOOLDRIDGE WAGE1")
print("=" * 70)

# ==========================================================
# 3. DISTRIBUTION STATISTICS
# ==========================================================

print("\n1. DISTRIBUTION STATISTICS")

distribution_stats = df[["wage", "lwage"]].agg(
    ["mean", "median", "std", "min", "max", "skew", "kurt"]
).T

print(distribution_stats.round(4))

distribution_stats.to_csv(
    os.path.join(TABLE_DIR, "wage_distribution_statistics.csv")
)

# ==========================================================
# 4. FIGURE 1: HOURLY WAGE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="wage",
    bins=30,
    kde=True
)

plt.axvline(
    df["wage"].mean(),
    linestyle="--",
    linewidth=2,
    label="Mean"
)

plt.axvline(
    df["wage"].median(),
    linestyle=":",
    linewidth=2,
    label="Median"
)

plt.title("Distribution of Hourly Wages", fontsize=15)
plt.xlabel("Hourly Wage (USD)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "01_wage_distribution.png"),
    dpi=300
)

plt.close()

# ==========================================================
# 5. FIGURE 2: LOG WAGE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="lwage",
    bins=30,
    kde=True
)

plt.axvline(
    df["lwage"].mean(),
    linestyle="--",
    linewidth=2,
    label="Mean"
)

plt.axvline(
    df["lwage"].median(),
    linestyle=":",
    linewidth=2,
    label="Median"
)

plt.title("Distribution of Log Hourly Wages", fontsize=15)
plt.xlabel("Log Hourly Wage")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "02_log_wage_distribution.png"),
    dpi=300
)

plt.close()

# ==========================================================
# 6. FIGURE 3: WAGE DISTRIBUTION BY GENDER
# ==========================================================

df["gender_label"] = df["female"].map({
    0: "Male",
    1: "Female"
})

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="gender_label",
    y="wage"
)

plt.title("Hourly Wage Distribution by Gender", fontsize=15)
plt.xlabel("Gender")
plt.ylabel("Hourly Wage (USD)")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "03_wage_by_gender.png"),
    dpi=300
)

plt.close()

# ==========================================================
# 7. FIGURE 4: EDUCATION AND LOG WAGES
# ==========================================================

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="educ",
    y="lwage",
    lowess=True,
    scatter_kws={"alpha": 0.45},
    line_kws={"linewidth": 2}
)

plt.title(
    "Relationship Between Education and Log Wages",
    fontsize=15
)

plt.xlabel("Years of Education")
plt.ylabel("Log Hourly Wage")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "04_education_vs_logwage.png"),
    dpi=300
)

plt.close()

# ==========================================================
# 8. FIGURE 5: EXPERIENCE AND LOG WAGES
# ==========================================================

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="exper",
    y="lwage",
    lowess=True,
    scatter_kws={"alpha": 0.45},
    line_kws={"linewidth": 2}
)

plt.title(
    "Relationship Between Experience and Log Wages",
    fontsize=15
)

plt.xlabel("Years of Potential Experience")
plt.ylabel("Log Hourly Wage")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "05_experience_vs_logwage.png"),
    dpi=300
)

plt.close()

# ==========================================================
# 9. FIGURE 6: CORRELATION HEATMAP
# ==========================================================

correlation_variables = [
    "lwage",
    "educ",
    "exper",
    "expersq",
    "tenure",
    "tenursq",
    "female",
    "married",
    "nonwhite",
    "smsa"
]

correlation_matrix = df[
    correlation_variables
].corr()

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)

plt.title("Correlation Matrix of Selected Variables", fontsize=15)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "06_correlation_heatmap.png"),
    dpi=300
)

plt.close()

# ==========================================================
# 10. CONDITIONAL DESCRIPTIVE QUANTILES BY GENDER
# ==========================================================

print("\n2. WAGE QUANTILES BY GENDER")

gender_quantiles = df.groupby("gender_label")["lwage"].quantile(
    [0.10, 0.25, 0.50, 0.75, 0.90]
).unstack()

gender_quantiles.columns = [
    "Q10", "Q25", "Q50", "Q75", "Q90"
]

print(gender_quantiles.round(4))

gender_quantiles.to_csv(
    os.path.join(TABLE_DIR, "gender_wage_quantiles.csv")
)

# ==========================================================
# 11. FINAL OUTPUT
# ==========================================================

print("\n3. GENERATED FIGURES")

for filename in sorted(os.listdir(FIGURE_DIR)):
    if filename.endswith(".png"):
        print(filename)

print("\nEDA COMPLETED SUCCESSFULLY.")

print("=" * 70)
