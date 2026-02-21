# Missing Values

> Phase 1 — Data Wrangling | Kata 1.2

---

## Concept & Intuition

### What problem are we solving?

Missing data is one of the most universal problems in data science. Sensors malfunction, survey respondents skip questions, databases merge imperfectly, and records get corrupted. Nearly every real-world dataset has missing values, and how you handle them can make or break your analysis. Ignoring missing data or handling it carelessly introduces bias, reduces statistical power, and can lead models to learn spurious patterns.

The key insight is that **missing data is not random** — it carries information. Data can be Missing Completely At Random (MCAR), Missing At Random (MAR), or Missing Not At Random (MNAR). If high-income individuals are less likely to report their income, the missing values are MNAR, and simply dropping those rows systematically biases your analysis toward lower incomes. Understanding the mechanism behind missingness is essential to choosing the right imputation strategy.

Imputation strategies range from simple (fill with mean/median/mode) to sophisticated (model-based imputation using k-NN or iterative methods). Each strategy has tradeoffs: simple methods are fast but can distort distributions and relationships; complex methods are more accurate but computationally expensive and harder to implement correctly.

### Why naive approaches fail

The most common naive approach — dropping all rows with any missing value — can be catastrophic. If 20% of rows are missing a value in column A and a different 20% are missing in column B, dropping rows with any missing value could eliminate 36% of your data. Worse, if missingness is correlated with the target variable, dropping rows introduces selection bias.

Another naive approach is filling all missing values with the column mean. This preserves the mean but artificially shrinks the variance and weakens correlations between features. If "age" is correlated with "income" and you fill missing ages with the overall mean, you destroy that correlation for the imputed rows, making your model less accurate.

### Mental models

- **The puzzle analogy**: Missing values are like missing puzzle pieces. You can leave gaps (drop rows), fill them with a neutral color (mean imputation), or look at surrounding pieces to guess what belongs (model-based imputation)
- **Three types of missingness**: MCAR = pieces fell off the table randomly. MAR = pieces fell off because of something you can observe. MNAR = pieces are missing because of what is on the piece itself (hardest to handle)
- **Imputation is estimation, not truth**: Any imputed value is a guess. The question is not whether the guess is exactly right, but whether it is less harmful than the alternatives (dropping data, leaving gaps)
- **The chain of consequences**: Missing value handling -> distribution changes -> correlation changes -> model behavior changes. Every imputation decision ripples through the entire analysis

### Visual explanations

```
Types of Missing Data
======================

MCAR (Missing Completely At Random)
  Missingness is purely random, unrelated to any variable.
  Example: A sensor randomly fails 5% of the time.

  Data:  [23, 45, ?, 67, ?, 89, 12, ?, 56, 34]
  The ?s have no pattern — safe to drop or impute simply.

MAR (Missing At Random)
  Missingness depends on OBSERVED variables.
  Example: Younger people are less likely to report income.

  Age:    [25, 55, 30, 60, 22, 58, 28, 62]
  Income: [ ?,  80,  ?,  95,  ?,  88,  ?, 92]
  Pattern: young -> missing. Can use age to impute.

MNAR (Missing Not At Random)
  Missingness depends on the MISSING value itself.
  Example: People with high debt avoid reporting it.

  Debt:  [?, low, ?, low, ?, low, ?, medium]
  The ?s ARE the high-debt people. No easy fix!


Impact of Imputation Strategy
===============================

  Original:     [10, 20, ?, 40, 50]  Mean=30, Std=15.8

  Drop missing: [10, 20, 40, 50]     Mean=30, Std=17.1  (higher variance)
  Mean fill:    [10, 20, 30, 40, 50]  Mean=30, Std=14.1  (lower variance!)
  Median fill:  [10, 20, 30, 40, 50]  Mean=30, Std=14.1  (same as mean here)
  KNN fill:     [10, 20, 28, 40, 50]  Mean=29.6, Std=14.5 (context-aware)
```

---

## Hands-on Exploration

1. **Diagnose the missingness**: Run the code below to see the missing data pattern. Are the missing values random, or do they cluster in certain rows/columns? Look at the heatmap to identify patterns.

2. **Compare imputation strategies**: The code implements four strategies (drop, mean, median, KNN). Compare the resulting distributions for each. Which strategy best preserves the original distribution shape?

3. **Measure the damage**: Look at how each imputation strategy affects the correlation between features. A good imputation should preserve inter-feature correlations; a bad one destroys them.

4. **Stress test**: Try increasing the fraction of missing values from 10% to 50%. At what point does each strategy break down? This reveals the practical limits of each approach.

---

## Live Code

```python
"""
Missing Values — Detection, impact analysis, and imputation strategies.

This code creates a dataset with realistic missing value patterns, then
compares multiple imputation strategies and their effects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer

# ============================================================
# Part 1: Create Dataset with Realistic Missing Patterns
# ============================================================

np.random.seed(42)
n = 300

# Generate complete data first
age = np.random.normal(40, 12, n).clip(18, 75).round()
income = 20000 + 1500 * age + np.random.normal(0, 15000, n)
income = income.clip(15000)
satisfaction = 3 + 0.05 * (income / 10000) + np.random.normal(0, 1.5, n)
satisfaction = satisfaction.clip(1, 10).round(1)
hours_worked = 30 + 0.3 * age + np.random.normal(0, 5, n)
hours_worked = hours_worked.clip(20, 70).round(1)

df_complete = pd.DataFrame({
    "age": age,
    "income": income.round(2),
    "satisfaction": satisfaction,
    "hours_worked": hours_worked,
})

# Introduce missing values with different mechanisms
df = df_complete.copy()

# MCAR: randomly remove 10% of satisfaction scores
mcar_mask = np.random.random(n) < 0.10
df.loc[mcar_mask, "satisfaction"] = np.nan

# MAR: younger people less likely to report income
mar_prob = 0.3 * (1 - (df["age"] - 18) / (75 - 18))  # Higher prob for young
mar_mask = np.random.random(n) < mar_prob
df.loc[mar_mask, "income"] = np.nan

# MNAR: people working extreme hours don't report hours
mnar_mask = (df_complete["hours_worked"] > 55) & (np.random.random(n) < 0.6)
df.loc[mnar_mask, "hours_worked"] = np.nan

print("=" * 60)
print("MISSING VALUES — Detection and Analysis")
print("=" * 60)

print("\n--- Missing Value Summary ---")
missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": (df.isnull().sum() / len(df) * 100).round(1),
    "Mechanism": ["None", "MAR (age-dependent)", "MCAR (random)", "MNAR (value-dependent)"],
})
print(missing_summary.to_string())

# ============================================================
# Part 2: Visualize Missing Patterns
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Missing data heatmap
ax = axes[0]
missing_matrix = df.isnull().astype(int)
ax.imshow(missing_matrix.values[:50], cmap="YlOrRd", aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45, ha="right")
ax.set_ylabel("Row index (first 50 rows)")
ax.set_title("Missing Data Pattern\n(yellow=present, red=missing)", fontweight="bold")

# MAR evidence: missing income vs age
ax = axes[1]
has_income = ~df["income"].isnull()
ax.hist(df.loc[has_income, "age"], bins=20, alpha=0.7, label="Income present", color="#2ecc71", density=True)
ax.hist(df.loc[~has_income, "age"], bins=20, alpha=0.7, label="Income missing", color="#e74c3c", density=True)
ax.set_xlabel("Age")
ax.set_ylabel("Density")
ax.set_title("MAR Evidence: Missing Income vs Age", fontweight="bold")
ax.legend()

# MNAR evidence: compare reported vs complete hours
ax = axes[2]
has_hours = ~df["hours_worked"].isnull()
ax.hist(df_complete["hours_worked"], bins=20, alpha=0.5, label="True (complete)", color="#3498db", density=True)
ax.hist(df.loc[has_hours, "hours_worked"], bins=20, alpha=0.5, label="Observed (after MNAR)", color="#e74c3c", density=True)
ax.set_xlabel("Hours Worked")
ax.set_ylabel("Density")
ax.set_title("MNAR Effect: Missing High Values", fontweight="bold")
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# Part 3: Compare Imputation Strategies
# ============================================================

print("\n" + "=" * 60)
print("IMPUTATION STRATEGY COMPARISON")
print("=" * 60)

# Strategy 1: Drop rows with any missing value
df_dropped = df.dropna()

# Strategy 2: Mean imputation
mean_imputer = SimpleImputer(strategy="mean")
df_mean = pd.DataFrame(mean_imputer.fit_transform(df), columns=df.columns)

# Strategy 3: Median imputation
median_imputer = SimpleImputer(strategy="median")
df_median = pd.DataFrame(median_imputer.fit_transform(df), columns=df.columns)

# Strategy 4: KNN imputation (uses neighboring rows)
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

strategies = {
    "Original (complete)": df_complete,
    "Drop missing": df_dropped,
    "Mean imputation": df_mean,
    "Median imputation": df_median,
    "KNN imputation (k=5)": df_knn,
}

# Compare statistics
print(f"\n{'Strategy':<25} {'N rows':>8} {'Income Mean':>13} {'Income Std':>12} {'Hours Mean':>12}")
print("-" * 75)
for name, data in strategies.items():
    n_rows = len(data)
    inc_mean = data["income"].mean()
    inc_std = data["income"].std()
    hrs_mean = data["hours_worked"].mean()
    print(f"{name:<25} {n_rows:>8} {inc_mean:>13,.0f} {inc_std:>12,.0f} {hrs_mean:>12.1f}")

# ============================================================
# Part 4: Impact on Distributions
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Income distribution comparison
ax = axes[0, 0]
for name, data in strategies.items():
    ax.hist(data["income"].dropna(), bins=30, alpha=0.4, label=name, density=True)
ax.set_xlabel("Income")
ax.set_ylabel("Density")
ax.set_title("Income Distribution by Strategy", fontweight="bold")
ax.legend(fontsize=8)

# Hours distribution comparison
ax = axes[0, 1]
for name, data in strategies.items():
    ax.hist(data["hours_worked"].dropna(), bins=30, alpha=0.4, label=name, density=True)
ax.set_xlabel("Hours Worked")
ax.set_ylabel("Density")
ax.set_title("Hours Distribution by Strategy", fontweight="bold")
ax.legend(fontsize=8)

# Correlation preservation
ax = axes[1, 0]
true_corr = df_complete[["age", "income"]].corr().iloc[0, 1]
corr_values = []
corr_names = []
for name, data in strategies.items():
    corr = data[["age", "income"]].corr().iloc[0, 1]
    corr_values.append(corr)
    corr_names.append(name.replace(" ", "\n"))

colors = ["#2ecc71" if abs(c - true_corr) < 0.05 else "#e74c3c" for c in corr_values]
ax.bar(range(len(corr_values)), corr_values, color=colors, edgecolor="black", alpha=0.7)
ax.axhline(y=true_corr, color="blue", linestyle="--", label=f"True correlation: {true_corr:.3f}")
ax.set_xticks(range(len(corr_names)))
ax.set_xticklabels(corr_names, fontsize=8)
ax.set_ylabel("Correlation (age vs income)")
ax.set_title("Correlation Preservation", fontweight="bold")
ax.legend(fontsize=9)

# Variance preservation
ax = axes[1, 1]
true_var = df_complete["income"].var()
var_values = []
for name, data in strategies.items():
    var_values.append(data["income"].var())

colors = ["#2ecc71" if abs(v / true_var - 1) < 0.1 else "#e74c3c" for v in var_values]
bars = ax.bar(range(len(var_values)), [v / 1e9 for v in var_values], color=colors,
              edgecolor="black", alpha=0.7)
ax.axhline(y=true_var / 1e9, color="blue", linestyle="--", label=f"True variance")
ax.set_xticks(range(len(corr_names)))
ax.set_xticklabels(corr_names, fontsize=8)
ax.set_ylabel("Variance (x10^9)")
ax.set_title("Variance Preservation", fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

print("\nKey insight: Mean imputation shrinks variance and weakens correlations.")
print("KNN imputation better preserves the data's statistical properties,")
print("but no imputation method can fully recover MNAR data — the information")
print("is fundamentally lost when high values selectively go missing.")
```

---

## Key Takeaways

- **Missing data is not random** — understanding WHY data is missing (MCAR, MAR, MNAR) is essential to choosing the right handling strategy
- **Dropping rows is wasteful and biased**: It reduces sample size and can introduce systematic bias if missingness is correlated with the target variable
- **Mean/median imputation is simple but harmful**: It preserves the mean but artificially shrinks variance and destroys correlations between features
- **KNN and model-based imputation** better preserve the statistical structure of the data by using information from similar observations to fill gaps
- **MNAR is the hardest case**: When values are missing because of the value itself (e.g., high earners refusing to report income), no imputation strategy can fully recover the lost information
