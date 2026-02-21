# Exploratory Data Analysis

> Phase 1 — Data Wrangling | Kata 1.7

---

## Concept & Intuition

### What problem are we solving?

Exploratory Data Analysis (EDA) is the process of visually and statistically summarizing a dataset to discover patterns, relationships, anomalies, and structure before building any model. It is the detective work of data science — looking at the data from every angle to understand what story it tells and what surprises it hides. Without EDA, you are modeling blindly, and blind modeling leads to garbage results.

EDA answers critical questions: What is the distribution of each feature? Are there correlations between features? Are there clusters or groups in the data? Which features are likely predictive of the target? Are there nonlinear relationships that a simple linear model would miss? These insights directly inform every downstream decision — which features to include, what transformations to apply, which model family to choose, and what performance to expect.

The core tools of EDA are **correlation analysis** (measuring linear relationships between pairs of features), **distribution analysis** (histograms, KDEs, box plots), and **multivariate visualization** (pairplots, heatmaps, scatter matrices). Together, these tools paint a comprehensive picture of your data's structure and guide the entire modeling process.

### Why naive approaches fail

The biggest failure mode is skipping EDA entirely and jumping straight to modeling. This leads to models that incorporate irrelevant features, miss important interactions, or are fit to distributions they cannot handle (e.g., linear models on highly skewed data). A five-minute pairplot often reveals more than an hour of hyperparameter tuning.

Another common mistake is only looking at summary statistics (mean, std) without visualizing the data. Anscombe's Quartet — four datasets with identical means, variances, and correlations but completely different visual patterns — is the classic demonstration of why visualization is irreplaceable. Summary statistics can be misleading; plots show the truth.

### Mental models

- **EDA as a conversation with your data**: You ask questions (Is age correlated with income? Is the target balanced?), the data answers through plots and statistics, and each answer leads to new questions
- **The funnel approach**: Start broad (overall shape, missing values, distributions), then narrow to specific relationships (correlations, group differences), then focus on target-related patterns
- **Correlation is not causation, but it is a clue**: A strong correlation between two features tells you they are related but not why. Use domain knowledge to interpret statistical findings
- **The 80/20 rule of EDA**: 80% of useful insights come from 20% of the possible analyses — focus on distributions, correlations, and the relationship between features and the target

### Visual explanations

```
The EDA Funnel
===============

  BROAD    ┌──────────────────────────────────────┐
           │  1. Shape, dtypes, missing values     │
           │  2. Summary statistics (describe)     │
           └──────────┬───────────────────────────┘
                      │
  MEDIUM   ┌──────────▼───────────────────────────┐
           │  3. Distributions (histograms)        │
           │  4. Correlations (heatmap)            │
           │  5. Group comparisons (box plots)     │
           └──────────┬───────────────────────────┘
                      │
  FOCUSED  ┌──────────▼───────────────────────────┐
           │  6. Feature vs target relationships   │
           │  7. Pairplots for key features        │
           │  8. Anomaly and outlier investigation  │
           └──────────────────────────────────────┘


Correlation Heatmap Reading Guide
===================================

  +1.0  Strong positive (both go up together)
  +0.5  Moderate positive
   0.0  No linear relationship
  -0.5  Moderate negative
  -1.0  Strong negative (one up, other down)

  Watch for:
  - Features highly correlated with target (useful!)
  - Features highly correlated with each other (redundant!)
  - Unexpected zero correlations (nonlinear relationships?)
```

---

## Hands-on Exploration

1. **Run the full EDA pipeline**: Execute the code below to see a comprehensive EDA of the housing dataset. Read every plot carefully and write down three things you learned about the data that you did not expect.

2. **Identify the best predictors**: From the correlation heatmap, identify the top 3 features most correlated with the target. Are these correlations expected given the domain?

3. **Spot the problems**: Look for features with skewed distributions, high correlations with each other (multicollinearity), or unusual patterns. These will need special handling before modeling.

4. **Generate hypotheses**: Based on the EDA, write down three hypotheses about what a good model for this data would look like (e.g., "A model using features X and Y should achieve R-squared > 0.7" or "Feature Z needs a log transform").

---

## Live Code

```python
"""
Exploratory Data Analysis — Distributions, correlations, and pairplots.

This code performs a comprehensive EDA on a synthetic housing dataset,
demonstrating the key visualizations and statistical summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Part 1: Create a Realistic Housing Dataset
# ============================================================

np.random.seed(42)
n = 500

# Generate correlated features
sqft = np.random.normal(1800, 500, n).clip(500, 5000)
bedrooms = (sqft / 600 + np.random.normal(0, 0.5, n)).clip(1, 7).round()
bathrooms = (bedrooms * 0.6 + np.random.normal(0, 0.3, n)).clip(1, 5).round(1)
age = np.random.exponential(15, n).clip(0, 80).round()
garage_size = np.random.choice([0, 1, 2, 3], size=n, p=[0.1, 0.3, 0.4, 0.2])
lot_size = np.random.lognormal(8.5, 0.4, n).round()
neighborhood_quality = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.35, 0.30, 0.15])

# Target: price (depends on all features)
price = (50 * sqft
         + 10000 * bedrooms
         + 15000 * bathrooms
         - 500 * age
         + 20000 * garage_size
         + 0.5 * lot_size
         + 30000 * neighborhood_quality
         + np.random.normal(0, 20000, n))
price = price.clip(50000)

df = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "garage_size": garage_size,
    "lot_size": lot_size,
    "neighborhood_quality": neighborhood_quality,
    "price": price,
})

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS — Housing Dataset")
print("=" * 60)

# ============================================================
# Part 2: Basic Profiling
# ============================================================

print(f"\n--- Shape: {df.shape[0]} rows, {df.shape[1]} columns ---")
print(f"\n--- Data Types ---")
print(df.dtypes.to_string())
print(f"\n--- Summary Statistics ---")
print(df.describe().round(2).to_string())
print(f"\n--- Missing Values ---")
print(df.isnull().sum().to_string())

# ============================================================
# Part 3: Distribution Analysis
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for idx, col in enumerate(df.columns):
    row, col_idx = divmod(idx, 4)
    ax = axes[row, col_idx]
    ax.hist(df[col], bins=30, color="#3498db", edgecolor="black", alpha=0.7)
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")

    # Add skewness annotation
    skew = df[col].skew()
    ax.text(0.95, 0.95, f"skew={skew:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))

plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# Part 4: Correlation Analysis
# ============================================================

corr_matrix = df.corr()

print("\n--- Correlations with Target (price) ---")
target_corr = corr_matrix["price"].drop("price").sort_values(ascending=False)
for feat, corr in target_corr.items():
    bar = "+" * int(abs(corr) * 30) if corr > 0 else "-" * int(abs(corr) * 30)
    print(f"  {feat:<25} {corr:>6.3f}  {bar}")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(corr_matrix.columns, fontsize=10)

# Add correlation values as text
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, label="Correlation")
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# Part 5: Feature vs Target Relationships
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

features = [c for c in df.columns if c != "price"]
for idx, feat in enumerate(features):
    row, col_idx = divmod(idx, 4)
    ax = axes[row, col_idx]
    ax.scatter(df[feat], df["price"], alpha=0.3, s=15, c="#3498db")
    ax.set_xlabel(feat)
    ax.set_ylabel("Price ($)")
    corr = df[feat].corr(df["price"])
    ax.set_title(f"{feat} vs Price (r={corr:.3f})", fontsize=10, fontweight="bold")

    # Add trend line
    z = np.polyfit(df[feat], df["price"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(x_range, p(x_range), "r-", linewidth=2, alpha=0.7)

# Hide the empty subplot
axes[1, 3].axis("off")

plt.suptitle("Feature vs Target (Price) Relationships", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# Part 6: Pairplot for Top Features
# ============================================================

top_features = target_corr.abs().nlargest(4).index.tolist()
top_features.append("price")
df_top = df[top_features]

fig, axes = plt.subplots(len(top_features), len(top_features), figsize=(14, 14))

for i, feat_i in enumerate(top_features):
    for j, feat_j in enumerate(top_features):
        ax = axes[i, j]
        if i == j:
            ax.hist(df_top[feat_i], bins=25, color="#3498db", edgecolor="black", alpha=0.7)
        else:
            ax.scatter(df_top[feat_j], df_top[feat_i], alpha=0.2, s=10, c="#3498db")
        if j == 0:
            ax.set_ylabel(feat_i, fontsize=9)
        if i == len(top_features) - 1:
            ax.set_xlabel(feat_j, fontsize=9)
        ax.tick_params(labelsize=7)

plt.suptitle("Pairplot: Top 4 Features + Price", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# Part 7: Box Plots — Group Comparisons
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Price by neighborhood quality
ax = axes[0]
groups = [df[df["neighborhood_quality"] == q]["price"] for q in sorted(df["neighborhood_quality"].unique())]
bp = ax.boxplot(groups, labels=[f"Q{q}" for q in sorted(df["neighborhood_quality"].unique())],
                patch_artist=True)
for patch, color in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.2, 0.8, 5))):
    patch.set_facecolor(color)
ax.set_xlabel("Neighborhood Quality")
ax.set_ylabel("Price ($)")
ax.set_title("Price by Neighborhood Quality", fontweight="bold")

# Price by bedrooms
ax = axes[1]
bed_vals = sorted(df["bedrooms"].unique())
groups = [df[df["bedrooms"] == b]["price"] for b in bed_vals]
bp = ax.boxplot(groups, labels=[str(int(b)) for b in bed_vals], patch_artist=True)
for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(bed_vals)))):
    patch.set_facecolor(color)
ax.set_xlabel("Bedrooms")
ax.set_ylabel("Price ($)")
ax.set_title("Price by Bedrooms", fontweight="bold")

# Price by garage size
ax = axes[2]
garage_vals = sorted(df["garage_size"].unique())
groups = [df[df["garage_size"] == g]["price"] for g in garage_vals]
bp = ax.boxplot(groups, labels=[str(int(g)) for g in garage_vals], patch_artist=True)
for patch, color in zip(bp["boxes"], plt.cm.Pastel1(np.linspace(0, 1, len(garage_vals)))):
    patch.set_facecolor(color)
ax.set_xlabel("Garage Size")
ax.set_ylabel("Price ($)")
ax.set_title("Price by Garage Size", fontweight="bold")

plt.suptitle("Price Distribution by Categorical Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target (price): mean=${df['price'].mean():,.0f}, median=${df['price'].median():,.0f}")
print(f"\nTop 3 predictors of price:")
for feat, corr in target_corr.head(3).items():
    print(f"  {feat}: r = {corr:.3f}")
print(f"\nSkewed features (|skew| > 1):")
for col in df.columns:
    skew = df[col].skew()
    if abs(skew) > 1:
        print(f"  {col}: skew = {skew:.2f} (consider log transform)")
print(f"\nHighly correlated feature pairs (potential multicollinearity):")
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        if abs(corr_matrix.values[i, j]) > 0.7 and corr_matrix.columns[i] != "price":
            print(f"  {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: "
                  f"r = {corr_matrix.values[i, j]:.3f}")
```

---

## Key Takeaways

- **EDA is not optional** — it is the foundation of every successful modeling effort. Skipping EDA is like prescribing medicine without examining the patient
- **Visualize, do not just summarize**: Summary statistics can be misleading; histograms, scatter plots, and heatmaps reveal patterns, outliers, and nonlinearities that numbers alone cannot show
- **Follow the funnel**: Start broad (shape, dtypes, missing values), then explore distributions and correlations, then focus on feature-target relationships. This systematic approach ensures you miss nothing
- **Correlation heatmaps serve double duty**: They reveal which features predict the target (useful) AND which features are redundant with each other (multicollinearity risk)
- **EDA generates hypotheses** that guide your modeling choices — which features to transform, which to drop, what model complexity to start with, and what performance to expect
