# Outlier Detection

> Phase 1 — Data Wrangling | Kata 1.3

---

## Concept & Intuition

### What problem are we solving?

Outliers are data points that deviate significantly from the rest of the dataset. They might be measurement errors (a sensor reading of -999), data entry mistakes (an age of 999 years), or genuinely extreme but valid observations (a billionaire's income in a salary survey). Detecting and appropriately handling outliers is critical because they can distort statistical analyses, inflate error metrics, and cause models to learn spurious patterns.

The challenge is that there is no universal definition of "outlier." What counts as extreme depends on the distribution of the data, the domain, and the analysis you are performing. A temperature of 50 degrees Celsius is an outlier in a weather dataset from London but perfectly normal in a dataset from the Sahara Desert. This is why outlier detection requires both statistical methods and domain knowledge.

The three most common statistical methods are the **Z-score** (based on standard deviations from the mean), the **Interquartile Range (IQR)** method (based on the spread of the middle 50% of data), and **visual methods** (box plots, scatter plots). Each has strengths: Z-scores work well for normally distributed data, IQR is robust to non-normal distributions, and visual methods provide intuitive understanding that numbers alone cannot.

### Why naive approaches fail

The naive approach of removing all statistical outliers is dangerous. In many real-world scenarios, the "outliers" ARE the interesting data. Fraud detection is literally the problem of finding outliers. In medical research, rare adverse reactions (outliers) are exactly what you need to study. Blindly removing outliers can eliminate the most valuable information in your dataset.

Another pitfall is using Z-scores on heavily skewed distributions. The Z-score assumes approximate normality. If your data is right-skewed (like income or housing prices), the mean and standard deviation are themselves distorted by the extreme values, making the Z-score unreliable. The IQR method is more robust in these cases because quartiles are resistant to extreme values.

### Mental models

- **The signal vs. noise lens**: An outlier is either signal (genuine extreme value worth studying) or noise (error worth removing). Your job is to figure out which
- **Robust vs. fragile statistics**: The mean is fragile (one outlier can shift it dramatically). The median is robust (outliers barely affect it). IQR-based methods use robust statistics
- **Context is king**: The same data point can be an outlier in one context and perfectly normal in another. Always ask "outlier relative to what?"
- **The three fates of an outlier**: Keep it (it is real and informative), remove it (it is an error), or transform it (cap it at a threshold to reduce its influence while keeping it in the dataset)

### Visual explanations

```
Z-Score Method
===============

  For a data point x:
    z = (x - mean) / std

  Common threshold: |z| > 3

  Normal distribution:
       ____
      /    \
     /      \
    / |    | \
   /  |    |  \       * Outlier (z > 3)
  /____|____|____\___*___
  -3  -2   0   2   3

  99.7% of data falls within z = +/- 3
  Points beyond are flagged as potential outliers


IQR Method
===========

  IQR = Q3 - Q1 (interquartile range)
  Lower fence = Q1 - 1.5 * IQR
  Upper fence = Q3 + 1.5 * IQR

  Box plot:

  Outliers      |---[ Q1 |  Q2  | Q3 ]---|      Outliers
     *     *    |        |      |        |    *
  ──────────────+────────+──────+────────+──────────────
              lower                     upper
              fence                     fence
              (Q1-1.5*IQR)             (Q3+1.5*IQR)

  Points outside the fences are potential outliers
```

---

## Hands-on Exploration

1. **Compare detection methods**: Run the code below to see how Z-score and IQR methods flag different points as outliers on the same dataset. Note which method is more conservative (flags fewer points) and which is more aggressive.

2. **Test with skewed data**: The dataset includes both normally distributed and skewed features. Compare how well Z-score and IQR perform on each. Notice that Z-score struggles with skewed data.

3. **Visualize the impact**: Look at the scatter plots before and after outlier removal. How does removing outliers change the apparent relationship between variables?

4. **Practice the judgment call**: For each detected outlier, decide whether it is likely a measurement error or a genuine extreme value. What additional information would you need to make this decision confidently?

---

## Live Code

```python
"""
Outlier Detection — Z-score, IQR, and visual methods for identifying anomalies.

This code generates a dataset with known outliers, then applies multiple
detection methods and compares their effectiveness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Part 1: Generate Data with Known Outliers
# ============================================================

np.random.seed(42)
n = 500

# Normal feature (symmetric)
normal_feature = np.random.normal(50, 10, n)
# Inject known outliers
outlier_indices_normal = [0, 1, 2, 3, 4]
normal_feature[outlier_indices_normal] = [120, -30, 110, 130, -25]

# Skewed feature (income-like, right-skewed)
skewed_feature = np.random.lognormal(mean=10, sigma=0.5, n)
# Inject outliers at the high end
outlier_indices_skewed = [5, 6, 7, 8]
skewed_feature[outlier_indices_skewed] = [500000, 600000, 450000, 550000]

# Correlated feature for scatter plot analysis
correlated = 0.8 * normal_feature + np.random.normal(0, 5, n)

df = pd.DataFrame({
    "normal_feature": normal_feature,
    "skewed_feature": skewed_feature,
    "correlated": correlated,
})

print("=" * 60)
print("OUTLIER DETECTION — Methods and Comparison")
print("=" * 60)

# ============================================================
# Part 2: Z-Score Method
# ============================================================

def detect_outliers_zscore(data, threshold=3.0):
    """Detect outliers using Z-score method."""
    mean = data.mean()
    std = data.std()
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold, z_scores

print("\n--- Z-Score Method (threshold = 3.0) ---")
for col in ["normal_feature", "skewed_feature"]:
    outlier_mask, z_scores = detect_outliers_zscore(df[col])
    n_outliers = outlier_mask.sum()
    print(f"\n  {col}:")
    print(f"    Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
    print(f"    Outliers detected: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    if n_outliers > 0:
        outlier_vals = df.loc[outlier_mask, col].values
        print(f"    Outlier values: {outlier_vals[:10].round(2)}")

# ============================================================
# Part 3: IQR Method
# ============================================================

def detect_outliers_iqr(data, factor=1.5):
    """Detect outliers using IQR method."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outlier_mask = (data < lower) | (data > upper)
    return outlier_mask, lower, upper

print("\n\n--- IQR Method (factor = 1.5) ---")
for col in ["normal_feature", "skewed_feature"]:
    outlier_mask, lower, upper = detect_outliers_iqr(df[col])
    n_outliers = outlier_mask.sum()
    print(f"\n  {col}:")
    print(f"    Q1: {df[col].quantile(0.25):.2f}, Q3: {df[col].quantile(0.75):.2f}, "
          f"IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
    print(f"    Fences: [{lower:.2f}, {upper:.2f}]")
    print(f"    Outliers detected: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

# ============================================================
# Part 4: Visual Methods
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Normal feature
ax = axes[0, 0]
z_mask, z_scores = detect_outliers_zscore(df["normal_feature"])
iqr_mask, lower, upper = detect_outliers_iqr(df["normal_feature"])
ax.scatter(range(n), df["normal_feature"], c="steelblue", alpha=0.5, s=20, label="Normal")
ax.scatter(np.where(z_mask)[0], df.loc[z_mask, "normal_feature"],
           c="red", s=60, marker="x", linewidths=2, label="Z-score outlier")
ax.scatter(np.where(iqr_mask)[0], df.loc[iqr_mask, "normal_feature"],
           c="orange", s=80, marker="D", facecolors="none", linewidths=2, label="IQR outlier")
ax.set_title("Normal Feature — Outlier Detection", fontweight="bold")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.boxplot(df["normal_feature"], vert=True, patch_artist=True,
           boxprops=dict(facecolor="#3498db", alpha=0.6))
ax.set_title("Normal Feature — Box Plot", fontweight="bold")
ax.set_ylabel("Value")

ax = axes[0, 2]
ax.hist(df["normal_feature"], bins=40, color="#3498db", edgecolor="black", alpha=0.7)
ax.axvline(df["normal_feature"].mean() - 3 * df["normal_feature"].std(),
           color="red", linestyle="--", label="Z = -3")
ax.axvline(df["normal_feature"].mean() + 3 * df["normal_feature"].std(),
           color="red", linestyle="--", label="Z = +3")
ax.set_title("Normal Feature — Histogram", fontweight="bold")
ax.set_xlabel("Value")
ax.legend(fontsize=9)

# Row 2: Skewed feature
ax = axes[1, 0]
z_mask_s, _ = detect_outliers_zscore(df["skewed_feature"])
iqr_mask_s, lower_s, upper_s = detect_outliers_iqr(df["skewed_feature"])
ax.scatter(range(n), df["skewed_feature"], c="steelblue", alpha=0.5, s=20, label="Normal")
ax.scatter(np.where(z_mask_s)[0], df.loc[z_mask_s, "skewed_feature"],
           c="red", s=60, marker="x", linewidths=2, label="Z-score outlier")
ax.scatter(np.where(iqr_mask_s)[0], df.loc[iqr_mask_s, "skewed_feature"],
           c="orange", s=80, marker="D", facecolors="none", linewidths=2, label="IQR outlier")
ax.set_title("Skewed Feature — Outlier Detection", fontweight="bold")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.boxplot(df["skewed_feature"], vert=True, patch_artist=True,
           boxprops=dict(facecolor="#e74c3c", alpha=0.6))
ax.set_title("Skewed Feature — Box Plot", fontweight="bold")
ax.set_ylabel("Value")

ax = axes[1, 2]
ax.hist(df["skewed_feature"], bins=40, color="#e74c3c", edgecolor="black", alpha=0.7)
ax.axvline(df["skewed_feature"].mean() + 3 * df["skewed_feature"].std(),
           color="red", linestyle="--", label="Z = +3")
ax.set_title("Skewed Feature — Histogram", fontweight="bold")
ax.set_xlabel("Value")
ax.legend(fontsize=9)

plt.suptitle("Outlier Detection: Z-Score vs IQR", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# Part 5: Impact of Outlier Removal on Analysis
# ============================================================

print("\n" + "=" * 60)
print("IMPACT OF OUTLIER REMOVAL")
print("=" * 60)

# Combine Z-score and IQR outliers for normal_feature
combined_mask = z_mask | iqr_mask
df_clean = df[~combined_mask].copy()

print(f"\nOriginal dataset: {len(df)} rows")
print(f"After outlier removal: {len(df_clean)} rows ({combined_mask.sum()} removed)")

print(f"\n{'Statistic':<20} {'With Outliers':>15} {'Without Outliers':>18} {'Change':>10}")
print("-" * 65)
for stat_name, stat_fn in [("Mean", np.mean), ("Std", np.std), ("Median", np.median)]:
    before = stat_fn(df["normal_feature"])
    after = stat_fn(df_clean["normal_feature"])
    change = (after - before) / before * 100
    print(f"{stat_name:<20} {before:>15.2f} {after:>18.2f} {change:>9.1f}%")

# Scatter plot: before and after
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(df["normal_feature"], df["correlated"], alpha=0.5, s=20, c="steelblue")
ax.scatter(df.loc[combined_mask, "normal_feature"], df.loc[combined_mask, "correlated"],
           c="red", s=60, marker="x", linewidths=2, label="Outliers")
corr_before = df[["normal_feature", "correlated"]].corr().iloc[0, 1]
ax.set_title(f"Before Removal (r = {corr_before:.3f})", fontweight="bold")
ax.set_xlabel("Normal Feature")
ax.set_ylabel("Correlated Feature")
ax.legend()

ax = axes[1]
ax.scatter(df_clean["normal_feature"], df_clean["correlated"], alpha=0.5, s=20, c="#2ecc71")
corr_after = df_clean[["normal_feature", "correlated"]].corr().iloc[0, 1]
ax.set_title(f"After Removal (r = {corr_after:.3f})", fontweight="bold")
ax.set_xlabel("Normal Feature")
ax.set_ylabel("Correlated Feature")

plt.suptitle("Impact of Outlier Removal on Correlation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nCorrelation before removal: {corr_before:.3f}")
print(f"Correlation after removal:  {corr_after:.3f}")
print("\nKey insight: Outlier removal strengthened the correlation,")
print("revealing the true relationship that outliers were masking.")
print("But always verify that removed points are truly errors,")
print("not legitimate extreme observations worth keeping.")
```

---

## Key Takeaways

- **Outliers can be signal or noise** — the first step is always to determine whether an extreme value is a measurement error or a genuine observation before deciding how to handle it
- **Z-score works best for normal distributions** but is unreliable for skewed data because the mean and standard deviation are themselves distorted by outliers
- **IQR is more robust** because it relies on quartiles (resistant to extreme values), making it suitable for both normal and skewed distributions
- **Visual methods (box plots, scatter plots, histograms)** provide intuitive understanding that complements statistical methods — always visualize before deciding
- **Removing outliers changes your analysis**: It affects means, variances, correlations, and model fits. Document every outlier decision so your analysis is reproducible and auditable
