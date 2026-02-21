# Feature Scaling

> Phase 1 — Data Wrangling | Kata 1.5

---

## Concept & Intuition

### What problem are we solving?

Machine learning features often live on wildly different scales. Age might range from 0 to 100, income from 10,000 to 10,000,000, and a boolean flag is either 0 or 1. When you feed these unscaled features to distance-based algorithms (k-NN, SVM, k-means) or gradient-based algorithms (linear regression, neural networks), the features with larger scales dominate the computation. An income difference of 10,000 dwarfs an age difference of 10, even though both might be equally important for prediction.

**Feature scaling** transforms all features to comparable ranges so that no single feature dominates due to its scale. This is not about changing the information content — it is about making the information accessible to algorithms that are sensitive to magnitude. The three most common scalers are **StandardScaler** (zero mean, unit variance), **MinMaxScaler** (scales to [0, 1]), and **RobustScaler** (uses median and IQR, resistant to outliers).

The choice of scaler matters. StandardScaler assumes roughly normal distributions and is sensitive to outliers. MinMaxScaler squashes everything to [0, 1] but is destroyed by even a single extreme outlier. RobustScaler uses the median and interquartile range, making it the best choice when outliers are present. Understanding these tradeoffs lets you match the scaler to your data characteristics.

### Why naive approaches fail

The most common mistake is not scaling at all. Many beginners skip scaling and wonder why their k-NN model performs poorly or why their gradient descent converges slowly. Without scaling, a feature measured in dollars ($0 to $1,000,000) will completely overpower a feature measured in years (0 to 50) in any distance computation.

An equally dangerous mistake is scaling before splitting into train and test sets. If you compute the mean and standard deviation using the entire dataset (including test data) before scaling, you leak information from the test set into the training process. This is called **data leakage** and produces optimistically biased performance estimates. The correct approach is to fit the scaler on training data only, then apply the same transformation to the test data.

### Mental models

- **The unit conversion analogy**: Scaling is like converting all measurements to the same unit system. You would not add 5 miles to 3 kilometers without converting first — similarly, you should not combine features on different scales without scaling
- **Fit on train, transform both**: Think of the scaler as a ruler calibrated on training data. You use the same ruler (same mean, same std) to measure test data. You never recalibrate the ruler using test data
- **The outlier vulnerability**: MinMaxScaler is like stretching a rubber band between the min and max values. One outlier at 1,000,000 stretches the band so far that all normal values (0-100) are squashed into a tiny region
- **Scaling preserves relationships**: Scaling changes the numbers but not the relative ordering or the relationships between features. A positive correlation before scaling is still positive after scaling

### Visual explanations

```
Why Scaling Matters for Distance-Based Models
================================================

  Unscaled: Age (0-100), Income (0-1,000,000)

  Person A: age=30, income=50,000
  Person B: age=31, income=50,100
  Person C: age=30, income=60,000

  Distance(A,B) = sqrt(1^2 + 100^2) = 100.005
  Distance(A,C) = sqrt(0^2 + 10000^2) = 10,000

  Age difference is INVISIBLE because income dominates!

  After StandardScaler:
  Person A: age=-0.5, income=-0.3
  Person B: age=-0.4, income=-0.29
  Person C: age=-0.5, income=0.1

  Now both features contribute meaningfully to distance.


Scaler Comparison
==================

  Original:  [10, 20, 30, 40, 50, 1000]  (outlier!)

  StandardScaler:  [-0.45, -0.40, -0.36, -0.31, -0.27, 1.79]
    Mean=0, Std=1. Outlier stretches the scale.

  MinMaxScaler:    [0.00, 0.01, 0.02, 0.03, 0.04, 1.00]
    Squashed to [0,1]. Normal values are COMPRESSED!

  RobustScaler:    [-0.67, -0.33, 0.00, 0.33, 0.67, 32.33]
    Median=0, IQR=1. Normal values well-spread. Outlier is just... big.
```

---

## Hands-on Exploration

1. **Visualize the effect**: Run the code below to see how each scaler transforms the same dataset. Pay special attention to the distribution shapes before and after scaling.

2. **Test with outliers**: The dataset includes outlier contamination. Compare how StandardScaler, MinMaxScaler, and RobustScaler handle it. Which scaler preserves the distribution shape best?

3. **K-NN with and without scaling**: Compare k-NN accuracy on scaled vs. unscaled data. This demonstrates why scaling is mandatory for distance-based models.

4. **Data leakage experiment**: Try scaling before and after the train-test split. Compare the reported test scores — the "wrong" approach (scaling before split) should produce slightly inflated scores.

---

## Live Code

```python
"""
Feature Scaling — StandardScaler, MinMaxScaler, and RobustScaler compared.

This code demonstrates why scaling matters, how different scalers behave,
and the critical importance of fitting on training data only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification

# ============================================================
# Part 1: Create Data with Different Scales and Outliers
# ============================================================

np.random.seed(42)
n = 500

# Features on very different scales
age = np.random.normal(40, 12, n)  # Range: ~10-70
income = np.random.lognormal(10.5, 0.6, n)  # Range: ~10K-200K
score = np.random.uniform(0, 1, n)  # Range: 0-1

# Add some outliers
age[0:3] = [150, -20, 200]
income[3:6] = [5000000, 4000000, 6000000]

df = pd.DataFrame({"age": age, "income": income, "score": score})

print("=" * 60)
print("FEATURE SCALING — Why It Matters")
print("=" * 60)

print("\n--- Feature Ranges (Before Scaling) ---")
for col in df.columns:
    print(f"  {col:>10}: min={df[col].min():>12.1f}  max={df[col].max():>12.1f}  "
          f"mean={df[col].mean():>12.1f}  std={df[col].std():>12.1f}")

# ============================================================
# Part 2: Apply Three Scalers
# ============================================================

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}

scaled_dfs = {}
for name, scaler in scalers.items():
    scaled_data = scaler.fit_transform(df)
    scaled_dfs[name] = pd.DataFrame(scaled_data, columns=df.columns)

print("\n--- Feature Ranges After Scaling ---")
for scaler_name, sdf in scaled_dfs.items():
    print(f"\n  {scaler_name}:")
    for col in sdf.columns:
        print(f"    {col:>10}: min={sdf[col].min():>8.2f}  max={sdf[col].max():>8.2f}  "
              f"mean={sdf[col].mean():>8.2f}  std={sdf[col].std():>8.2f}")

# ============================================================
# Part 3: Visualization
# ============================================================

fig, axes = plt.subplots(4, 3, figsize=(15, 16))

datasets = {"Original": df, **scaled_dfs}
colors = ["#3498db", "#2ecc71", "#e74c3c"]

for row, (ds_name, ds) in enumerate(datasets.items()):
    for col_idx, col in enumerate(df.columns):
        ax = axes[row, col_idx]
        ax.hist(ds[col], bins=40, color=colors[col_idx], edgecolor="black", alpha=0.7)
        ax.set_title(f"{ds_name}\n{col}", fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("Count")
        # Add stats annotation
        ax.text(0.95, 0.95, f"$\\mu$={ds[col].mean():.2f}\n$\\sigma$={ds[col].std():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.suptitle("Feature Distributions: Original vs Scaled", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# Part 4: Impact on K-NN Classification
# ============================================================

print("\n" + "=" * 60)
print("IMPACT ON K-NN CLASSIFICATION")
print("=" * 60)

# Create a classification dataset with features on different scales
X, y = make_classification(n_samples=500, n_features=5, n_informative=3,
                            n_redundant=1, random_state=42)

# Artificially scale features to different ranges
X[:, 0] *= 1          # Feature 0: scale ~1
X[:, 1] *= 100        # Feature 1: scale ~100
X[:, 2] *= 10000      # Feature 2: scale ~10,000
X[:, 3] *= 0.01       # Feature 3: scale ~0.01
X[:, 4] *= 1000000    # Feature 4: scale ~1,000,000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nFeature scale ranges in training data:")
for i in range(5):
    print(f"  Feature {i}: [{X_train[:, i].min():.1f}, {X_train[:, i].max():.1f}]")

knn = KNeighborsClassifier(n_neighbors=5)

# Without scaling
knn.fit(X_train, y_train)
score_unscaled = knn.score(X_test, y_test)
print(f"\nK-NN accuracy WITHOUT scaling: {score_unscaled:.4f}")

# With each scaler (correct: fit on train, transform both)
for name, scaler in scalers.items():
    scaler_inst = type(scaler)()  # Fresh instance
    X_train_scaled = scaler_inst.fit_transform(X_train)
    X_test_scaled = scaler_inst.transform(X_test)  # Use train's parameters!
    knn.fit(X_train_scaled, y_train)
    score_scaled = knn.score(X_test_scaled, y_test)
    print(f"K-NN accuracy with {name}: {score_scaled:.4f}")

# ============================================================
# Part 5: Data Leakage Demonstration
# ============================================================

print("\n" + "=" * 60)
print("DATA LEAKAGE: Scaling Before vs After Split")
print("=" * 60)

# WRONG: Scale on entire dataset before splitting
scaler_wrong = StandardScaler()
X_all_scaled = scaler_wrong.fit_transform(X)
X_train_wrong, X_test_wrong, y_train_w, y_test_w = train_test_split(
    X_all_scaled, y, test_size=0.2, random_state=42
)
knn.fit(X_train_wrong, y_train_w)
score_wrong = knn.score(X_test_wrong, y_test_w)

# CORRECT: Scale after splitting (fit on train only)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler_correct = StandardScaler()
X_train_correct = scaler_correct.fit_transform(X_train_c)
X_test_correct = scaler_correct.transform(X_test_c)
knn.fit(X_train_correct, y_train_c)
score_correct = knn.score(X_test_correct, y_test_c)

print(f"\n  WRONG (scale before split):  {score_wrong:.4f}  <- Optimistically biased!")
print(f"  CORRECT (scale after split): {score_correct:.4f}  <- Honest estimate")
print(f"  Difference: {score_wrong - score_correct:+.4f}")

# ============================================================
# Part 6: Summary Chart
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

methods = ["No Scaling", "StandardScaler", "MinMaxScaler", "RobustScaler"]
# Re-compute for the chart
scores_list = [score_unscaled]
for scaler_class in [StandardScaler, MinMaxScaler, RobustScaler]:
    sc = scaler_class()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    knn.fit(Xtr, y_train)
    scores_list.append(knn.score(Xte, y_test))

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
bars = ax.bar(methods, scores_list, color=colors, edgecolor="black", alpha=0.8)
ax.set_ylabel("K-NN Accuracy", fontsize=12)
ax.set_title("Impact of Feature Scaling on K-NN Performance", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.1)
for bar, score in zip(bars, scores_list):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{score:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nKey insight: Feature scaling is not optional for distance-based and")
print("gradient-based algorithms. Always fit the scaler on training data only,")
print("then apply the same transformation to test data.")
```

---

## Key Takeaways

- **Feature scaling is mandatory** for distance-based (k-NN, SVM, k-means) and gradient-based (linear regression, neural networks) algorithms — features on larger scales will dominate otherwise
- **StandardScaler** (mean=0, std=1) is the most common choice but is sensitive to outliers because the mean and standard deviation are affected by extreme values
- **MinMaxScaler** (range [0,1]) is useful when you need bounded outputs but is severely distorted by even a single outlier
- **RobustScaler** (median=0, IQR=1) is the best choice when outliers are present because it uses statistics that are resistant to extreme values
- **Always fit on training data only**: Computing scaling parameters on the full dataset before splitting introduces data leakage, producing optimistically biased performance estimates
