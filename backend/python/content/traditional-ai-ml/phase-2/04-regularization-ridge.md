# Regularization: Ridge Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.4

---

## Concept & Intuition

### What problem are we solving?

Ordinary Least Squares (OLS) finds the coefficients that minimize the sum of squared residuals. But OLS has a dark side: when features are correlated (multicollinearity), when there are many features relative to observations, or when polynomial features are used, OLS produces coefficients that are large, unstable, and overfit the training data. **Ridge regression** (L2 regularization) fixes this by adding a penalty for large coefficients to the loss function.

The Ridge objective is: minimize [Sum of Squared Residuals + alpha * Sum of Squared Coefficients]. The penalty term alpha * ||w||^2 discourages the model from assigning large values to any coefficient. When alpha = 0, Ridge is identical to OLS. As alpha increases, coefficients are pushed toward zero (but never exactly to zero), producing a simpler, more stable model that generalizes better.

This is the **bias-variance tradeoff** in action. By accepting a small amount of bias (the model does not fit the training data as closely), Ridge dramatically reduces variance (the model's sensitivity to the specific training data), leading to better performance on new data. The parameter alpha controls where you sit on this tradeoff — and finding the right alpha is the key challenge.

### Why naive approaches fail

OLS fails when the design matrix is ill-conditioned (features are highly correlated or nearly redundant). In this case, the matrix inverse that OLS requires becomes numerically unstable, producing coefficients that are enormous in magnitude and flip sign with tiny changes to the data. These wild coefficients might fit the training data well but are useless for prediction.

A naive fix — removing correlated features manually — is brittle and throws away information. Ridge regression is a principled alternative: it automatically shrinks the contributions of correlated features without removing any of them, stabilizing the model while preserving all the information in the data.

### Mental models

- **The elastic band analogy**: OLS lets coefficients go wherever they want. Ridge attaches an elastic band to each coefficient, pulling it toward zero. Stronger elastic (higher alpha) = smaller coefficients = simpler model
- **The regularization dial**: alpha is a dial that goes from "pure OLS" (alpha=0) to "all coefficients nearly zero" (alpha=infinity). Somewhere in between is the sweet spot
- **Shrinkage, not elimination**: Ridge shrinks ALL coefficients toward zero but never sets any exactly to zero. Every feature still contributes, just with reduced influence. (Lasso, which we will see next, can zero out coefficients)
- **The condition number**: Ridge adds alpha to the diagonal of X'X, improving its condition number and making the matrix inverse numerically stable. This is why it fixes multicollinearity

### Visual explanations

```
OLS vs Ridge: The Penalty
===========================

  OLS Loss:   L = sum( (y_i - y_pred_i)^2 )
                     ↑
               Fit the data

  Ridge Loss: L = sum( (y_i - y_pred_i)^2 ) + alpha * sum( w_j^2 )
                     ↑                              ↑
               Fit the data                   Keep weights small


The Effect of Alpha
=====================

  alpha = 0  (OLS):     Coefficients: [120, -85, 45, -200, 150]
                         -> Large, unstable, overfit

  alpha = 1:             Coefficients: [30, -20, 15, -40, 35]
                         -> Smaller, more stable

  alpha = 100:           Coefficients: [3, -2, 1.5, -4, 3.5]
                         -> Very small, underfit

  alpha = inf:           Coefficients: [0, 0, 0, 0, 0]
                         -> All zero, model predicts mean


Coefficient Paths (as alpha increases)
========================================

  Coeff
    |
  20|  \_____ Feature A
    |  _\____
  10|/  \___ Feature B
    |    \__
   0|------\------------ alpha -->
    |      \___ Feature C
 -10|
    |

  All coefficients shrink toward zero as alpha increases.
  They NEVER reach exactly zero (unlike Lasso).
```

---

## Hands-on Exploration

1. **Adjust alpha**: Use the slider to try different alpha values. Watch how coefficients shrink and how the train/test R-squared changes. Find the alpha that maximizes test performance.

2. **Coefficient paths**: Look at the coefficient path plot. Identify which features are most important (their coefficients shrink last) and which are least important (shrink first).

3. **Compare with OLS on collinear data**: The code includes highly correlated features. Compare OLS coefficient stability vs. Ridge. Notice how OLS coefficients are wild while Ridge coefficients are well-behaved.

4. **Cross-validation for alpha selection**: Use the cross-validation curve to find the optimal alpha. Compare it with your manual selection — cross-validation is usually more reliable than eyeballing.

---

## Live Code

```python
"""
Ridge Regression — L2 regularization, bias-variance tradeoff, and coefficient shrinkage.

Adjust alpha to see how regularization strength affects model behavior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# @param alpha float 0.01 100 1.0

# ============================================================
# Part 1: Generate Data with Multicollinearity
# ============================================================

np.random.seed(42)
n = 200
p = 10

# Create correlated features
X_base = np.random.randn(n, 3)
# Add correlated copies with noise
X = np.hstack([
    X_base,
    X_base[:, 0:1] + np.random.randn(n, 1) * 0.1,  # Nearly identical to feature 0
    X_base[:, 1:2] + np.random.randn(n, 1) * 0.2,   # Nearly identical to feature 1
    np.random.randn(n, 5),  # Independent features
])

feature_names = [f"x{i}" for i in range(X.shape[1])]

# True relationship uses only first 3 features
true_coefs = np.array([5, -3, 2, 0, 0, 0, 0, 0, 0, 0])
y = X @ true_coefs + np.random.randn(n) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("=" * 60)
print(f"RIDGE REGRESSION (alpha = {alpha})")
print("=" * 60)

# ============================================================
# Part 2: Fit OLS and Ridge
# ============================================================

# OLS (no regularization)
ols = LinearRegression()
ols.fit(X_train_s, y_train)
ols_train_r2 = ols.score(X_train_s, y_train)
ols_test_r2 = ols.score(X_test_s, y_test)

# Ridge with selected alpha
ridge = Ridge(alpha=alpha)
ridge.fit(X_train_s, y_train)
ridge_train_r2 = ridge.score(X_train_s, y_train)
ridge_test_r2 = ridge.score(X_test_s, y_test)

print(f"\n--- OLS (no regularization) ---")
print(f"  Train R²: {ols_train_r2:.4f}  |  Test R²: {ols_test_r2:.4f}")
print(f"  Coefficients: {np.round(ols.coef_, 2)}")
print(f"  Coeff L2 norm: {np.sqrt(np.sum(ols.coef_**2)):.2f}")

print(f"\n--- Ridge (alpha = {alpha}) ---")
print(f"  Train R²: {ridge_train_r2:.4f}  |  Test R²: {ridge_test_r2:.4f}")
print(f"  Coefficients: {np.round(ridge.coef_, 2)}")
print(f"  Coeff L2 norm: {np.sqrt(np.sum(ridge.coef_**2)):.2f}")

# ============================================================
# Part 3: Coefficient Paths Across Alpha Values
# ============================================================

alphas = np.logspace(-3, 4, 100)
coef_paths = []
train_r2s = []
test_r2s = []

for a in alphas:
    r = Ridge(alpha=a)
    r.fit(X_train_s, y_train)
    coef_paths.append(r.coef_.copy())
    train_r2s.append(r.score(X_train_s, y_train))
    test_r2s.append(r.score(X_test_s, y_test))

coef_paths = np.array(coef_paths)

# Cross-validation to find best alpha
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_s, y_train)
best_alpha = ridge_cv.alpha_
print(f"\n  Best alpha (CV): {best_alpha:.4f}")
print(f"  Best CV Test R²: {ridge_cv.score(X_test_s, y_test):.4f}")

# ============================================================
# Part 4: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Coefficient paths
ax = axes[0, 0]
for i in range(X.shape[1]):
    ax.semilogx(alphas, coef_paths[:, i], linewidth=1.5, label=feature_names[i])
ax.axvline(x=alpha, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"alpha={alpha}")
ax.axvline(x=best_alpha, color="green", linestyle=":", linewidth=2, alpha=0.7,
           label=f"Best CV alpha={best_alpha:.2f}")
ax.set_xlabel("Alpha (log scale)", fontsize=11)
ax.set_ylabel("Coefficient Value", fontsize=11)
ax.set_title("Ridge Coefficient Paths", fontsize=13, fontweight="bold")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# R-squared vs alpha
ax = axes[0, 1]
ax.semilogx(alphas, train_r2s, "b-", linewidth=2, label="Train R²")
ax.semilogx(alphas, test_r2s, "r-", linewidth=2, label="Test R²")
ax.axvline(x=alpha, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"Selected alpha={alpha}")
ax.axvline(x=best_alpha, color="green", linestyle=":", linewidth=2, alpha=0.7,
           label=f"Best CV alpha={best_alpha:.2f}")
ax.set_xlabel("Alpha (log scale)", fontsize=11)
ax.set_ylabel("R-squared", fontsize=11)
ax.set_title("R² vs Regularization Strength", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# OLS vs Ridge coefficients
ax = axes[1, 0]
x_pos = np.arange(len(feature_names))
width = 0.35
ax.bar(x_pos - width / 2, ols.coef_, width, color="#e74c3c", alpha=0.7,
       edgecolor="black", label="OLS")
ax.bar(x_pos + width / 2, ridge.coef_, width, color="#3498db", alpha=0.7,
       edgecolor="black", label=f"Ridge (alpha={alpha})")
ax.bar(x_pos, true_coefs, width * 0.3, color="green", alpha=0.5,
       edgecolor="green", label="True coefficients")
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel("Coefficient Value")
ax.set_title("OLS vs Ridge Coefficients", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Coefficient L2 norm vs alpha
ax = axes[1, 1]
l2_norms = np.sqrt(np.sum(coef_paths ** 2, axis=1))
ax.semilogx(alphas, l2_norms, "b-", linewidth=2)
ax.axvline(x=alpha, color="red", linestyle="--", linewidth=2, alpha=0.7,
           label=f"Selected alpha={alpha}")
ax.axhline(y=np.sqrt(np.sum(ols.coef_ ** 2)), color="gray", linestyle=":",
           label=f"OLS L2 norm = {np.sqrt(np.sum(ols.coef_**2)):.2f}")
ax.set_xlabel("Alpha (log scale)", fontsize=11)
ax.set_ylabel("||w||₂ (L2 norm of coefficients)", fontsize=11)
ax.set_title("Coefficient Shrinkage", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Part 5: Stability Comparison
# ============================================================

print("\n" + "=" * 60)
print("STABILITY COMPARISON: OLS vs Ridge on Perturbed Data")
print("=" * 60)

ols_coefs_list = []
ridge_coefs_list = []

for trial in range(20):
    # Perturb training data slightly
    noise = np.random.randn(*X_train_s.shape) * 0.1
    X_perturbed = X_train_s + noise

    ols_temp = LinearRegression().fit(X_perturbed, y_train)
    ridge_temp = Ridge(alpha=best_alpha).fit(X_perturbed, y_train)

    ols_coefs_list.append(ols_temp.coef_)
    ridge_coefs_list.append(ridge_temp.coef_)

ols_std = np.std(ols_coefs_list, axis=0)
ridge_std = np.std(ridge_coefs_list, axis=0)

print(f"\n  {'Feature':<10} {'OLS Std':>10} {'Ridge Std':>10} {'Stability Gain':>15}")
print(f"  {'-' * 50}")
for i, feat in enumerate(feature_names):
    gain = ols_std[i] / ridge_std[i] if ridge_std[i] > 0 else float("inf")
    print(f"  {feat:<10} {ols_std[i]:>10.3f} {ridge_std[i]:>10.3f} {gain:>14.1f}x")

print(f"\nKey insight: Ridge regression stabilizes coefficients by penalizing")
print(f"large values. The alpha parameter controls the strength of this penalty.")
print(f"Use cross-validation to find the optimal alpha value.")
```

---

## Key Takeaways

- **Ridge regression adds an L2 penalty** (sum of squared coefficients) to the OLS loss function, discouraging large coefficients and reducing overfitting
- **The alpha parameter controls regularization strength**: alpha=0 is OLS, larger alpha means more shrinkage. Use cross-validation (RidgeCV) to find the optimal value
- **Ridge fixes multicollinearity**: By shrinking correlated feature coefficients, Ridge produces stable, interpretable models even when features are highly correlated
- **Shrinkage trades bias for variance**: Ridge introduces a small bias (coefficients are systematically smaller than OLS) but dramatically reduces variance (coefficients are more stable across datasets)
- **Ridge never zeros out coefficients**: All features remain in the model with reduced influence. If you need actual feature elimination, use Lasso (L1 regularization) instead
