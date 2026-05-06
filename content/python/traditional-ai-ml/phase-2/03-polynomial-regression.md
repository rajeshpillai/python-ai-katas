# Polynomial Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.3

---

## Concept & Intuition

### What problem are we solving?

The real world is rarely linear. The relationship between study time and exam scores, drug dosage and effectiveness, or temperature and ice cream sales often follows curves, not straight lines. **Polynomial regression** extends linear regression to capture these nonlinear relationships by adding polynomial features — x^2, x^3, and higher powers — to the model. The model y = b0 + b1*x + b2*x^2 + b3*x^3 is still "linear" in its parameters (the b's), even though it fits a curve through the data.

The key decision is the **degree** of the polynomial. Degree 1 gives a straight line, degree 2 gives a parabola, degree 3 gives an S-curve, and so on. Higher degrees can fit more complex patterns, but they also introduce a critical risk: **overfitting**. A degree-20 polynomial can pass through every data point perfectly but will make absurd predictions between and beyond the data points. This is the bias-variance tradeoff in its purest form.

Polynomial regression beautifully illustrates the tension between underfitting (too simple, missing the pattern) and overfitting (too complex, fitting the noise). Learning to choose the right degree — complex enough to capture the signal, simple enough to generalize — is one of the most important skills in machine learning.

### Why naive approaches fail

The naive approach of choosing a very high polynomial degree to maximize training accuracy is a classic overfitting trap. A degree-n polynomial with n+1 parameters can perfectly fit n+1 data points, achieving zero training error. But this "perfect" model oscillates wildly between data points and makes catastrophic predictions on new data. Training accuracy is a lie when the model has memorized rather than learned.

Another pitfall is failing to scale features before polynomial expansion. If x ranges from 0 to 1000, then x^5 ranges from 0 to 10^15 — a numerical range that causes floating-point overflow, numerical instability, and optimization failure. Always scale features before generating polynomial terms.

### Mental models

- **The telescope analogy**: Low-degree polynomials are like looking at a landscape from far away — you see the broad shape but miss details. High-degree polynomials are like using a microscope — you see every bump and wrinkle, including noise you should ignore
- **The bias-variance seesaw**: As polynomial degree increases, bias decreases (the model can fit more complex patterns) but variance increases (the model becomes more sensitive to specific data points). The sweet spot minimizes total error
- **Polynomial features are not magic**: Adding x^2 to a linear model is equivalent to adding a new feature that happens to be the square of the original. The model is still linear in its parameters — you are just giving it more raw material to work with
- **Extrapolation disaster**: Polynomials are particularly dangerous for extrapolation. Even a well-fit polynomial can explode or collapse just outside the range of training data

### Visual explanations

```
Polynomial Degree and Fit Quality
====================================

Degree 1 (Underfit):         Degree 3 (Good Fit):         Degree 15 (Overfit):
   y                           y                            y
   |                           |     __                     |  /\/\  __/\
   |     *  *                  |   _/  \  *                 | / *  \/  * \
   |  *  /  *                  | _/ *   \                   |/ * *    *   \
   | * /     *                 |/ *      \*                 |*    *  *    *
   |* / *                      |*    *    \                 |  *   *    *
   |/  *                       | *        \*                | *         *
   +────────── x              +────────── x               +────────── x

   High bias                  Balanced                    High variance
   Low variance               bias & variance             Low bias
   R² = 0.5                  R² = 0.9                    R² = 0.99 (train)
                                                          R² = 0.3 (test!)


The Bias-Variance Tradeoff
============================

  Error
  |
  | \                              ___--- Total error
  |  \                      ___---
  |   \               ___---
  |    \         __---
  |     \___---          ______--- Variance
  |     ___---    ______-
  |____----------
  |                                Bias
  +──────────────────────────── Degree
    1    2    3    4    5   ...  15

  Sweet spot: minimum total error
```

---

## Hands-on Exploration

1. **Vary the degree**: Use the slider annotation to try polynomial degrees from 1 to 10. Watch how the curve fits the data more closely at first, then starts oscillating wildly at high degrees.

2. **Train vs test gap**: Observe how training R-squared always increases with degree, but test R-squared increases, peaks, then decreases. The peak is your optimal degree.

3. **Try extrapolation**: Look at the model's predictions beyond the training data range. Notice how high-degree polynomials make absurd predictions outside the data — this is why polynomials are dangerous for extrapolation.

4. **The sample size effect**: Try reducing the dataset size from 100 to 30. Notice how overfitting becomes worse with less data — there are fewer points to constrain the polynomial.

---

## Live Code

```python
"""
Polynomial Regression — Nonlinear fits, degree selection, and overfitting.

Adjust the degree parameter to see how polynomial complexity affects
the fit quality, bias-variance tradeoff, and generalization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

# @param degree int 1 10 2

# ============================================================
# Part 1: Generate Nonlinear Data
# ============================================================

np.random.seed(42)
n = 100

x = np.sort(np.random.uniform(0, 10, n))
# True relationship: a cubic with noise
y_true = 0.5 * (x - 2) * (x - 5) * (x - 8) + 50
y = y_true + np.random.normal(0, 10, n)

X = x.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 60)
print(f"POLYNOMIAL REGRESSION (degree = {degree})")
print("=" * 60)

# ============================================================
# Part 2: Fit Polynomial Model with Selected Degree
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n  Degree: {degree}")
print(f"  Number of features: {degree} (x, x², ..., x^{degree})")
print(f"  Train R²: {r2_train:.4f}")
print(f"  Test R²:  {r2_test:.4f}")
print(f"  Test RMSE: {rmse_test:.2f}")
print(f"  Overfit gap: {r2_train - r2_test:.4f}")

# ============================================================
# Part 3: Compare All Degrees
# ============================================================

degrees = range(1, 11)
train_r2s = []
test_r2s = []
cv_r2s = []

for d in degrees:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    train_r2s.append(r2_score(y_train, pipe.predict(X_train)))
    test_r2s.append(r2_score(y_test, pipe.predict(X_test)))
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
    cv_r2s.append(cv.mean())

best_degree_cv = list(degrees)[np.argmax(cv_r2s)]
print(f"\n  Best degree by cross-validation: {best_degree_cv}")

# ============================================================
# Part 4: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Polynomial fit for selected degree
ax = axes[0, 0]
x_smooth = np.linspace(-1, 11, 300).reshape(-1, 1)
y_smooth = pipeline.predict(x_smooth)

ax.scatter(X_train, y_train, alpha=0.6, s=30, c="#3498db", label="Train", edgecolors="black", linewidth=0.5)
ax.scatter(X_test, y_test, alpha=0.6, s=30, c="#e74c3c", marker="s", label="Test", edgecolors="black", linewidth=0.5)
ax.plot(x_smooth, y_smooth, "g-", linewidth=2, label=f"Degree {degree} fit")
ax.plot(np.sort(x), 0.5 * (np.sort(x) - 2) * (np.sort(x) - 5) * (np.sort(x) - 8) + 50,
        "k--", linewidth=1.5, alpha=0.5, label="True function")
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(y.min() - 20, y.max() + 20)
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title(f"Polynomial Degree {degree} Fit\n(Train R²={r2_train:.3f}, Test R²={r2_test:.3f})",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Compare multiple degrees
ax = axes[0, 1]
x_sm = np.linspace(-0.5, 10.5, 300).reshape(-1, 1)
ax.scatter(X_train, y_train, alpha=0.3, s=15, c="gray")
for d_show in [1, 3, best_degree_cv, 10]:
    pipe_show = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=d_show, include_bias=False)),
        ("model", LinearRegression()),
    ])
    pipe_show.fit(X_train, y_train)
    y_sm = pipe_show.predict(x_sm)
    ax.plot(x_sm, y_sm, linewidth=2, label=f"Degree {d_show}")
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(y.min() - 30, y.max() + 30)
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("Multiple Polynomial Fits Compared", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bias-variance tradeoff
ax = axes[1, 0]
ax.plot(list(degrees), train_r2s, "bo-", linewidth=2, markersize=8, label="Train R²")
ax.plot(list(degrees), test_r2s, "ro-", linewidth=2, markersize=8, label="Test R²")
ax.plot(list(degrees), cv_r2s, "gs--", linewidth=2, markersize=8, label="CV R² (mean)")
ax.axvline(x=degree, color="purple", linestyle=":", linewidth=2, alpha=0.7, label=f"Selected (d={degree})")
ax.axvline(x=best_degree_cv, color="green", linestyle=":", linewidth=2, alpha=0.7,
           label=f"Best CV (d={best_degree_cv})")
ax.set_xlabel("Polynomial Degree", fontsize=11)
ax.set_ylabel("R-squared", fontsize=11)
ax.set_title("Bias-Variance Tradeoff", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(list(degrees))

# Extrapolation danger
ax = axes[1, 1]
x_extrap = np.linspace(-3, 14, 300).reshape(-1, 1)
for d_show in [3, 7, 10]:
    pipe_show = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=d_show, include_bias=False)),
        ("model", LinearRegression()),
    ])
    pipe_show.fit(X_train, y_train)
    y_extrap = pipe_show.predict(x_extrap)
    ax.plot(x_extrap, y_extrap, linewidth=2, label=f"Degree {d_show}")

ax.scatter(X_train, y_train, alpha=0.3, s=15, c="gray", zorder=5)
ax.axvspan(-3, 0, alpha=0.1, color="red", label="Extrapolation zone")
ax.axvspan(10, 14, alpha=0.1, color="red")
ax.set_xlim(-3, 14)
ax.set_ylim(-200, 200)
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("Extrapolation Danger\n(red zones = outside training range)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table
print("\n--- Degree Comparison Table ---")
print(f"  {'Degree':>6} {'Train R²':>10} {'Test R²':>10} {'CV R²':>10} {'Gap':>10}")
print(f"  {'-' * 50}")
for d, tr, te, cv in zip(degrees, train_r2s, test_r2s, cv_r2s):
    marker = " <-- selected" if d == degree else (" <-- best CV" if d == best_degree_cv else "")
    print(f"  {d:>6} {tr:>10.4f} {te:>10.4f} {cv:>10.4f} {tr - te:>10.4f}{marker}")

print(f"\nKey insight: Polynomial degree controls the bias-variance tradeoff.")
print(f"Use cross-validation (not training R²) to select the optimal degree.")
print(f"Higher degree = more flexible but greater overfitting risk.")
```

---

## Key Takeaways

- **Polynomial regression captures nonlinear relationships** by adding powers of x as features, while remaining a linear model in its parameters (the coefficients)
- **Degree selection is critical**: Too low (underfitting) misses the pattern; too high (overfitting) fits noise. Cross-validation is the reliable way to choose
- **Training R-squared always increases with degree** but test R-squared peaks and then declines — the gap between them is the overfitting signal
- **Polynomials are dangerous for extrapolation**: Even a well-fit polynomial can make absurd predictions outside the training data range, as high-degree terms dominate
- **Always scale features before polynomial expansion**: Without scaling, high-degree terms (x^5, x^10) create numerical overflow and instability
