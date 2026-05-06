# Simple Linear Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.1

---

## Concept & Intuition

### What problem are we solving?

Simple linear regression is the most fundamental predictive model in statistics and machine learning. Given a single input feature x, it finds the best straight line y = mx + b that predicts a continuous target variable y. "Best" means the line that minimizes the total squared distance between the predicted values and the actual values — this is the **Ordinary Least Squares (OLS)** criterion.

Despite its simplicity, linear regression is remarkably powerful and widely used. It forms the basis of nearly every more advanced regression technique (multiple regression, polynomial regression, regularized models). Understanding it deeply — what it assumes, when it works, and why it fails — is essential to building intuition for all of supervised learning.

The key quantities to understand are the **slope** (m, how much y changes per unit change in x), the **intercept** (b, the value of y when x=0), the **residuals** (the differences between actual and predicted values), and **R-squared** (the fraction of variance in y that is explained by the model, ranging from 0 to 1).

### Why naive approaches fail

The most common mistake is assuming that a low R-squared means the model is useless, or that a high R-squared means the model is good. R-squared only measures linear fit — a model can have R-squared = 0 for data with a perfect nonlinear relationship (like a parabola). Conversely, R-squared can be artificially inflated by outliers or by overfitting.

Another pitfall is ignoring the residuals. OLS assumes that residuals are normally distributed, have constant variance (homoscedasticity), and are independent. Violating these assumptions does not prevent the model from running, but it makes the results unreliable — confidence intervals become wrong, and the model may be systematically biased in certain regions of the feature space.

### Mental models

- **The rubber band analogy**: OLS finds the line that minimizes the total stretch of rubber bands connecting each data point vertically to the line. Squaring the distances means longer rubber bands are penalized disproportionately
- **R-squared as explained variance**: If R-squared = 0.75, the model explains 75% of why y varies. The remaining 25% is due to factors not captured by x (noise, missing features, nonlinearity)
- **Slope as sensitivity**: The slope tells you "if I increase x by 1 unit, y changes by m units on average." This is the key insight for interpretation
- **Residuals as the model's confession**: Residuals tell you where and how the model fails. Patterns in residuals (like a U-shape) reveal systematic errors that summary statistics miss

### Visual explanations

```
Ordinary Least Squares (OLS)
==============================

  y
  |          *
  |        /  *
  |      / *
  |    * /         * = actual data point
  |  / *           / = regression line
  | /  *           | = residual (error)
  |/*______________
  +─────────────── x

  Goal: Minimize Sum of Squared Residuals
        = sum( (actual_i - predicted_i)^2 )


R-squared Interpretation
==========================

  R^2 = 1 - (SS_res / SS_tot)

  SS_tot = sum( (y_i - y_mean)^2 )   <- total variance
  SS_res = sum( (y_i - y_pred_i)^2 ) <- unexplained variance

  R^2 = 0.0  ->  Model no better than predicting the mean
  R^2 = 0.5  ->  Model explains half the variance
  R^2 = 1.0  ->  Perfect fit (suspicious — check for overfitting!)


Residual Patterns
==================

  Good (random scatter):     Bad (pattern = nonlinearity):

  res |  * *   *            res |  *     *
      |*  *  * *   *            |   *       *
  ----+-------*-----        ----+-----*--------
      | *  * *  *               |      *
      |*   *     *              |  *       *
      +──────────── x          +──────────── x

  Random = assumptions OK       Pattern = model is wrong!
```

---

## Hands-on Exploration

1. **Fit and interpret**: Run the code below to fit a simple linear regression. Identify the slope, intercept, and R-squared. Write a sentence interpreting the slope in plain English.

2. **Examine the residuals**: Look at the residual plot. Are residuals randomly scattered around zero? Is there a visible pattern? What would a pattern tell you?

3. **Add noise**: Modify the noise level in the data generation and re-fit. Observe how R-squared decreases as noise increases, while the estimated slope stays close to the true value.

4. **Break the model**: Create data with a nonlinear relationship (e.g., y = x^2) and fit a linear model. Observe the poor R-squared and the telltale U-shaped residual pattern.

---

## Live Code

```python
"""
Simple Linear Regression — OLS, residuals, and R-squared.

This code fits a simple linear regression, visualizes the fit and residuals,
and demonstrates how to interpret the key quantities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# Part 1: Generate Data with Known Relationship
# ============================================================

np.random.seed(42)
n = 150

# True relationship: y = 3x + 10 + noise
true_slope = 3.0
true_intercept = 10.0
noise_std = 8.0

x = np.random.uniform(0, 20, n)
y = true_slope * x + true_intercept + np.random.normal(0, noise_std, n)

X = x.reshape(-1, 1)

print("=" * 60)
print("SIMPLE LINEAR REGRESSION")
print("=" * 60)

# ============================================================
# Part 2: Fit the Model
# ============================================================

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
residuals = y - y_pred

print(f"\n--- True Parameters ---")
print(f"  True slope:     {true_slope}")
print(f"  True intercept: {true_intercept}")

print(f"\n--- Estimated Parameters ---")
print(f"  Estimated slope:     {model.coef_[0]:.4f}")
print(f"  Estimated intercept: {model.intercept_:.4f}")

print(f"\n--- Model Performance ---")
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print(f"  R-squared: {r2:.4f}")
print(f"  MSE:       {mse:.2f}")
print(f"  RMSE:      {rmse:.2f}")

print(f"\n--- Interpretation ---")
print(f"  For each 1-unit increase in x, y increases by {model.coef_[0]:.2f} units on average.")
print(f"  When x = 0, the predicted y is {model.intercept_:.2f}.")
print(f"  The model explains {r2:.1%} of the variance in y.")

# ============================================================
# Part 3: OLS Derivation Visualization
# ============================================================

print(f"\n--- Residual Statistics ---")
print(f"  Mean residual:    {residuals.mean():.6f} (should be ~0)")
print(f"  Std of residuals: {residuals.std():.2f}")
print(f"  Min residual:     {residuals.min():.2f}")
print(f"  Max residual:     {residuals.max():.2f}")

# ============================================================
# Part 4: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter plot with regression line
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.6, s=30, c="#3498db", edgecolors="black", linewidth=0.5)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = model.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, "r-", linewidth=2, label=f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
ax.plot(x_line, true_slope * x_line + true_intercept, "g--", linewidth=2, alpha=0.7,
        label=f"True: y = {true_slope}x + {true_intercept}")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(f"Simple Linear Regression (R² = {r2:.3f})", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Residuals vs predicted
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.6, s=30, c="#e74c3c", edgecolors="black", linewidth=0.5)
ax.axhline(y=0, color="black", linewidth=1)
ax.set_xlabel("Predicted y", fontsize=12)
ax.set_ylabel("Residual (actual - predicted)", fontsize=12)
ax.set_title("Residual Plot (should be random)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# Residual histogram
ax = axes[1, 0]
ax.hist(residuals, bins=25, color="#2ecc71", edgecolor="black", alpha=0.7, density=True)
# Overlay normal curve
from scipy import stats
xr = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()), "r-", linewidth=2,
        label="Normal fit")
ax.set_xlabel("Residual", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Residual Distribution (should be normal)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# R-squared decomposition
ax = axes[1, 1]
ss_tot = np.sum((y - y.mean()) ** 2)
ss_res = np.sum(residuals ** 2)
ss_reg = ss_tot - ss_res

labels = [f"Explained\n(R² = {r2:.3f})", f"Unexplained\n(1 - R² = {1 - r2:.3f})"]
sizes = [ss_reg, ss_res]
colors = ["#2ecc71", "#e74c3c"]
ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
       textprops={"fontsize": 11})
ax.set_title("Variance Decomposition", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.show()

# ============================================================
# Part 5: Visualize Squared Residuals (OLS Criterion)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Show a subset of points with their squared residual rectangles
subset = np.argsort(np.abs(residuals))[-10:]  # 10 largest residuals
ax.scatter(x, y, alpha=0.4, s=20, c="#3498db")
ax.plot(x_line, y_line, "r-", linewidth=2)

for i in subset:
    xi, yi, yi_pred = x[i], y[i], y_pred[i]
    res = yi - yi_pred
    # Draw the residual line
    ax.plot([xi, xi], [yi, yi_pred], "k-", linewidth=1, alpha=0.5)
    # Draw a square proportional to the squared residual
    size = abs(res) / 3
    rect = plt.Rectangle((xi - size / 2, min(yi, yi_pred)), size, abs(res),
                          linewidth=1, edgecolor="orange", facecolor="orange", alpha=0.3)
    ax.add_patch(rect)

ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title("OLS: Minimizing Squared Residuals\n(orange squares show error magnitudes)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nKey insight: Simple linear regression finds the line that minimizes")
print("the sum of squared residuals. R-squared tells you how much variance")
print("the line explains. But always check residual plots — a high R-squared")
print("with patterned residuals means the model is systematically wrong.")
```

---

## Key Takeaways

- **OLS finds the best straight line** by minimizing the sum of squared residuals (vertical distances from data points to the line)
- **The slope is the key insight**: It tells you the expected change in y for a one-unit increase in x, making it directly interpretable
- **R-squared measures explanatory power**: It is the fraction of variance in y explained by the model, but it should never be the sole metric — always examine residuals too
- **Residual analysis is non-negotiable**: Random residuals confirm the model is appropriate. Patterns in residuals (curves, fans, clusters) indicate the model is systematically wrong
- **Simple linear regression is the building block** for all more advanced regression techniques — understanding it deeply pays dividends throughout your ML journey
