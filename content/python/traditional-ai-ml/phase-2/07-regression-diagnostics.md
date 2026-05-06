# Regression Diagnostics

> Phase 2 — Supervised Learning: Regression | Kata 2.7

---

## Concept & Intuition

### What problem are we solving?

Fitting a regression model is only half the battle. The other half is **diagnosing** whether the model is trustworthy — whether its assumptions are met, its predictions are reliable, and its conclusions are valid. Regression diagnostics are the set of techniques for answering the critical question: "Is this model good enough to use, and if not, what is wrong with it?"

Linear regression makes several assumptions: (1) the relationship between features and target is linear, (2) residuals are normally distributed, (3) residuals have constant variance across all predicted values (**homoscedasticity**), and (4) residuals are independent of each other. Violating these assumptions does not make the model fail to run — it produces output that looks plausible but may be systematically wrong. Diagnostics are the tools that reveal these hidden problems.

The three most important diagnostic tools are **residual plots** (scatter plots of residuals vs. predicted values — patterns indicate model misspecification), **Q-Q plots** (comparing residual distribution to a normal distribution — deviations indicate non-normality), and tests for **heteroscedasticity** (non-constant residual variance — which invalidates standard errors and confidence intervals).

### Why naive approaches fail

The most dangerous failure is relying solely on R-squared and RMSE to evaluate a model. A model can have a respectable R-squared of 0.85 while systematically underpredicting for large values and overpredicting for small values — a pattern invisible in summary statistics but immediately obvious in a residual plot. Summary metrics tell you "how much" error there is; diagnostics tell you "what kind" of error and whether it can be fixed.

Another common mistake is ignoring heteroscedasticity. When residual variance increases with the predicted value (a "funnel" shape in the residual plot), the model's confidence intervals are wrong — too narrow for large predictions and too wide for small ones. This leads to overconfident predictions in the high range and underconfident predictions in the low range.

### Mental models

- **Diagnostics as a medical checkup**: You can feel fine (low RMSE) and still have hidden problems (heteroscedasticity, non-normality) that will cause trouble later. Diagnostics are the blood tests that reveal hidden issues
- **Residuals as the model's report card**: If the model is correctly specified, residuals should look like pure random noise — no patterns, constant spread, normally distributed. Any departure from this is a diagnostic signal
- **The Q-Q plot as a mirror**: A Q-Q plot compares your residuals' distribution to the theoretical normal distribution. If they match, points lie on the diagonal. Deviations reveal heavy tails (outliers), skewness, or other non-normality
- **Fix the diagnosis, not the metric**: If residual plots show a curve, the fix is not to tweak alpha or add regularization — it is to add nonlinear features (polynomial terms, interactions) that capture the missing pattern

### Visual explanations

```
Residual Plot Patterns and Their Meaning
==========================================

  GOOD: Random scatter              BAD: Curved pattern
  (assumptions OK)                  (nonlinearity — add polynomial terms)

  res |  * *   *  *                res |  *     *
      |*  *  * *   *                   |   *       *
  ----+------*------              ----+-----*--------
      | *  * *  *                      |      *
      |*   *     *                     |  *       *
      +──────────── pred              +──────────── pred


  BAD: Funnel shape                BAD: Cluster pattern
  (heteroscedasticity)             (missing categorical variable)

  res |         * *  *  *         res |  * *         * *
      |      *    * *   *              |  * *         * *
  ----+---*---*---------          ----+-------*-*--------
      |   * *   *  *  *              |        * *
      |*  *                           |        * *
      +──────────── pred              +──────────── pred


Q-Q Plot Reading Guide
========================

  Normal (good):        Heavy tails:         Right-skewed:
  quantile              quantile             quantile
     |    /                |  _/                |     _/
     |   /                 | /                  |   _/
     |  /                  |/                   | _/
     | /                  /|                   _/|
     |/                 _/ |                 _/ |
     +──── theory       +──── theory        +──── theory

  Points on line =     Points curve at     Points curve above
  residuals are normal ends = outliers     = positive skew
```

---

## Hands-on Exploration

1. **Run the full diagnostic suite**: Execute the code below to see residual plots, Q-Q plots, and heteroscedasticity tests for both a well-specified and a misspecified model. Compare the two.

2. **Identify the problem**: The misspecified model has a clear pattern in its residual plot. What type of model modification would fix it? (Hint: look at the shape of the residual pattern.)

3. **Test your fix**: After identifying the problem, modify the model (e.g., add polynomial terms) and re-run diagnostics. Verify that the residual pattern disappears.

4. **Real-world practice**: Apply these diagnostics to any regression model you have built. Look for patterns you might have missed. Even experienced data scientists regularly discover issues through residual analysis.

---

## Live Code

```python
"""
Regression Diagnostics — Residual plots, Q-Q plots, and heteroscedasticity tests.

This code demonstrates how to diagnose regression model assumptions and
identify systematic problems that summary metrics cannot reveal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# Part 1: Generate Two Datasets — One Linear, One Nonlinear
# ============================================================

np.random.seed(42)
n = 300

# Dataset 1: True linear relationship (homoscedastic)
x1 = np.random.uniform(0, 20, n)
y1 = 3 * x1 + 10 + np.random.normal(0, 5, n)

# Dataset 2: Nonlinear relationship with heteroscedasticity
x2 = np.random.uniform(0, 20, n)
y2 = 0.5 * x2 ** 2 - 3 * x2 + 20 + np.random.normal(0, 2 * (1 + 0.3 * x2), n)

print("=" * 60)
print("REGRESSION DIAGNOSTICS")
print("=" * 60)

# ============================================================
# Part 2: Fit Linear Models to Both Datasets
# ============================================================

models = {}
predictions = {}
residuals_dict = {}

for name, (x, y_actual) in [("Linear Data", (x1, y1)), ("Nonlinear Data", (x2, y2))]:
    X = x.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_actual, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    res = y_actual - y_pred

    models[name] = model
    predictions[name] = y_pred
    residuals_dict[name] = res

    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    print(f"\n--- {name} ---")
    print(f"  R²: {r2:.4f}  |  RMSE: {rmse:.2f}")
    print(f"  Slope: {model.coef_[0]:.4f}  |  Intercept: {model.intercept_:.4f}")

# ============================================================
# Part 3: Comprehensive Diagnostic Plots
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

datasets = [("Linear Data", x1, y1), ("Nonlinear Data", x2, y2)]

for col, (name, x, y_actual) in enumerate(datasets):
    y_pred = predictions[name]
    res = residuals_dict[name]

    # Row 1: Scatter plot with regression line
    ax = axes[0, col]
    ax.scatter(x, y_actual, alpha=0.4, s=20, c="#3498db")
    x_sorted = np.sort(x)
    y_line = models[name].predict(x_sorted.reshape(-1, 1))
    ax.plot(x_sorted, y_line, "r-", linewidth=2, label="Linear fit")
    r2 = r2_score(y_actual, y_pred)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{name}: Scatter + Fit (R² = {r2:.3f})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Residual plot
    ax = axes[1, col]
    ax.scatter(y_pred, res, alpha=0.4, s=20, c="#e74c3c")
    ax.axhline(y=0, color="black", linewidth=1)
    # Add LOWESS smoothing to reveal patterns
    from statsmodels.nonparametric.smoothers_lowess import lowess
    try:
        smooth = lowess(res, y_pred, frac=0.3)
        ax.plot(smooth[:, 0], smooth[:, 1], "b-", linewidth=2, label="LOWESS trend")
    except Exception:
        pass
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")
    ax.set_title(f"{name}: Residual Plot", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3: Q-Q plot
    ax = axes[2, col]
    (theoretical, sample), (slope, intercept, r) = stats.probplot(res, dist="norm")
    ax.scatter(theoretical, sample, alpha=0.5, s=20, c="#2ecc71")
    x_line = np.array([theoretical.min(), theoretical.max()])
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label="Reference line")
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles (Residuals)")
    ax.set_title(f"{name}: Q-Q Plot", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Part 4: Statistical Tests
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL DIAGNOSTIC TESTS")
print("=" * 60)

for name, (x, y_actual) in [("Linear Data", (x1, y1)), ("Nonlinear Data", (x2, y2))]:
    res = residuals_dict[name]
    y_pred = predictions[name]

    print(f"\n--- {name} ---")

    # Normality test (Shapiro-Wilk)
    if len(res) > 5000:
        stat, p_val = stats.shapiro(res[:5000])
    else:
        stat, p_val = stats.shapiro(res)
    normal = "YES" if p_val > 0.05 else "NO"
    print(f"  Shapiro-Wilk normality test: statistic={stat:.4f}, p={p_val:.4f} -> Normal? {normal}")

    # Heteroscedasticity test (Breusch-Pagan style: regress squared residuals on predictions)
    res_squared = res ** 2
    bp_model = LinearRegression()
    bp_model.fit(y_pred.reshape(-1, 1), res_squared)
    bp_r2 = bp_model.score(y_pred.reshape(-1, 1), res_squared)
    bp_f = bp_r2 * n / (1 - bp_r2 + 1e-10)
    bp_p = 1 - stats.f.cdf(bp_f, 1, n - 2)
    homo = "YES" if bp_p > 0.05 else "NO"
    print(f"  Heteroscedasticity test: F={bp_f:.2f}, p={bp_p:.4f} -> Homoscedastic? {homo}")

    # Mean residual (should be ~0)
    print(f"  Mean residual: {res.mean():.6f} (should be ~0)")

    # Residual autocorrelation (Durbin-Watson approximation)
    dw = np.sum(np.diff(res) ** 2) / np.sum(res ** 2)
    autocorr = "Likely" if dw < 1.5 or dw > 2.5 else "Unlikely"
    print(f"  Durbin-Watson: {dw:.3f} (ideal = 2.0) -> Autocorrelation? {autocorr}")

# ============================================================
# Part 5: Fix the Nonlinear Model
# ============================================================

print("\n" + "=" * 60)
print("FIXING THE MISSPECIFIED MODEL")
print("=" * 60)

# Apply polynomial features to fix the nonlinear model
X2 = x2.reshape(-1, 1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

fixed_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("model", LinearRegression()),
])
fixed_pipeline.fit(X2_train, y2_train)
y2_pred_fixed = fixed_pipeline.predict(X2)
res_fixed = y2 - y2_pred_fixed

r2_linear = r2_score(y2, predictions["Nonlinear Data"])
r2_fixed = r2_score(y2, y2_pred_fixed)

print(f"\n  Linear model R²: {r2_linear:.4f}")
print(f"  Polynomial (degree 2) model R²: {r2_fixed:.4f}")
print(f"  Improvement: {r2_fixed - r2_linear:+.4f}")

# Diagnostic comparison: before and after fix
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Before fix: residual plot
ax = axes[0, 0]
res_before = residuals_dict["Nonlinear Data"]
y_pred_before = predictions["Nonlinear Data"]
ax.scatter(y_pred_before, res_before, alpha=0.4, s=20, c="#e74c3c")
ax.axhline(y=0, color="black", linewidth=1)
ax.set_title(f"BEFORE Fix: Residual Plot (Linear Model)\nR² = {r2_linear:.3f}",
             fontweight="bold", color="#e74c3c")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
ax.grid(True, alpha=0.3)

# After fix: residual plot
ax = axes[0, 1]
ax.scatter(y2_pred_fixed, res_fixed, alpha=0.4, s=20, c="#2ecc71")
ax.axhline(y=0, color="black", linewidth=1)
ax.set_title(f"AFTER Fix: Residual Plot (Polynomial Model)\nR² = {r2_fixed:.3f}",
             fontweight="bold", color="#2ecc71")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
ax.grid(True, alpha=0.3)

# Before fix: Q-Q plot
ax = axes[1, 0]
(theoretical, sample), (slope, intercept, r) = stats.probplot(res_before, dist="norm")
ax.scatter(theoretical, sample, alpha=0.5, s=20, c="#e74c3c")
x_line = np.array([theoretical.min(), theoretical.max()])
ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2)
ax.set_title("BEFORE Fix: Q-Q Plot", fontweight="bold", color="#e74c3c")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.grid(True, alpha=0.3)

# After fix: Q-Q plot
ax = axes[1, 1]
(theoretical, sample), (slope, intercept, r) = stats.probplot(res_fixed, dist="norm")
ax.scatter(theoretical, sample, alpha=0.5, s=20, c="#2ecc71")
x_line = np.array([theoretical.min(), theoretical.max()])
ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2)
ax.set_title("AFTER Fix: Q-Q Plot", fontweight="bold", color="#2ecc71")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.grid(True, alpha=0.3)

plt.suptitle("Before vs After: Fixing Model Misspecification", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# Part 6: Heteroscedasticity Deep Dive
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scale-Location plot (sqrt of standardized residuals vs predicted)
for col, (name, res, y_pred) in enumerate([
    ("Linear (Homoscedastic)", residuals_dict["Linear Data"], predictions["Linear Data"]),
    ("Nonlinear (Heteroscedastic)", residuals_dict["Nonlinear Data"], predictions["Nonlinear Data"]),
]):
    ax = axes[col]
    std_res = res / res.std()
    sqrt_abs_res = np.sqrt(np.abs(std_res))
    ax.scatter(y_pred, sqrt_abs_res, alpha=0.4, s=20, c="#f39c12")

    # Add trend line
    z = np.polyfit(y_pred, sqrt_abs_res, 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(y_pred.min(), y_pred.max(), 100)
    ax.plot(x_range, p_line(x_range), "r-", linewidth=2, label="Trend")

    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("sqrt(|Standardized Residual|)")
    ax.set_title(f"Scale-Location Plot: {name}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nKey insight: Regression diagnostics reveal problems that summary")
print("statistics hide. Always check residual plots, Q-Q plots, and")
print("heteroscedasticity before trusting a model's predictions.")
print("A model with good R² but bad diagnostics is unreliable.")
```

---

## Key Takeaways

- **Summary metrics are not enough**: R-squared and RMSE tell you how much error there is, but residual plots tell you what KIND of error — systematic vs. random
- **Residual plots are the most important diagnostic**: Random scatter = good. Curves = nonlinearity (add polynomial terms). Funnels = heteroscedasticity (transform the target or use weighted regression)
- **Q-Q plots check normality of residuals**: Points on the diagonal = normal. Curved ends = heavy tails/outliers. Systematic curves = skewness. Non-normal residuals invalidate confidence intervals and hypothesis tests
- **Heteroscedasticity invalidates standard errors**: When residual variance changes with the predicted value, confidence intervals are wrong. Use robust standard errors, weighted regression, or log-transform the target
- **Diagnostics guide model improvement**: They do not just say "the model is bad" — they tell you exactly what is wrong and point toward specific fixes (add features, transform variables, change model family)
