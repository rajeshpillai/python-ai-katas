# Multiple Linear Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.2

---

## Concept & Intuition

### What problem are we solving?

Real-world outcomes rarely depend on a single variable. House prices depend on size, location, age, and dozens of other factors simultaneously. Multiple linear regression extends simple linear regression to handle multiple input features: y = b0 + b1*x1 + b2*x2 + ... + bn*xn. Each coefficient bi tells you how much y changes when feature xi increases by one unit, **holding all other features constant**.

This "holding all other features constant" caveat is crucial and often misunderstood. In simple regression of house price on square footage, the coefficient captures everything correlated with size — including the number of bedrooms, neighborhood quality, and lot size. In multiple regression, the coefficient for square footage captures only the effect of size after accounting for all other included features. This is why coefficients can change dramatically when you add or remove features.

Multiple regression is the workhorse of applied statistics and data science. It handles multiple predictors, provides interpretable coefficients, and serves as the foundation for regularized models (Ridge, Lasso) and many other techniques. Understanding its mechanics, assumptions, and limitations is essential for any data scientist.

### Why naive approaches fail

The most dangerous mistake in multiple regression is **multicollinearity** — including features that are highly correlated with each other. When two features carry nearly the same information (e.g., square footage and number of rooms), the model cannot distinguish their individual effects. The coefficients become unstable, with wild values that flip sign when a single data point changes. The model's predictions may still be accurate, but the individual coefficients become meaningless.

Another common mistake is interpreting coefficients as causal effects. The coefficient for "ice cream sales" in a model predicting "drowning deaths" is positive, but ice cream does not cause drowning — both are driven by warm weather (a confounding variable). Regression finds correlations, not causes.

### Mental models

- **The tug of war analogy**: Each feature pulls the prediction in its direction with a strength proportional to its coefficient. The final prediction is the combined effect of all features pulling together
- **Coefficients as partial effects**: Think of each coefficient as the answer to: "If I could change ONLY this feature and hold everything else fixed, how much would y change?" This is the essence of "all else being equal"
- **The attribution problem**: Multiple regression decomposes the prediction into contributions from each feature. This is like attributing a team's success to individual players — the attribution depends on who else is on the team
- **Adding features is not free**: Each additional feature costs you a degree of freedom and increases the risk of overfitting and multicollinearity. More features do not always mean better models

### Visual explanations

```
Multiple Linear Regression
============================

  Simple (1 feature):       Multiple (2 features):
  y = b0 + b1*x             y = b0 + b1*x1 + b2*x2

  y                          y
  |       /                  |      ╱╱╱╱
  |     /                    |    ╱╱╱╱    <- regression PLANE
  |   /  <- line             |  ╱╱╱╱        (not a line!)
  | /                        |╱╱╱╱
  +────── x                  +────────── x1
                            ╱
                          x2


Coefficient Interpretation
============================

  Model: Price = $50,000 + $100 * sqft + $15,000 * bedrooms + $30,000 * quality

  Reading each coefficient:
  - Each additional sq ft adds $100 to price (holding bedrooms, quality fixed)
  - Each additional bedroom adds $15,000 (holding sqft, quality fixed)
  - Each quality level adds $30,000 (holding sqft, bedrooms fixed)

  Note: These are PARTIAL effects — the effect of one feature
  after accounting for all others.


Multicollinearity Problem
===========================

  Feature A: square footage     (correlated with bedrooms!)
  Feature B: number of bedrooms

  Alone:  Price = $100 * sqft         (coefficient = $100)
  Alone:  Price = $50,000 * bedrooms  (coefficient = $50,000)

  Together: The model cannot separate their effects!
  Coefficients become unstable and hard to interpret.

  sqft ←──── highly correlated ────→ bedrooms
              (r = 0.85)

  Solution: Check VIF (Variance Inflation Factor)
  VIF > 5-10 = multicollinearity concern!
```

---

## Hands-on Exploration

1. **Fit and interpret**: Run the code below to fit a multiple regression on housing data. Interpret each coefficient in plain English using the "one unit increase, all else equal" framework.

2. **Feature importance**: Look at how the coefficients compare in magnitude. But be careful — coefficients on different scales are not directly comparable. The code also shows standardized coefficients for fair comparison.

3. **Multicollinearity detection**: Examine the VIF (Variance Inflation Factor) for each feature. Features with VIF > 5-10 are collinear. Try removing one of a collinear pair and see how the remaining coefficients change.

4. **Add and remove features**: Start with one feature and progressively add more. Watch how R-squared changes and how existing coefficients shift. This reveals how features interact and compete for explanatory power.

---

## Live Code

```python
"""
Multiple Linear Regression — Multiple features, coefficients, and multicollinearity.

This code fits a multiple regression model, interprets coefficients,
detects multicollinearity, and visualizes feature contributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Part 1: Generate Housing Dataset
# ============================================================

np.random.seed(42)
n = 500

sqft = np.random.normal(1800, 500, n).clip(500, 5000)
bedrooms = (sqft / 600 + np.random.normal(0, 0.5, n)).clip(1, 7).round()
age = np.random.exponential(15, n).clip(0, 80).round()
garage = np.random.choice([0, 1, 2, 3], size=n, p=[0.1, 0.3, 0.4, 0.2])
neighborhood = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.35, 0.30, 0.15])
lot_size = np.random.lognormal(8.5, 0.4, n).round()

# True relationship
price = (100 * sqft
         + 15000 * bedrooms
         - 800 * age
         + 25000 * garage
         + 35000 * neighborhood
         + 0.3 * lot_size
         + np.random.normal(0, 20000, n))

df = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "age": age,
    "garage": garage,
    "neighborhood": neighborhood,
    "lot_size": lot_size,
    "price": price,
})

features = ["sqft", "bedrooms", "age", "garage", "neighborhood", "lot_size"]
X = df[features]
y = df["price"]

print("=" * 60)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 60)
print(f"\nDataset: {n} houses, {len(features)} features")
print(f"Target: price (mean=${y.mean():,.0f}, std=${y.std():,.0f})")

# ============================================================
# Part 2: Fit the Model
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"\n--- Model Performance ---")
print(f"  Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"  Test R²:  {r2_score(y_test, y_pred_test):.4f}")
print(f"  Test RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test)):,.0f}")

# ============================================================
# Part 3: Coefficient Interpretation
# ============================================================

print(f"\n--- Coefficients (Raw) ---")
print(f"  {'Feature':<20} {'Coefficient':>15} {'Interpretation'}")
print(f"  {'-' * 70}")
print(f"  {'Intercept':<20} {model.intercept_:>15,.2f}  Base price when all features = 0")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:<20} {coef:>15,.2f}  +1 {feat} -> ${coef:+,.0f} in price")

# Standardized coefficients (for fair comparison)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model_std = LinearRegression()
model_std.fit(X_train_scaled, y_train)

print(f"\n--- Standardized Coefficients (for feature importance comparison) ---")
std_coefs = sorted(zip(features, model_std.coef_), key=lambda x: abs(x[1]), reverse=True)
for feat, coef in std_coefs:
    bar = "+" * int(abs(coef) / 1000) if coef > 0 else "-" * int(abs(coef) / 1000)
    print(f"  {feat:<20} {coef:>12,.0f}  {bar}")

# ============================================================
# Part 4: Multicollinearity Detection (VIF)
# ============================================================

print(f"\n--- Multicollinearity Check (VIF) ---")

def compute_vif(X_df):
    """Compute Variance Inflation Factor for each feature."""
    vifs = []
    for i, col in enumerate(X_df.columns):
        others = [c for c in X_df.columns if c != col]
        r2_i = LinearRegression().fit(X_df[others], X_df[col]).score(X_df[others], X_df[col])
        vif = 1 / (1 - r2_i) if r2_i < 1 else float("inf")
        vifs.append(vif)
    return vifs

vifs = compute_vif(X_train)
print(f"  {'Feature':<20} {'VIF':>8}  {'Status'}")
print(f"  {'-' * 50}")
for feat, vif in zip(features, vifs):
    status = "OK" if vif < 5 else ("WARNING" if vif < 10 else "HIGH!")
    print(f"  {feat:<20} {vif:>8.2f}  {status}")

# ============================================================
# Part 5: Progressive Feature Addition
# ============================================================

print(f"\n--- Progressive Feature Addition ---")
print(f"  {'Features':<50} {'Train R²':>10} {'Test R²':>10}")
print(f"  {'-' * 75}")

feature_order = ["sqft", "neighborhood", "age", "garage", "bedrooms", "lot_size"]
for k in range(1, len(feature_order) + 1):
    feats_k = feature_order[:k]
    model_k = LinearRegression()
    model_k.fit(X_train[feats_k], y_train)
    tr_r2 = model_k.score(X_train[feats_k], y_train)
    te_r2 = model_k.score(X_test[feats_k], y_test)
    print(f"  {', '.join(feats_k):<50} {tr_r2:>10.4f} {te_r2:>10.4f}")

# ============================================================
# Part 6: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=20, c="#3498db")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Price ($)", fontsize=11)
ax.set_ylabel("Predicted Price ($)", fontsize=11)
ax.set_title(f"Actual vs Predicted (R² = {r2_score(y_test, y_pred_test):.3f})",
             fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# Coefficient bar chart
ax = axes[0, 1]
std_feats, std_vals = zip(*std_coefs)
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in std_vals]
ax.barh(range(len(std_feats)), std_vals, color=colors, edgecolor="black", alpha=0.8)
ax.set_yticks(range(len(std_feats)))
ax.set_yticklabels(std_feats)
ax.set_xlabel("Standardized Coefficient")
ax.set_title("Feature Importance\n(Standardized Coefficients)", fontsize=13, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.5)
ax.grid(axis="x", alpha=0.3)

# Residual plot
ax = axes[1, 0]
residuals_test = y_test - y_pred_test
ax.scatter(y_pred_test, residuals_test, alpha=0.5, s=20, c="#e74c3c")
ax.axhline(y=0, color="black", linewidth=1)
ax.set_xlabel("Predicted Price ($)", fontsize=11)
ax.set_ylabel("Residual ($)", fontsize=11)
ax.set_title("Residual Plot", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# VIF bar chart
ax = axes[1, 1]
colors_vif = ["#2ecc71" if v < 5 else ("#f39c12" if v < 10 else "#e74c3c") for v in vifs]
ax.bar(features, vifs, color=colors_vif, edgecolor="black", alpha=0.8)
ax.axhline(y=5, color="orange", linestyle="--", label="Warning threshold (5)")
ax.axhline(y=10, color="red", linestyle="--", label="Critical threshold (10)")
ax.set_ylabel("VIF")
ax.set_title("Variance Inflation Factor\n(Multicollinearity Check)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

print("\nKey insight: Multiple regression decomposes predictions into contributions")
print("from each feature. Coefficients show partial effects (all else equal),")
print("but multicollinearity makes individual coefficients unreliable.")
print("Always check VIF and use standardized coefficients for fair comparison.")
```

---

## Key Takeaways

- **Multiple regression models y as a linear combination of multiple features**, with each coefficient representing the partial effect of that feature while holding all others constant
- **Standardize coefficients for fair comparison**: Raw coefficients depend on feature scales and cannot be compared directly. Standardized coefficients show which features have the largest impact
- **Multicollinearity is the hidden danger**: When features are highly correlated, individual coefficients become unstable and uninterpretable. Use VIF (Variance Inflation Factor) to detect it
- **Adding features has diminishing returns**: Each new feature increases model complexity and multicollinearity risk. Monitor how test R-squared changes with each addition
- **Correlation is not causation**: Regression coefficients show associations, not causal effects. Confounding variables, omitted variables, and reverse causality can all mislead interpretation
