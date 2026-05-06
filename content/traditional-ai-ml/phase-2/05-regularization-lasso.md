# Regularization: Lasso Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.5

---

## Concept & Intuition

### What problem are we solving?

Ridge regression shrinks coefficients toward zero but never eliminates them. In datasets with many features, many of which are irrelevant, Ridge still keeps all features in the model with small but nonzero weights. **Lasso regression** (Least Absolute Shrinkage and Selection Operator) uses an L1 penalty — the sum of absolute values of coefficients — which has a remarkable property: it can drive coefficients to exactly zero, performing automatic **feature selection**.

The Lasso objective is: minimize [Sum of Squared Residuals + alpha * Sum of |coefficients|]. The L1 penalty creates "corners" in the optimization landscape where the optimal solution lies exactly on an axis — meaning some coefficients are exactly zero. This is not just a mathematical curiosity; it is practically transformative. A Lasso model with 100 features might discover that only 8 of them matter, automatically discarding the other 92.

This automatic feature selection makes Lasso invaluable when you suspect that only a few features are truly predictive (a common situation in genomics, text analysis, and high-dimensional datasets). The resulting sparse model is more interpretable, more efficient to deploy, and often generalizes better than models that keep all features.

### Why naive approaches fail

Manual feature selection (trying every combination of features) is combinatorially explosive. With 20 features, there are over one million possible subsets. Stepwise selection (adding or removing features one at a time) is a greedy heuristic that often misses the best subset and is statistically unreliable. Lasso performs feature selection as part of the optimization itself — elegantly, efficiently, and automatically.

A subtler issue is using Lasso without scaling features first. Because the L1 penalty treats all coefficients equally, features on larger scales will have smaller coefficients and be penalized less. Without scaling, Lasso's feature selection is biased by feature scales rather than feature importance.

### Mental models

- **L1 as a diamond constraint**: Geometrically, Ridge constrains coefficients to a sphere (circle in 2D), while Lasso constrains them to a diamond. The diamond's corners lie on the axes — that is where coefficients are exactly zero
- **The hiring analogy**: Ridge is like reducing everyone's salary proportionally during a budget cut. Lasso is like laying off some employees entirely while keeping others at close to full pay. Lasso makes the hard choices
- **Sparsity as a feature**: A model that says "only features X, Y, and Z matter" is more useful than one that says "all 100 features contribute a little bit." Sparsity improves interpretability, deployment efficiency, and often accuracy
- **The grouping problem**: When features are correlated, Lasso tends to pick one and zero out the others. This can be arbitrary (it picks whichever happens to fit the training data slightly better). If you want correlated features to be kept or dropped together, use Elastic Net

### Visual explanations

```
L1 (Lasso) vs L2 (Ridge) Geometry
=====================================

  L2 (Ridge): Circle constraint    L1 (Lasso): Diamond constraint

       w2                               w2
       |  ___                            | /\
       | /   \                           |/  \
  ─────|─(   )───── w1            ─────|─    ──── w1
       | \___/                          |\  /
       |                                | \/

  OLS solution (★) intersects        OLS solution (★) intersects
  the circle at a smooth point       the diamond at a CORNER
  -> both coefficients nonzero       -> one coefficient = 0 (sparse!)


Coefficient Paths: Ridge vs Lasso
===================================

  Ridge:                        Lasso:
  Coeff                         Coeff
    |\_____                       |\_
    | \____                       |  \____
    |  \___                       |       \____
    |   \__                       |            \_____
    |    \_                       |                  \___
   0|─────\───── alpha →        0|━━━━━━━━━━━━━━━━━━━━━━━ alpha →
    |      \                      |
    |       \                     | Coefficients hit ZERO and stay!

  Coefficients shrink             Some coefficients become
  but never reach zero            exactly zero (feature selection!)


When to Use Lasso vs Ridge
============================

  Many irrelevant features?     →  Lasso (eliminates them)
  All features somewhat useful? →  Ridge (keeps all, shrinks)
  Correlated feature groups?    →  Elastic Net (combines both)
  Need interpretability?        →  Lasso (sparse = interpretable)
  Need stability?               →  Ridge (stable coefficients)
```

---

## Hands-on Exploration

1. **Adjust alpha**: Use the slider to try different alpha values. Watch how increasing alpha zeros out more and more coefficients. At what alpha do the irrelevant features get eliminated while the true features are preserved?

2. **Compare with Ridge**: Look at the coefficient comparison plot. Notice how Ridge shrinks all coefficients uniformly, while Lasso drives some to exactly zero and keeps others relatively large.

3. **Count selected features**: For each alpha, count how many features have nonzero coefficients. Plot this count vs. alpha — this is the "sparsity path."

4. **Identify the true features**: The data was generated with known true features. Check whether Lasso correctly identifies them. At what alpha does it start making mistakes (dropping true features or keeping false ones)?

---

## Live Code

```python
"""
Lasso Regression — L1 regularization, automatic feature selection, and sparsity.

Adjust alpha to see how Lasso selects features by driving coefficients to zero.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# @param alpha float 0.01 100 1.0

# ============================================================
# Part 1: Generate Sparse Data (few true features, many noise features)
# ============================================================

np.random.seed(42)
n = 200
p = 20  # Total features

X = np.random.randn(n, p)
feature_names = [f"x{i}" for i in range(p)]

# Only first 4 features are truly predictive
true_coefs = np.zeros(p)
true_coefs[0] = 8.0
true_coefs[1] = -5.0
true_coefs[2] = 3.0
true_coefs[3] = -2.0
# Features 4-19 are irrelevant (noise)

y = X @ true_coefs + np.random.randn(n) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("=" * 60)
print(f"LASSO REGRESSION (alpha = {alpha})")
print("=" * 60)

print(f"\nDataset: {n} samples, {p} features")
print(f"True features: x0 (8.0), x1 (-5.0), x2 (3.0), x3 (-2.0)")
print(f"Noise features: x4 through x{p-1} (all zero)")

# ============================================================
# Part 2: Fit OLS, Ridge, and Lasso
# ============================================================

ols = LinearRegression().fit(X_train_s, y_train)
ridge = Ridge(alpha=alpha).fit(X_train_s, y_train)
lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train_s, y_train)

models = {"OLS": ols, "Ridge": ridge, "Lasso": lasso}

print(f"\n--- Model Comparison ---")
print(f"  {'Model':<10} {'Train R²':>10} {'Test R²':>10} {'Nonzero Coefs':>15} {'Coef L1 Norm':>15}")
print(f"  {'-' * 65}")
for name, model in models.items():
    tr_r2 = model.score(X_train_s, y_train)
    te_r2 = model.score(X_test_s, y_test)
    nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    l1_norm = np.sum(np.abs(model.coef_))
    print(f"  {name:<10} {tr_r2:>10.4f} {te_r2:>10.4f} {nonzero:>15} {l1_norm:>15.2f}")

# ============================================================
# Part 3: Lasso Feature Selection Analysis
# ============================================================

print(f"\n--- Lasso Feature Selection (alpha = {alpha}) ---")
print(f"  {'Feature':<10} {'Lasso Coef':>12} {'True Coef':>12} {'Selected?':>10}")
print(f"  {'-' * 50}")
for i, (feat, coef) in enumerate(zip(feature_names, lasso.coef_)):
    selected = "YES" if abs(coef) > 1e-6 else "no"
    true = true_coefs[i]
    marker = " *" if (abs(coef) > 1e-6) != (abs(true) > 0) else ""
    print(f"  {feat:<10} {coef:>12.4f} {true:>12.1f} {selected:>10}{marker}")

n_true_selected = sum(abs(lasso.coef_[i]) > 1e-6 for i in range(4))
n_false_selected = sum(abs(lasso.coef_[i]) > 1e-6 for i in range(4, p))
print(f"\n  True features selected: {n_true_selected}/4")
print(f"  False features selected: {n_false_selected}/{p-4}")

# ============================================================
# Part 4: Coefficient Paths
# ============================================================

alphas = np.logspace(-3, 2, 100)
lasso_paths = []
ridge_paths = []
n_selected = []

for a in alphas:
    l = Lasso(alpha=a, max_iter=10000).fit(X_train_s, y_train)
    r = Ridge(alpha=a).fit(X_train_s, y_train)
    lasso_paths.append(l.coef_.copy())
    ridge_paths.append(r.coef_.copy())
    n_selected.append(np.sum(np.abs(l.coef_) > 1e-6))

lasso_paths = np.array(lasso_paths)
ridge_paths = np.array(ridge_paths)

# Cross-validation for optimal alpha
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_s, y_train)
best_alpha = lasso_cv.alpha_
print(f"\n  Best alpha (CV): {best_alpha:.4f}")
print(f"  Best Lasso Test R²: {lasso_cv.score(X_test_s, y_test):.4f}")

# ============================================================
# Part 5: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Lasso coefficient paths
ax = axes[0, 0]
true_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
for i in range(p):
    color = true_colors[i] if i < 4 else "gray"
    lw = 2.5 if i < 4 else 0.5
    alpha_line = 1.0 if i < 4 else 0.3
    label = f"{feature_names[i]} (true)" if i < 4 else (f"noise" if i == 4 else None)
    ax.semilogx(alphas, lasso_paths[:, i], linewidth=lw, alpha=alpha_line,
                color=color, label=label)
ax.axvline(x=alpha, color="purple", linestyle="--", linewidth=2, label=f"alpha={alpha}")
ax.axvline(x=best_alpha, color="green", linestyle=":", linewidth=2, label=f"Best CV={best_alpha:.3f}")
ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("Coefficient")
ax.set_title("Lasso Coefficient Paths\n(true features in color, noise in gray)", fontweight="bold")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Number of selected features vs alpha
ax = axes[0, 1]
ax.semilogx(alphas, n_selected, "b-", linewidth=2)
ax.axvline(x=alpha, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"alpha={alpha}")
ax.axvline(x=best_alpha, color="green", linestyle=":", linewidth=2, label=f"Best CV={best_alpha:.3f}")
ax.axhline(y=4, color="gray", linestyle=":", alpha=0.5, label="True # features (4)")
ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("Number of Selected Features")
ax.set_title("Feature Sparsity vs Alpha", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# OLS vs Ridge vs Lasso coefficients
ax = axes[1, 0]
x_pos = np.arange(p)
width = 0.25
ax.bar(x_pos - width, ols.coef_, width, color="#95a5a6", alpha=0.6, label="OLS")
ax.bar(x_pos, ridge.coef_, width, color="#3498db", alpha=0.7, label=f"Ridge (a={alpha})")
ax.bar(x_pos + width, lasso.coef_, width, color="#e74c3c", alpha=0.7, label=f"Lasso (a={alpha})")
ax.scatter(range(4), true_coefs[:4], color="green", s=100, marker="*", zorder=5, label="True coefs")
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45, fontsize=8)
ax.set_ylabel("Coefficient Value")
ax.set_title("Coefficient Comparison: OLS vs Ridge vs Lasso", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# L1 vs L2 geometry (2D illustration)
ax = axes[1, 1]
theta = np.linspace(0, 2 * np.pi, 100)

# L2 (circle)
ax.plot(np.cos(theta), np.sin(theta), "b-", linewidth=2, label="L2 (Ridge): circle")

# L1 (diamond)
diamond_x = [1, 0, -1, 0, 1]
diamond_y = [0, 1, 0, -1, 0]
ax.plot(diamond_x, diamond_y, "r-", linewidth=2, label="L1 (Lasso): diamond")

# OLS solution
ax.scatter([0.8], [0.9], c="green", s=200, marker="*", zorder=5, label="OLS solution")

# Show that Lasso solution hits a corner
ax.annotate("Lasso solution\n(w₁ = 0, sparse!)", xy=(0, 1), xytext=(0.5, 1.5),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red", fontweight="bold")

ax.set_xlabel("w₁")
ax.set_ylabel("w₂")
ax.set_title("L1 vs L2 Geometry\n(Lasso hits corners = sparsity)", fontweight="bold")
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)

plt.tight_layout()
plt.show()

print(f"\nKey insight: Lasso performs automatic feature selection by driving")
print(f"irrelevant coefficients to exactly zero. This is especially valuable")
print(f"when you have many features but suspect only a few are truly predictive.")
print(f"Use LassoCV to find the optimal regularization strength.")
```

---

## Key Takeaways

- **Lasso uses an L1 penalty** (sum of absolute coefficient values) which can drive coefficients to exactly zero, performing automatic feature selection
- **Sparsity is Lasso's superpower**: In high-dimensional problems, Lasso identifies the truly important features and eliminates the rest, producing interpretable, efficient models
- **The L1 geometry explains sparsity**: The diamond-shaped L1 constraint has corners on the axes where coefficients are exactly zero; the optimal solution often lands on these corners
- **Lasso has a grouping problem**: When features are correlated, Lasso arbitrarily picks one and zeros out the others. Elastic Net (L1 + L2) addresses this by tending to keep or drop correlated features together
- **Cross-validation (LassoCV) is essential** for choosing the right alpha — too small keeps too many features (overfitting), too large eliminates true features (underfitting)
