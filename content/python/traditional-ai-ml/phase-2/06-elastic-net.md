# Elastic Net

> Phase 2 — Supervised Learning: Regression | Kata 2.6

---

## Concept & Intuition

### What problem are we solving?

Ridge regression keeps all features but shrinks them (L2 penalty). Lasso performs feature selection by zeroing some out (L1 penalty). But what if you want BOTH — some feature selection AND grouped shrinkage of correlated features? **Elastic Net** combines the L1 and L2 penalties, getting the best of both worlds. Its loss function is: minimize [Sum of Squared Residuals + alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) * ||w||_2^2].

The `l1_ratio` parameter controls the mix: l1_ratio = 1 is pure Lasso, l1_ratio = 0 is pure Ridge, and values in between blend both penalties. In practice, a small amount of L2 (e.g., l1_ratio = 0.5) fixes Lasso's two main weaknesses: the grouping problem (Lasso arbitrarily picks one feature from a correlated group and drops the rest) and the instability when p > n (more features than observations, where Lasso selects at most n features).

Elastic Net is especially valuable in high-dimensional settings like genomics (thousands of genes, hundreds of patients), text analysis (large vocabularies), and sensor networks (many correlated measurements). Whenever you need feature selection but have correlated features, Elastic Net should be your first choice.

### Why naive approaches fail

Using pure Lasso on correlated features leads to inconsistent feature selection — one run might select feature A from a correlated group, another run (with slightly different data) might select feature B instead. This instability makes the model hard to interpret and reproduce. The L2 component of Elastic Net stabilizes the selection by encouraging correlated features to be included or excluded together.

Using pure Ridge in high dimensions (many features, few observations) keeps all features with nonzero coefficients, making the model hard to interpret and potentially overfitting on noise features. Elastic Net's L1 component provides the sparsity that Ridge cannot achieve.

### Mental models

- **The best of both worlds**: Think of Elastic Net as a dimmer switch between Ridge (all features, all shrunk) and Lasso (some features, rest eliminated). You can set it anywhere in between
- **The correlated group behavior**: Lasso is like a talent scout who picks one player from each position group. Elastic Net is like a scout who picks the whole position group or none of them — more stable and reproducible
- **l1_ratio as a personality dial**: l1_ratio near 1 = "aggressive feature selector" (Lasso-like). Near 0 = "conservative, keep everything" (Ridge-like). 0.5 = balanced blend
- **Two knobs, one goal**: alpha controls how much total regularization (more = simpler model). l1_ratio controls what kind of regularization (L1 vs L2 mix). Both need tuning

### Visual explanations

```
Elastic Net = L1 + L2 Combined
=================================

  Loss = RSS + alpha * [l1_ratio * |w| + (1-l1_ratio) * w²]
                            ↑                    ↑
                        Lasso part           Ridge part
                        (sparsity)           (stability)


Constraint Geometry
=====================

  L1 (Lasso)        Elastic Net       L2 (Ridge)
  l1_ratio=1        l1_ratio=0.5      l1_ratio=0

    w2                  w2                  w2
    |/\                 |.·.               |  ___
    /  \               .|   |.            | /   \
  ─/────\─ w1       ──|─────|── w1     ──|─(   )── w1
    \  /               ·|   |·            | \___/
    |\/                 |·.·|              |

  Diamond             Rounded diamond      Circle
  (corners → zeros)   (some corners)       (no corners)
  Max sparsity        Some sparsity        No sparsity


When to Use What
==================

  +───────────────+────────────+──────────────────────────+
  | Scenario      | Best Model | Why                      |
  +───────────────+────────────+──────────────────────────+
  | Few features, | Ridge      | All features likely      |
  | no sparsity   |            | useful, just regularize  |
  +───────────────+────────────+──────────────────────────+
  | Many features,| Lasso      | Need feature selection,  |
  | uncorrelated  |            | features independent     |
  +───────────────+────────────+──────────────────────────+
  | Many features,| Elastic Net| Need feature selection + |
  | correlated    |            | grouped shrinkage        |
  +───────────────+────────────+──────────────────────────+
  | p >> n        | Elastic Net| Lasso limited to n       |
  | (features >>  |            | features; Elastic Net    |
  | observations) |            | can select more          |
  +───────────────+────────────+──────────────────────────+
```

---

## Hands-on Exploration

1. **Sweep l1_ratio**: Run the code with different l1_ratio values from 0 (pure Ridge) to 1 (pure Lasso). Watch how the number of selected features and model behavior changes smoothly between the two extremes.

2. **The grouping effect**: Look at what happens to correlated features (features 0 and 3-5 are correlated in the generated data). With Lasso, one gets selected and the rest are dropped. With Elastic Net, they tend to be selected or dropped together.

3. **Cross-validate both parameters**: The code uses ElasticNetCV to simultaneously optimize alpha and l1_ratio. Compare the selected values with your manual exploration.

4. **High-dimensional experiment**: The code includes a scenario with more features than observations (p > n). Compare Lasso, Ridge, and Elastic Net performance in this challenging setting.

---

## Live Code

```python
"""
Elastic Net — L1 + L2 regularization for sparse, stable regression.

This code compares Elastic Net with Ridge and Lasso, demonstrating
the grouping effect, sparsity control, and optimal parameter selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# Part 1: Generate Data with Correlated Feature Groups
# ============================================================

np.random.seed(42)
n = 200
p = 15

# Create feature groups
base1 = np.random.randn(n)
base2 = np.random.randn(n)

X = np.column_stack([
    base1,                                          # x0: group 1
    base2,                                          # x1: group 2
    np.random.randn(n),                             # x2: independent true feature
    base1 + np.random.randn(n) * 0.1,              # x3: correlated with x0
    base1 + np.random.randn(n) * 0.15,             # x4: correlated with x0
    base2 + np.random.randn(n) * 0.1,              # x5: correlated with x1
    *[np.random.randn(n) for _ in range(9)],        # x6-x14: noise
])

feature_names = [f"x{i}" for i in range(p)]
feature_labels = [
    "x0 (grp1)", "x1 (grp2)", "x2 (indep)",
    "x3 (grp1)", "x4 (grp1)", "x5 (grp2)",
] + [f"x{i} (noise)" for i in range(6, p)]

# True relationship: uses features from group 1, group 2, and x2
true_coefs = np.zeros(p)
true_coefs[0] = 5.0   # group 1 leader
true_coefs[1] = -3.0  # group 2 leader
true_coefs[2] = 2.0   # independent

y = X @ true_coefs + np.random.randn(n) * 1.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("=" * 60)
print("ELASTIC NET — L1 + L2 Combined Regularization")
print("=" * 60)

# ============================================================
# Part 2: Compare Lasso, Ridge, and Elastic Net
# ============================================================

alpha_val = 0.5

ols = LinearRegression().fit(X_train_s, y_train)
ridge = Ridge(alpha=alpha_val).fit(X_train_s, y_train)
lasso = Lasso(alpha=alpha_val, max_iter=10000).fit(X_train_s, y_train)
elastic = ElasticNet(alpha=alpha_val, l1_ratio=0.5, max_iter=10000).fit(X_train_s, y_train)

models = {"OLS": ols, "Ridge": ridge, "Lasso": lasso, "Elastic Net (0.5)": elastic}

print(f"\n--- Model Comparison (alpha = {alpha_val}) ---")
print(f"  {'Model':<25} {'Train R²':>10} {'Test R²':>10} {'#Nonzero':>10} {'L1 Norm':>10}")
print(f"  {'-' * 70}")
for name, model in models.items():
    tr = model.score(X_train_s, y_train)
    te = model.score(X_test_s, y_test)
    nz = np.sum(np.abs(model.coef_) > 1e-6)
    l1 = np.sum(np.abs(model.coef_))
    print(f"  {name:<25} {tr:>10.4f} {te:>10.4f} {nz:>10} {l1:>10.2f}")

# ============================================================
# Part 3: Grouping Effect Demonstration
# ============================================================

print(f"\n--- Grouping Effect (Correlated Features) ---")
print(f"  Features x0, x3, x4 are correlated (group 1)")
print(f"  Features x1, x5 are correlated (group 2)")
print(f"\n  {'Feature':<15} {'Lasso':>10} {'Elastic Net':>12} {'Ridge':>10}")
print(f"  {'-' * 50}")
for i, label in enumerate(feature_labels):
    l_coef = lasso.coef_[i]
    e_coef = elastic.coef_[i]
    r_coef = ridge.coef_[i]
    marker = ""
    if i in [0, 3, 4]:
        marker = " [group 1]"
    elif i in [1, 5]:
        marker = " [group 2]"
    print(f"  {label:<15} {l_coef:>10.3f} {e_coef:>12.3f} {r_coef:>10.3f}{marker}")

# ============================================================
# Part 4: Cross-Validation for Optimal Parameters
# ============================================================

print(f"\n--- Cross-Validation for Optimal Parameters ---")

l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
alphas = np.logspace(-3, 1, 50)

enet_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=5, max_iter=10000)
enet_cv.fit(X_train_s, y_train)

print(f"  Best alpha: {enet_cv.alpha_:.4f}")
print(f"  Best l1_ratio: {enet_cv.l1_ratio_:.2f}")
print(f"  Test R²: {enet_cv.score(X_test_s, y_test):.4f}")
print(f"  Nonzero features: {np.sum(np.abs(enet_cv.coef_) > 1e-6)}")

# ============================================================
# Part 5: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Coefficient comparison
ax = axes[0, 0]
x_pos = np.arange(p)
width = 0.2
ax.bar(x_pos - 1.5 * width, ridge.coef_, width, color="#3498db", alpha=0.7, label="Ridge")
ax.bar(x_pos - 0.5 * width, lasso.coef_, width, color="#e74c3c", alpha=0.7, label="Lasso")
ax.bar(x_pos + 0.5 * width, elastic.coef_, width, color="#2ecc71", alpha=0.7, label="Elastic Net")
ax.bar(x_pos + 1.5 * width, enet_cv.coef_, width, color="#f39c12", alpha=0.7, label="ElasticNetCV")
ax.scatter([0, 1, 2], [true_coefs[0], true_coefs[1], true_coefs[2]], color="black",
           s=100, marker="*", zorder=5, label="True coefs")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"x{i}" for i in range(p)], rotation=45, fontsize=8)
ax.set_ylabel("Coefficient")
ax.set_title("Coefficient Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)

# Elastic Net paths for different l1_ratios
ax = axes[0, 1]
test_r2_by_ratio = {}
for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r2s = []
    for a in alphas:
        if ratio == 0.0:
            m = Ridge(alpha=a).fit(X_train_s, y_train)
        else:
            m = ElasticNet(alpha=a, l1_ratio=ratio, max_iter=10000).fit(X_train_s, y_train)
        r2s.append(m.score(X_test_s, y_test))
    label = "Ridge" if ratio == 0 else ("Lasso" if ratio == 1 else f"EN (l1={ratio})")
    ax.semilogx(alphas, r2s, linewidth=2, label=label)

ax.set_xlabel("Alpha (log scale)")
ax.set_ylabel("Test R²")
ax.set_title("Test R² vs Alpha for Different l1_ratios", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Sparsity vs l1_ratio
ax = axes[1, 0]
ratios_sweep = np.linspace(0.01, 1.0, 50)
n_nonzero = []
for ratio in ratios_sweep:
    m = ElasticNet(alpha=0.3, l1_ratio=ratio, max_iter=10000).fit(X_train_s, y_train)
    n_nonzero.append(np.sum(np.abs(m.coef_) > 1e-6))

ax.plot(ratios_sweep, n_nonzero, "b-", linewidth=2)
ax.set_xlabel("l1_ratio (0=Ridge, 1=Lasso)")
ax.set_ylabel("Number of Nonzero Coefficients")
ax.set_title("Sparsity vs l1_ratio (alpha=0.3)", fontweight="bold")
ax.axhline(y=3, color="green", linestyle=":", alpha=0.5, label="True # features (3)")
ax.legend()
ax.grid(True, alpha=0.3)

# Decision guide
ax = axes[1, 1]
ax.axis("off")
guide = [
    "Elastic Net Decision Guide",
    "",
    "Use Ridge (l1_ratio = 0) when:",
    "  - All features are likely useful",
    "  - You want stable coefficients",
    "  - Multicollinearity is the main concern",
    "",
    "Use Lasso (l1_ratio = 1) when:",
    "  - Many features are irrelevant (sparse signal)",
    "  - Features are mostly independent",
    "  - You want feature selection",
    "",
    "Use Elastic Net (0 < l1_ratio < 1) when:",
    "  - You need feature selection AND have correlated features",
    "  - p >> n (more features than observations)",
    "  - You want the best of both worlds",
    "",
    "Always use cross-validation (ElasticNetCV)",
    "to select both alpha and l1_ratio!",
]
for i, line in enumerate(guide):
    weight = "bold" if i == 0 or "Use " in line or "Always" in line else "normal"
    ax.text(0.05, 0.95 - i * 0.048, line, fontsize=10, fontweight=weight,
            family="monospace", transform=ax.transAxes)

plt.tight_layout()
plt.show()

print(f"\nKey insight: Elastic Net combines L1 and L2 penalties, getting Lasso's")
print(f"feature selection with Ridge's stability for correlated features.")
print(f"Use ElasticNetCV to automatically find the optimal alpha and l1_ratio.")
```

---

## Key Takeaways

- **Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties**, providing both feature selection and grouped shrinkage of correlated features
- **The l1_ratio parameter controls the mix**: 0 = pure Ridge, 1 = pure Lasso, values between blend both properties. Cross-validate to find the optimal mix
- **Elastic Net fixes Lasso's grouping problem**: When features are correlated, Lasso arbitrarily picks one; Elastic Net tends to include or exclude the whole group
- **In high-dimensional settings (p >> n)**, Elastic Net is preferred over Lasso because Lasso can select at most n features, while Elastic Net has no such limitation
- **Use ElasticNetCV** to simultaneously optimize both alpha (regularization strength) and l1_ratio (penalty mix) through cross-validation
