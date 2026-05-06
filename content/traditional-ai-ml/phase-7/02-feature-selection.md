# Feature Selection

> Phase 7 — Feature Engineering & Pipelines | Kata 7.2

---

## Concept & Intuition

### What problem are we solving?

After creating many features, you often end up with far more than the model needs. Irrelevant and redundant features add noise, increase computation, and can degrade performance through the curse of dimensionality. Feature selection identifies which features actually matter and discards the rest, producing a simpler, faster, more interpretable model.

There are three main families of feature selection methods. **Filter methods** score each feature independently using statistical tests (correlation, mutual information, chi-squared) and keep the top-k. They are fast but ignore feature interactions. **Wrapper methods** evaluate subsets of features by training the actual model and measuring performance (forward selection, backward elimination, recursive feature elimination). They are accurate but computationally expensive. **Embedded methods** perform feature selection as part of model training (Lasso's L1 regularization, tree-based feature importance). They balance speed and quality.

The right approach depends on your situation. With thousands of features, start with a fast filter to get to a manageable number, then use a wrapper or embedded method for final selection. With a moderate number of features, go directly to embedded methods like Lasso or Random Forest importance.

### Why naive approaches fail

Keeping all features "just in case" is tempting but harmful. With p features and n samples, if p is close to n, the model can find spurious correlations that do not generalize. Removing even obviously irrelevant features improves stability and reduces overfitting. Conversely, selecting features based on their correlation with the target *on the training set* without cross-validation leads to optimistic estimates and information leakage.

### Mental models

- **Signal-to-noise ratio**: each irrelevant feature adds noise. Removing them is like turning down static on a radio -- the signal comes through clearer.
- **Occam's razor**: given two models with equal performance, prefer the simpler one (fewer features). It is more likely to generalize.
- **Three sieves**: filter (coarse, fast) -> embedded (medium, moderate) -> wrapper (fine, slow). Each sieve catches what the previous one missed.

### Visual explanations

```
Feature Selection Methods:

FILTER (fast, independent):
  Score each feature:  F1=0.82  F2=0.15  F3=0.91  F4=0.03  F5=0.67
  Keep top-3:          F3       F1       F5
  (ignores interactions between features)

WRAPPER (slow, model-dependent):
  Try {F1}          --> accuracy = 0.72
  Try {F1, F3}      --> accuracy = 0.85
  Try {F1, F3, F5}  --> accuracy = 0.89
  Try {F1, F3, F5, F2} --> accuracy = 0.88  (F2 hurts!)
  Best: {F1, F3, F5}

EMBEDDED (built into training):
  Lasso:  coef = [0.45, 0.00, 0.73, 0.00, 0.31]
  Selected: F1, F3, F5  (nonzero coefficients)
  (L1 penalty drives irrelevant features to exactly zero)
```

---

## Hands-on Exploration

1. Create a dataset with 20 features where only 5 are truly informative. Fit a model on all features and on just the informative ones -- compare accuracy.
2. Apply a filter method (mutual information) to rank features and observe how well it identifies the informative ones.
3. Use Recursive Feature Elimination (RFE) as a wrapper method and compare the selected subset to the filter method's ranking.
4. Apply Lasso (L1 regularization) as an embedded method and examine which feature coefficients are driven to zero.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest, RFE
)
from sklearn.metrics import accuracy_score

np.random.seed(42)

# --- Dataset: 20 features, only 5 informative ---
X, y = make_classification(
    n_samples=300, n_features=20, n_informative=5,
    n_redundant=3, n_clusters_per_class=2,
    flip_y=0.05, random_state=42
)
feature_names = [f"F{i:02d}" for i in range(20)]

# --- Baseline: all features ---
lr_all = LogisticRegression(max_iter=1000, random_state=42)
scores_all = cross_val_score(lr_all, X, y, cv=5, scoring="accuracy")
print("=== Baseline (all 20 features) ===")
print(f"  Accuracy: {scores_all.mean():.4f} (+/- {scores_all.std():.4f})\n")

# --- FILTER: Mutual Information ---
print("=== Filter Method: Mutual Information ===\n")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_ranking = np.argsort(mi_scores)[::-1]

for rank, idx in enumerate(mi_ranking[:10]):
    bar = "|" * int(mi_scores[idx] * 50)
    print(f"  Rank {rank+1:>2}: {feature_names[idx]} = {mi_scores[idx]:.3f}  {bar}")

# Select top-k features
k_values = [3, 5, 7, 10, 15, 20]
filter_results = []
for k in k_values:
    selector = SelectKBest(mutual_info_classif, k=k)
    X_filtered = selector.fit_transform(X, y)
    scores = cross_val_score(lr_all, X_filtered, y, cv=5, scoring="accuracy")
    filter_results.append(scores.mean())

print(f"\n  Accuracy by number of features (filter):")
for k, acc in zip(k_values, filter_results):
    print(f"    k={k:>2}: {acc:.4f}")

# --- WRAPPER: Recursive Feature Elimination ---
print("\n=== Wrapper Method: RFE with Random Forest ===\n")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=5, step=1)
rfe.fit(X, y)

rfe_selected = [feature_names[i] for i in range(20) if rfe.support_[i]]
rfe_ranking = rfe.ranking_

print(f"  Selected features: {rfe_selected}")
scores_rfe = cross_val_score(lr_all, X[:, rfe.support_], y, cv=5, scoring="accuracy")
print(f"  Accuracy (RFE top-5): {scores_rfe.mean():.4f}\n")

print("  Full RFE ranking:")
for idx in np.argsort(rfe_ranking):
    marker = " <-- selected" if rfe.support_[idx] else ""
    print(f"    {feature_names[idx]}: rank {rfe_ranking[idx]}{marker}")

# --- EMBEDDED: Lasso (L1 regularization) ---
print("\n=== Embedded Method: Lasso ===\n")
from sklearn.linear_model import LogisticRegression as LR

lasso_lr = LR(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=42)
lasso_lr.fit(X, y)
coefs = lasso_lr.coef_[0]

nonzero = np.where(np.abs(coefs) > 1e-6)[0]
lasso_selected = [feature_names[i] for i in nonzero]
print(f"  Non-zero coefficients: {len(nonzero)} / 20")
print(f"  Selected features: {lasso_selected}\n")

sorted_coef_idx = np.argsort(np.abs(coefs))[::-1]
for idx in sorted_coef_idx:
    if abs(coefs[idx]) > 1e-6:
        bar = "|" * int(abs(coefs[idx]) * 20)
        print(f"    {feature_names[idx]}: {coefs[idx]:+.4f}  {bar}")

scores_lasso = cross_val_score(lr_all, X[:, nonzero], y, cv=5, scoring="accuracy")
print(f"\n  Accuracy (Lasso-selected): {scores_lasso.mean():.4f}")

# --- Summary comparison ---
print("\n=== Summary ===\n")
print(f"  All 20 features:     {scores_all.mean():.4f}")
print(f"  Filter (top-5):      {filter_results[1]:.4f}")
print(f"  RFE (top-5):         {scores_rfe.mean():.4f}")
print(f"  Lasso (auto):        {scores_lasso.mean():.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# MI scores
axes[0].barh(
    [feature_names[i] for i in mi_ranking],
    [mi_scores[i] for i in mi_ranking],
    color=["coral" if mi_scores[i] > np.median(mi_scores) else "steelblue"
           for i in mi_ranking]
)
axes[0].set_xlabel("Mutual Information")
axes[0].set_title("Filter: Mutual Information Scores")
axes[0].invert_yaxis()

# Accuracy vs k
axes[1].plot(k_values, filter_results, 'bo-', markersize=6)
axes[1].axhline(scores_all.mean(), color='r', linestyle='--', label="All features")
axes[1].set_xlabel("Number of Features (k)")
axes[1].set_ylabel("CV Accuracy")
axes[1].set_title("Filter: Accuracy vs k")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Lasso coefficients
axes[2].barh(
    [feature_names[i] for i in sorted_coef_idx],
    [coefs[i] for i in sorted_coef_idx],
    color=["coral" if abs(coefs[i]) > 1e-6 else "lightgray"
           for i in sorted_coef_idx]
)
axes[2].set_xlabel("Coefficient")
axes[2].set_title("Embedded: Lasso Coefficients")
axes[2].axvline(0, color="k", linewidth=0.5)
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("feature_selection.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to feature_selection.png")
```

---

## Key Takeaways

- **Feature selection removes irrelevant and redundant features.** Fewer features often means better generalization, faster training, and easier interpretation.
- **Filter methods are fast but blind to interactions.** Mutual information and correlation score features independently -- good as a first pass, not as a final answer.
- **Wrapper methods like RFE are accurate but expensive.** They evaluate feature subsets using actual model performance, capturing interactions at the cost of computational time.
- **Embedded methods (Lasso, tree importance) are the practical sweet spot.** They perform selection during training, balancing speed and quality.
- **Always cross-validate feature selection.** Selecting features on the full training set and then evaluating on the same data overestimates performance. The selection step must be inside the cross-validation loop.
