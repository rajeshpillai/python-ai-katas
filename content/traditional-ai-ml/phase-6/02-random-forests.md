# Random Forests

> Phase 6 — Ensemble Methods | Kata 6.2

---

## Concept & Intuition

### What problem are we solving?

Bagging helps, but there is a subtle problem: if one feature is very strong, every bootstrap tree will split on it first. This means all your trees look similar and their errors are correlated. Averaging correlated predictions gives less variance reduction than averaging independent ones. Random Forests fix this by adding a second layer of randomness: at each split, only a random subset of features is considered.

By forcing trees to sometimes ignore the best feature and find alternative splitting strategies, Random Forests produce a more diverse set of trees. This decorrelation is the key innovation over plain bagging. The result is that the ensemble benefits more from averaging, achieving lower error with the same number of trees.

Random Forests also provide a natural measure of feature importance (how much each feature contributes to reducing prediction error across all trees) and out-of-bag (OOB) error estimation (a free validation score computed from the ~37% of data each tree never sees during training).

### Why naive approaches fail

Plain bagging with unrestricted trees tends to produce similar-looking trees because the strongest feature dominates the first few splits everywhere. You end up with 100 trees that are 80% identical in structure. The effective ensemble size is much smaller than the actual tree count. Random feature selection at each split breaks this pattern and forces the ensemble to explore the full feature space, leading to genuinely different trees with more independent errors.

### Mental models

- **Diverse committee**: instead of letting every committee member read the same briefing document, you give each member a random subset of documents. They form different opinions and the vote becomes more informative.
- **Feature exploration**: by sometimes blocking the "obvious" feature, the forest discovers secondary patterns that a single tree would never find.
- **Free lunch (almost)**: OOB error gives you a validation score without setting aside any data -- every tree automatically has a held-out set.

### Visual explanations

```
Bagging (all features available):         Random Forest (random feature subsets):

Tree 1: split on F3 > 5                   Tree 1: [F1, F4, F7] --> split on F4
Tree 2: split on F3 > 5                   Tree 2: [F2, F3, F5] --> split on F3
Tree 3: split on F3 > 5                   Tree 3: [F1, F6, F3] --> split on F6
  (all trees look similar)                   (trees are genuinely different)

Correlation between trees: HIGH            Correlation between trees: LOW
Variance reduction: MODERATE               Variance reduction: HIGH

Feature Importance (from Random Forest):
  F3  ||||||||||||||||||||  0.42
  F4  ||||||||||||||        0.28
  F6  ||||||||              0.15
  F1  |||||                 0.09
  F2  |||                   0.06
```

---

## Hands-on Exploration

1. Train a Random Forest on a classification dataset and compare its accuracy to a single decision tree and to a plain bagging ensemble.
2. Extract and visualize feature importances. Identify which features the forest considers most predictive.
3. Compute the OOB error score and compare it to the cross-validation score -- they should be very close.
4. Vary `n_estimators` from 10 to 200 and plot how both OOB error and test error converge, demonstrating diminishing returns.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

np.random.seed(42)

# @param n_estimators int 10 200 100

# --- Generate dataset with informative and noisy features ---
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, n_clusters_per_class=2, random_state=42
)
feature_names = [f"F{i}" for i in range(X.shape[1])]

# --- Single tree baseline ---
tree = DecisionTreeClassifier(random_state=42)
tree_scores = cross_val_score(tree, X, y, cv=5, scoring="accuracy")

# --- Bagging (all features at each split) ---
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=n_estimators, random_state=42
)
bag_scores = cross_val_score(bagging, X, y, cv=5, scoring="accuracy")

# --- Random Forest (random feature subset at each split) ---
rf = RandomForestClassifier(
    n_estimators=n_estimators, oob_score=True, random_state=42
)
rf.fit(X, y)
rf_scores = cross_val_score(
    RandomForestClassifier(n_estimators=n_estimators, random_state=42),
    X, y, cv=5, scoring="accuracy"
)

print("=== Accuracy Comparison ===")
print(f"Single tree:     {tree_scores.mean():.4f} (+/- {tree_scores.std():.4f})")
print(f"Bagging ({n_estimators:>3}):    {bag_scores.mean():.4f} (+/- {bag_scores.std():.4f})")
print(f"Random Forest:   {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
print(f"RF OOB score:    {rf.oob_score_:.4f}  (free, no CV needed)\n")

# --- Feature importance ---
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("=== Feature Importances ===")
for i in sorted_idx:
    bar = "|" * int(importances[i] * 50)
    print(f"  {feature_names[i]:>4}  {bar:<25} {importances[i]:.3f}")

# --- OOB error convergence ---
oob_errors = []
test_range = list(range(10, n_estimators + 1, 5))
for n in test_range:
    rf_temp = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf_temp.fit(X, y)
    oob_errors.append(1 - rf_temp.oob_score_)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Feature importance bar chart
axes[0].barh(
    [feature_names[i] for i in sorted_idx],
    [importances[i] for i in sorted_idx],
    color="steelblue"
)
axes[0].set_xlabel("Importance")
axes[0].set_title("Random Forest Feature Importances")
axes[0].invert_yaxis()

# OOB error convergence
axes[1].plot(test_range, oob_errors, "b-o", markersize=3)
axes[1].set_xlabel("Number of Trees")
axes[1].set_ylabel("OOB Error Rate")
axes[1].set_title("OOB Error vs Number of Trees")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("random_forests.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to random_forests.png")
```

---

## Key Takeaways

- **Random Forests add feature randomness on top of bagging.** At each split, only a random subset of features is considered, decorrelating the trees.
- **Decorrelation is the key.** Less correlated trees means the average benefits more from cancellation of errors, yielding lower variance.
- **OOB error is a free validation metric.** Each tree's out-of-bag samples provide an unbiased error estimate without needing a separate validation set.
- **Feature importances reveal what the data cares about.** The forest naturally ranks features by how much they contribute to prediction quality.
- **More trees never hurts, but returns diminish.** After 50-100 trees, improvement becomes marginal. The computational cost is the only downside.
