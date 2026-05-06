# Bagging Intuition

> Phase 6 — Ensemble Methods | Kata 6.1

---

## Concept & Intuition

### What problem are we solving?

A single decision tree is unstable: change a few training points and the tree structure can look completely different. This instability means high variance -- individual trees overfit to the noise in whatever sample they see. Bagging (Bootstrap AGGregatING) solves this by training many trees on different random subsets of the data and averaging their predictions.

The core idea is borrowed from statistics: if you have many noisy, independent estimators, their average is far less noisy than any individual one. A single tree might have 40% error variance, but averaging 50 trees cuts that variance dramatically. The individual trees are still overfit, but they overfit to *different* noise, so the errors cancel out.

Bagging works best when the base learner is powerful but unstable. Decision trees are the perfect candidate because they can capture complex patterns (low bias) but fluctuate wildly with different data (high variance). By aggregating many such trees, you keep the expressiveness while taming the instability.

### Why naive approaches fail

You might think: "Why not just build one really good tree?" The problem is that pruning a tree to reduce overfitting also reduces its ability to capture real patterns. You are trading variance for bias. Bagging offers a way to reduce variance *without* increasing bias -- you keep the deep, expressive trees but average away their individual quirks. Training the same tree multiple times on the same data is useless; the bootstrap sampling step is what creates the diversity needed for the average to work.

### Mental models

- **Wisdom of crowds**: ask 100 people to guess the weight of an ox. Individual guesses are noisy, but the average is remarkably accurate -- as long as the errors are not all in the same direction.
- **Parallel committee**: each tree is an expert trained on a slightly different version of reality. The committee vote is more reliable than any single member.
- **Noise cancellation**: each tree's error is partly signal (real pattern) and partly noise (random fluctuation). Averaging preserves the signal and cancels the noise.

### Visual explanations

```
Original Data: [x1, x2, x3, x4, x5, x6, x7, x8]

Bootstrap Sample 1: [x2, x2, x5, x1, x7, x3, x3, x8]  --> Tree 1 --> pred_1
Bootstrap Sample 2: [x4, x1, x6, x6, x3, x8, x5, x2]  --> Tree 2 --> pred_2
Bootstrap Sample 3: [x7, x3, x1, x5, x5, x2, x8, x4]  --> Tree 3 --> pred_3
  ...                                                        ...
Bootstrap Sample B: [x3, x8, x2, x2, x6, x1, x4, x7]  --> Tree B --> pred_B

Final prediction = average(pred_1, pred_2, ..., pred_B)

Variance Reduction:
  Single tree:    Var(pred)
  Bagged (B):     Var(pred) / B   (if trees were independent)
  In practice:    somewhere in between (trees are correlated)
```

---

## Hands-on Exploration

1. Generate a noisy regression dataset with a known nonlinear function. Fit a single deep decision tree and observe its jagged, overfit predictions.
2. Manually create 10 bootstrap samples by sampling with replacement. Train a separate tree on each and plot their individual predictions -- notice how they differ wildly.
3. Average all 10 tree predictions together and compare the smooth ensemble prediction against the single tree. Measure MSE for both.
4. Increase the number of trees from 1 to 100 and plot how the test MSE decreases and stabilizes, showing diminishing returns after a certain point.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# --- True function + noisy data ---
def true_fn(x):
    return np.sin(2 * x) + 0.5 * np.cos(5 * x)

n_train = 80
X_train = np.sort(np.random.uniform(0, 4, n_train)).reshape(-1, 1)
y_train = true_fn(X_train.ravel()) + np.random.normal(0, 0.3, n_train)

X_test = np.linspace(0, 4, 300).reshape(-1, 1)
y_true = true_fn(X_test.ravel())

# --- Single deep tree (overfits) ---
single_tree = DecisionTreeRegressor(max_depth=None, random_state=0)
single_tree.fit(X_train, y_train)
pred_single = single_tree.predict(X_test)

# --- Manual bagging ---
n_bags = 50
bag_preds = np.zeros((n_bags, len(X_test)))

for i in range(n_bags):
    # Bootstrap sample: sample with replacement
    indices = np.random.randint(0, n_train, size=n_train)
    X_bag = X_train[indices]
    y_bag = y_train[indices]

    tree = DecisionTreeRegressor(max_depth=None)
    tree.fit(X_bag, y_bag)
    bag_preds[i] = tree.predict(X_test)

pred_bagged = bag_preds.mean(axis=0)

# --- MSE comparison ---
mse_single = np.mean((pred_single - y_true) ** 2)
mse_bagged = np.mean((pred_bagged - y_true) ** 2)

print(f"Single tree MSE:  {mse_single:.4f}")
print(f"Bagged ({n_bags} trees) MSE: {mse_bagged:.4f}")
print(f"Variance reduction: {(1 - mse_bagged / mse_single) * 100:.1f}%\n")

# --- Effect of number of bags on MSE ---
bag_counts = [1, 3, 5, 10, 20, 30, 50]
mses = []
for b in bag_counts:
    pred_b = bag_preds[:b].mean(axis=0)
    mses.append(np.mean((pred_b - y_true) ** 2))
    print(f"  {b:>3} trees --> MSE = {mses[-1]:.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].scatter(X_train, y_train, s=10, alpha=0.5, label="Training data")
axes[0].plot(X_test, pred_single, 'r-', linewidth=1.5, label="Single tree")
axes[0].plot(X_test, y_true, 'k--', linewidth=1, label="True function")
axes[0].set_title(f"Single Tree (MSE={mse_single:.4f})")
axes[0].legend(fontsize=8)

axes[1].scatter(X_train, y_train, s=10, alpha=0.5, label="Training data")
axes[1].plot(X_test, pred_bagged, 'b-', linewidth=1.5, label=f"Bagged ({n_bags} trees)")
axes[1].plot(X_test, y_true, 'k--', linewidth=1, label="True function")
axes[1].set_title(f"Bagged Trees (MSE={mse_bagged:.4f})")
axes[1].legend(fontsize=8)

axes[2].plot(bag_counts, mses, 'go-')
axes[2].axhline(mse_single, color='r', linestyle='--', label="Single tree")
axes[2].set_xlabel("Number of bags")
axes[2].set_ylabel("MSE")
axes[2].set_title("MSE vs Number of Trees")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("bagging_intuition.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to bagging_intuition.png")
```

---

## Key Takeaways

- **Bagging reduces variance by averaging many overfit models.** Each tree is intentionally deep and overfit, but their average is stable.
- **Bootstrap sampling creates diversity.** Sampling with replacement means each tree sees a different version of the data, so they make different errors.
- **More trees is always better (with diminishing returns).** MSE drops rapidly at first, then plateaus. There is no overfitting risk from adding more trees.
- **Bagging does not reduce bias.** If each individual tree is biased (e.g., too shallow), averaging will not fix that. The base learner must be expressive.
- **The ~37% of data left out of each bootstrap sample (out-of-bag) can be used for free validation.** This is a bonus of the bootstrap procedure.
