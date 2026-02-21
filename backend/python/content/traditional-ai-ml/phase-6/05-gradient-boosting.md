# Gradient Boosting

> Phase 6 — Ensemble Methods | Kata 6.5

---

## Concept & Intuition

### What problem are we solving?

Gradient Boosting generalizes the boosting idea into a powerful framework: instead of reweighting samples like AdaBoost, it directly fits each new tree to the *negative gradient* of the loss function. For squared error loss, the negative gradient is simply the residual (actual minus predicted). For other losses (absolute error, log-loss for classification), the gradient points in the direction of steepest improvement.

Think of it as gradient descent, but instead of updating parameters of a single model, you are adding entire new functions (trees) to the ensemble. Each tree takes a small step in function space toward the optimal prediction. The learning rate controls the step size: smaller steps mean slower convergence but better generalization, because each tree corrects less aggressively and the ensemble has more room to course-correct.

This framework is incredibly flexible. By changing the loss function, you can handle regression, classification, ranking, quantile estimation, and more -- all with the same algorithmic structure. This generality, combined with strong empirical performance, has made gradient boosting the single most important algorithm for tabular data in practice.

### Why naive approaches fail

Fitting residuals directly without a learning rate leads to overfitting: each tree makes a large correction, and after a few rounds the ensemble starts memorizing noise. The learning rate (shrinkage) is essential -- it forces each tree to contribute only a small correction, requiring more trees but producing a smoother, more generalizable ensemble. Without shrinkage, gradient boosting is just as prone to overfitting as a single deep tree.

### Mental models

- **Gradient descent in function space**: instead of adjusting numbers (parameters), you are adjusting the prediction function itself by adding small tree-shaped corrections.
- **Sculptor's refinement**: the first tree carves the rough shape. Each subsequent tree refines the details. The learning rate controls how aggressive each chisel stroke is.
- **Residual stacking**: each tree predicts "what is still unexplained" after all previous trees have had their say.

### Visual explanations

```
Gradient Boosting (regression with MSE loss):

  Step 0: F_0(x) = mean(y)              (initial prediction)
  Step 1: r_1 = y - F_0(x)              (residuals = negative gradient)
          h_1 = tree.fit(X, r_1)        (fit tree to residuals)
          F_1(x) = F_0(x) + lr * h_1(x) (update with small step)
  Step 2: r_2 = y - F_1(x)              (new residuals)
          h_2 = tree.fit(X, r_2)
          F_2(x) = F_1(x) + lr * h_2(x)
  ...
  Step M: F_M(x) = F_0(x) + lr * sum(h_m(x) for m=1..M)

Learning Rate Effect:
  lr = 1.0:  |==========|  (big steps, fast, overfits)
  lr = 0.1:  |=|           (small steps, slow, generalizes)
  lr = 0.01: |             (tiny steps, very slow, very smooth)

  More trees needed with smaller lr, but better final result.
```

---

## Hands-on Exploration

1. Implement a manual gradient boosting loop for regression: start with the mean prediction, compute residuals, fit a shallow tree to them, update predictions with a learning rate, and repeat for 10 rounds.
2. Vary the learning rate from 0.01 to 1.0 and observe how it affects the training curve -- smaller rates need more trees but achieve lower test error.
3. Compare sklearn's GradientBoostingRegressor with different `max_depth` settings (1, 3, 5) to see the interaction between tree complexity and boosting.
4. Plot the loss curve over boosting rounds and identify the optimal stopping point using a validation set.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# @param learning_rate float 0.01 1.0 0.1

# --- Dataset ---
n = 300
X = np.sort(np.random.uniform(0, 6, n)).reshape(-1, 1)
y_true = np.sin(2 * X.ravel()) + 0.3 * X.ravel() ** 1.5
y = y_true + np.random.normal(0, 0.4, n)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_plot = np.linspace(0, 6, 300).reshape(-1, 1)

# --- Manual gradient boosting ---
n_rounds = 100
max_depth = 3

# Initialize with mean
F_train = np.full(len(X_train), y_train.mean())
F_test = np.full(len(X_test), y_train.mean())
F_plot = np.full(300, y_train.mean())

train_mses = []
test_mses = []

print(f"=== Manual Gradient Boosting (lr={learning_rate}) ===\n")
for r in range(n_rounds):
    # Negative gradient (= residuals for MSE loss)
    residuals = y_train - F_train

    # Fit tree to residuals
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X_train, residuals)

    # Update predictions with learning rate
    F_train += learning_rate * tree.predict(X_train)
    F_test += learning_rate * tree.predict(X_test)
    F_plot += learning_rate * tree.predict(X_plot)

    train_mses.append(mean_squared_error(y_train, F_train))
    test_mses.append(mean_squared_error(y_test, F_test))

    if (r + 1) in [1, 5, 10, 25, 50, 100]:
        print(f"  Round {r+1:>3}: train MSE = {train_mses[-1]:.4f}, "
              f"test MSE = {test_mses[-1]:.4f}")

best_round = np.argmin(test_mses) + 1
print(f"\n  Best test MSE at round {best_round}: {min(test_mses):.4f}")

# --- Compare learning rates ---
print(f"\n=== Learning Rate Comparison (100 rounds) ===\n")
lrs = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
lr_results = {}
for lr in lrs:
    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=lr, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)
    test_mse = mean_squared_error(y_test, gb.predict(X_test))
    lr_results[lr] = test_mse
    marker = " <-- current" if abs(lr - learning_rate) < 0.001 else ""
    print(f"  lr = {lr:.2f}: test MSE = {test_mse:.4f}{marker}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Train vs test error over rounds
axes[0].plot(range(1, n_rounds+1), train_mses, 'b-', label="Train MSE", alpha=0.7)
axes[0].plot(range(1, n_rounds+1), test_mses, 'r-', label="Test MSE", alpha=0.7)
axes[0].axvline(best_round, color='g', linestyle='--', label=f"Best round ({best_round})")
axes[0].set_xlabel("Boosting Round")
axes[0].set_ylabel("MSE")
axes[0].set_title(f"Loss Curve (lr={learning_rate})")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Final fit
axes[1].scatter(X_train, y_train, s=8, alpha=0.4, color="gray", label="Train")
axes[1].scatter(X_test, y_test, s=12, alpha=0.6, color="orange", label="Test")
axes[1].plot(X_plot, F_plot, 'r-', linewidth=2, label="GB prediction")
y_plot_true = np.sin(2 * X_plot.ravel()) + 0.3 * X_plot.ravel() ** 1.5
axes[1].plot(X_plot, y_plot_true, 'k--', linewidth=1, label="True function")
axes[1].set_title(f"Gradient Boosting Fit ({n_rounds} rounds)")
axes[1].legend(fontsize=8)

# Learning rate comparison
axes[2].bar([str(lr) for lr in lrs],
            [lr_results[lr] for lr in lrs],
            color=["steelblue" if abs(lr - learning_rate) > 0.001 else "coral" for lr in lrs])
axes[2].set_xlabel("Learning Rate")
axes[2].set_ylabel("Test MSE")
axes[2].set_title("Effect of Learning Rate (100 rounds)")

plt.tight_layout()
plt.savefig("gradient_boosting.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to gradient_boosting.png")
```

---

## Key Takeaways

- **Gradient boosting is gradient descent in function space.** Each tree is a step in the direction that most reduces the loss.
- **The learning rate is the most important hyperparameter.** Smaller learning rates need more trees but generalize better. This is the classic bias-variance tradeoff.
- **Fitting residuals is just a special case.** For MSE loss, residuals equal the negative gradient. For other losses, the gradient is different but the framework is the same.
- **Early stopping prevents overfitting.** Unlike Random Forests where more trees is always safe, gradient boosting will overfit if you add too many trees. Monitor validation loss and stop when it starts increasing.
- **Tree depth controls interaction complexity.** Depth 1 = additive model (no interactions). Depth 2 = pairwise interactions. Depth 3+ = higher-order interactions. Typical values are 3-6.
