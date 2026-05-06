# XGBoost

> Phase 6 — Ensemble Methods | Kata 6.6

---

## Concept & Intuition

### What problem are we solving?

XGBoost (eXtreme Gradient Boosting) takes the gradient boosting framework and makes it fast, scalable, and regularized. While the core idea is the same -- sequentially fit trees to negative gradients -- XGBoost adds several critical innovations: L1 and L2 regularization on leaf weights to prevent overfitting, a more sophisticated split-finding algorithm using histogram binning for speed, built-in handling of missing values, and column subsampling (borrowing the Random Forest trick).

The regularization in XGBoost is particularly important. Standard gradient boosting controls complexity only through tree depth and learning rate. XGBoost adds explicit penalties on the number of leaves (gamma) and the magnitude of leaf predictions (lambda for L2, alpha for L1). This means each tree is itself regularized, not just the ensemble. The result is better generalization with fewer trees.

XGBoost became the dominant algorithm in machine learning competitions from 2014 onward and remains the go-to algorithm for tabular data in industry. Its success comes from the combination of strong theoretical foundations, engineering optimizations (parallel tree building, cache-aware computation), and practical features (early stopping, custom objectives, GPU support).

### Why naive approaches fail

Vanilla gradient boosting grows full trees and relies solely on the learning rate and tree depth for regularization. This often leads to overly complex individual trees with extreme leaf predictions. XGBoost's regularization penalizes large leaf weights, producing smoother, more conservative predictions per tree. Additionally, standard implementations are slow because they evaluate every possible split point; XGBoost's histogram binning reduces this to a manageable number of bins, making it orders of magnitude faster on large datasets.

### Mental models

- **Regularized gradient descent**: not only do you take small steps (learning rate), but each step is also penalized for being too extreme (L2 regularization on leaf values).
- **Efficient search**: instead of checking every possible split value, group values into histogram bins and evaluate bin boundaries. Same quality, fraction of the cost.
- **Missing data as information**: XGBoost learns which direction to send missing values at each split, treating missingness as a potentially useful signal.

### Visual explanations

```
XGBoost Objective (for each tree):

  Obj = Sum(Loss(y_i, pred_i)) + gamma * num_leaves + lambda/2 * Sum(w_j^2)
        |________________________|   |______________|   |___________________|
           Data fit                   Leaf count          Leaf weight
           (minimize error)           penalty             penalty (L2)

Histogram Binning:
  Raw feature values:    1.2  3.7  0.5  2.1  4.8  1.9  3.3  ...  (N values)
  Binned:                [0.0-1.0] [1.0-2.0] [2.0-3.0] [3.0-4.0] [4.0-5.0]
  Split candidates:      Only 4 boundaries to evaluate (vs N-1 without binning)

Missing Value Handling:
  Node split: Feature F3 < 2.5 ?
    Left:  F3 < 2.5              (includes missing values if that's better)
    Right: F3 >= 2.5             (or missing goes here if that's better)
    --> XGBoost tries BOTH and picks the direction that reduces loss more
```

---

## Hands-on Exploration

1. Train an XGBoost model and a standard GradientBoostingRegressor on the same dataset. Compare their test performance and training time.
2. Explore the regularization parameters: vary lambda (L2) and gamma (min split loss) and observe how they affect model complexity and test error.
3. Introduce missing values into the dataset and verify that XGBoost handles them automatically without imputation.
4. Use XGBoost's built-in feature importance (gain, cover, weight) and compare the three different importance metrics.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import time

np.random.seed(42)

# --- Dataset ---
X, y = make_regression(
    n_samples=1000, n_features=15, n_informative=8,
    noise=20, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Try to import xgboost; fall back to sklearn's HistGradientBoosting ---
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor
    print("XGBoost not installed. Using sklearn's HistGradientBoostingRegressor")
    print("(pip install xgboost to get the full experience)\n")

# --- Standard Gradient Boosting baseline ---
start = time.time()
gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
)
gb.fit(X_train, y_train)
gb_time = time.time() - start
gb_pred = gb.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_pred)

print("=== Standard Gradient Boosting ===")
print(f"  Test MSE: {gb_mse:.2f}, Time: {gb_time:.3f}s\n")

# --- XGBoost / HistGradientBoosting ---
if HAS_XGB:
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)

    print("=== XGBoost ===")
    print(f"  Test MSE: {xgb_mse:.2f}, Time: {xgb_time:.3f}s")
    print(f"  Speedup: {gb_time / xgb_time:.1f}x faster\n")

    # --- Regularization sweep ---
    print("=== Regularization Effect (lambda / L2 penalty) ===\n")
    lambdas = [0, 0.1, 1, 5, 10, 50]
    reg_mses = []
    for lam in lambdas:
        model = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            reg_lambda=lam, random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        reg_mses.append(mse)
        print(f"  lambda={lam:<5}: test MSE = {mse:.2f}")

    # --- Feature importance ---
    importance = xgb_model.feature_importances_
    feat_names = [f"F{i}" for i in range(X.shape[1])]
    sorted_idx = np.argsort(importance)[::-1]

    print("\n=== Feature Importance (gain-based) ===")
    for i in sorted_idx[:10]:
        bar = "|" * int(importance[i] * 50)
        print(f"  {feat_names[i]:>4}: {bar:<25} {importance[i]:.3f}")

    # --- Missing value handling ---
    print("\n=== Missing Value Handling ===")
    X_train_missing = X_train.copy()
    mask = np.random.random(X_train_missing.shape) < 0.1
    X_train_missing[mask] = np.nan
    n_missing = mask.sum()
    print(f"  Introduced {n_missing} missing values "
          f"({mask.mean()*100:.1f}% of training data)")

    X_test_missing = X_test.copy()
    mask_test = np.random.random(X_test_missing.shape) < 0.1
    X_test_missing[mask_test] = np.nan

    model_missing = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        random_state=42, verbosity=0
    )
    model_missing.fit(X_train_missing, y_train)
    mse_missing = mean_squared_error(y_test, model_missing.predict(X_test_missing))
    print(f"  MSE with missing data: {mse_missing:.2f}")
    print(f"  MSE without missing:   {xgb_mse:.2f}")
    print(f"  Degradation: {(mse_missing/xgb_mse - 1)*100:.1f}%")

else:
    start = time.time()
    hgb = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.1, max_depth=4, random_state=42
    )
    hgb.fit(X_train, y_train)
    hgb_time = time.time() - start
    hgb_pred = hgb.predict(X_test)
    hgb_mse = mean_squared_error(y_test, hgb_pred)

    print("=== HistGradientBoosting (sklearn) ===")
    print(f"  Test MSE: {hgb_mse:.2f}, Time: {hgb_time:.3f}s")
    print(f"  Speedup vs standard GB: {gb_time / hgb_time:.1f}x\n")

    lambdas = [0, 0.1, 1, 5, 10, 50]
    reg_mses = []
    for lam in lambdas:
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.1, max_depth=4,
            l2_regularization=lam, random_state=42
        )
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        reg_mses.append(mse)
        print(f"  l2_reg={lam:<5}: test MSE = {mse:.2f}")

    importance = np.zeros(X.shape[1])
    xgb_mse = hgb_mse

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Regularization sweep
axes[0].plot(lambdas, reg_mses, 'bo-', markersize=6)
axes[0].set_xlabel("Lambda (L2 regularization)")
axes[0].set_ylabel("Test MSE")
axes[0].set_title("Effect of L2 Regularization")
axes[0].grid(True, alpha=0.3)

# Feature importance
if HAS_XGB:
    axes[1].barh(
        [feat_names[i] for i in sorted_idx],
        [importance[i] for i in sorted_idx],
        color="steelblue"
    )
    axes[1].set_xlabel("Importance (gain)")
    axes[1].set_title("XGBoost Feature Importances")
    axes[1].invert_yaxis()
else:
    axes[1].text(0.5, 0.5, "Install xgboost for\nfeature importance plot",
                 ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title("Feature Importance (requires xgboost)")

plt.tight_layout()
plt.savefig("xgboost.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to xgboost.png")
```

---

## Key Takeaways

- **XGBoost adds regularization to gradient boosting.** L2 (lambda) and L1 (alpha) penalties on leaf weights, plus a penalty on the number of leaves (gamma), prevent overfitting.
- **Histogram binning makes XGBoost fast.** By discretizing features into bins, split-finding becomes much cheaper than evaluating every unique value.
- **Missing values are handled natively.** XGBoost learns the optimal direction for missing values at each split -- no imputation needed.
- **Column subsampling (from Random Forests) further reduces overfitting.** Using a random subset of features per tree decorrelates the trees in the ensemble.
- **Key hyperparameters to tune: learning_rate, max_depth, n_estimators, reg_lambda, subsample, colsample_bytree.** Start with learning_rate=0.1, max_depth=4-6, and tune from there.
