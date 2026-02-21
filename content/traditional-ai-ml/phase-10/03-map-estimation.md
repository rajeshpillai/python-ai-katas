# MAP Estimation

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.3

---

## Concept & Intuition

### What problem are we solving?

MLE finds the parameters that maximize the likelihood of the data, but it ignores prior knowledge and can overfit with small samples. **Maximum A Posteriori (MAP) estimation** fixes this by finding the parameters that maximize the **posterior** — the product of likelihood and prior. While full Bayesian inference computes the entire posterior distribution, MAP takes a shortcut: it finds just the single most probable parameter value (the posterior mode).

The key insight is that MAP estimation is equivalent to **regularized MLE**. A Gaussian prior on parameters corresponds to L2 regularization (Ridge regression). A Laplace prior corresponds to L1 regularization (Lasso). This connection is profound: regularization, which practitioners often treat as a tuning trick, has a principled probabilistic interpretation. The regularization strength is the inverse of the prior variance — a strong prior (small variance) means heavy regularization.

MAP sits between MLE and full Bayesian inference. It is computationally as cheap as MLE (just an optimization problem) but incorporates prior information. The trade-off is that it gives a point estimate rather than a full distribution, so it does not capture parameter uncertainty the way full Bayesian inference does.

### Why naive approaches fail

MLE with high-dimensional models and limited data leads to overfitting — the model memorizes noise. Adding regularization helps, but choosing the regularization strength is often ad-hoc. MAP estimation provides a principled framework: specify your prior beliefs about parameter magnitudes, and the regularization strength follows automatically. If you believe parameters are small, use a tight prior (strong regularization). If you have little prior knowledge, use a broad prior (weak regularization).

### Mental models

- **MAP = MLE with a penalty**: The prior adds a penalty term to the negative log-likelihood. Gaussian prior -> L2 penalty. Laplace prior -> L1 penalty.
- **Prior as a rubber band**: It pulls the parameters toward zero (or whatever the prior mean is). The more data you have, the less the rubber band matters.
- **MAP vs. MLE vs. Bayesian**: MLE ignores the prior. MAP finds the peak of the posterior. Full Bayesian computes the entire posterior. Each is progressively more informative but more computationally expensive.

### Visual explanations

```
MAP vs MLE:

  MLE:  argmax_theta  P(data | theta)
  MAP:  argmax_theta  P(data | theta) * P(theta)
        = argmax_theta  log P(data | theta) + log P(theta)
        = argmin_theta  NLL(theta) - log P(theta)
                        ^^^^^^^^^^   ^^^^^^^^^^^^^
                        data fit      regularization

Gaussian prior -> L2 regularization:
  P(theta) = N(0, sigma_prior)
  -log P(theta) = theta^2 / (2 * sigma_prior^2) + const
  MAP loss = NLL + lambda * ||theta||^2   where lambda = 1/(2*sigma_prior^2)

Laplace prior -> L1 regularization:
  P(theta) = Laplace(0, b)
  -log P(theta) = |theta| / b + const
  MAP loss = NLL + lambda * ||theta||_1   where lambda = 1/b

Visual comparison:

  NLL only (MLE):          NLL + L2 (MAP/Ridge):       NLL + L1 (MAP/Lasso):
     |                        |                           |
     |  \      /              |   \    /                  |   \  |  /
     |   \    /               |    \  /                   |    \ | /
     |    \  /                |     \/                    |     \|/
     |     \/                 |     /\                    |     /|\
     |     theta_mle          |   theta_map               |  theta_map
     |                        |  (pulled toward 0)        |  (may be exactly 0)
```

---

## Hands-on Exploration

1. Fit a polynomial regression (degree 10) to 15 data points using MLE (no regularization). Observe overfitting.
2. Add an L2 penalty (Gaussian prior). Watch the fit smooth out. Vary the regularization strength and see how it affects the curve.
3. Derive the MAP estimate for a Gaussian mean with a Gaussian prior. Show that the MAP estimate is a weighted average of the prior mean and the MLE.
4. Compare L1 (Laplace prior) and L2 (Gaussian prior) regularization. Which drives parameters to exactly zero?

---

## Live Code

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# --- Generate polynomial data ---
n_points = 20
X = np.sort(np.random.uniform(-3, 3, n_points))
y_true = np.sin(X) + 0.5 * X  # true underlying function
y = y_true + np.random.normal(0, 0.5, n_points)

# @param degree int 2 15 10
degree = 10
# @param alpha_l2 float 0.001 100.0 1.0
alpha_l2 = 1.0  # L2 regularization strength (Ridge = Gaussian prior)
# @param alpha_l1 float 0.001 100.0 1.0
alpha_l1 = 1.0  # L1 regularization strength (Lasso = Laplace prior)

# Create polynomial features
poly = PolynomialFeatures(degree, include_bias=False)
X_poly = poly.fit_transform(X.reshape(-1, 1))

# Test points for evaluation
X_test = np.linspace(-3, 3, 100)
X_test_poly = poly.transform(X_test.reshape(-1, 1))
y_test_true = np.sin(X_test) + 0.5 * X_test

# --- MLE (no regularization) ---
mle_model = LinearRegression()
mle_model.fit(X_poly, y)
y_pred_mle = mle_model.predict(X_test_poly)
mse_mle = mean_squared_error(y_test_true, y_pred_mle)

# --- MAP with Gaussian prior (Ridge = L2) ---
ridge_model = Ridge(alpha=alpha_l2)
ridge_model.fit(X_poly, y)
y_pred_ridge = ridge_model.predict(X_test_poly)
mse_ridge = mean_squared_error(y_test_true, y_pred_ridge)

# --- MAP with Laplace prior (Lasso = L1) ---
lasso_model = Lasso(alpha=alpha_l1, max_iter=10000)
lasso_model.fit(X_poly, y)
y_pred_lasso = lasso_model.predict(X_test_poly)
mse_lasso = mean_squared_error(y_test_true, y_pred_lasso)

# --- Results ---
print(f"=== MAP Estimation: Regularization as Prior ===")
print(f"Polynomial degree: {degree}, Training points: {n_points}\n")

print(f"{'Method':>25}  {'Train MSE':>10}  {'Test MSE':>10}  {'Coeff Norm':>11}")
print("-" * 62)
for name, model, y_pred in [
    ("MLE (no reg)", mle_model, y_pred_mle),
    (f"MAP/Ridge (alpha={alpha_l2})", ridge_model, y_pred_ridge),
    (f"MAP/Lasso (alpha={alpha_l1})", lasso_model, y_pred_lasso),
]:
    train_pred = model.predict(X_poly)
    train_mse = mean_squared_error(y, train_pred)
    test_mse = mean_squared_error(y_test_true, y_pred)
    coeff_norm = np.sqrt(np.sum(model.coef_ ** 2))
    print(f"{name:>25}  {train_mse:>10.4f}  {test_mse:>10.4f}  {coeff_norm:>11.4f}")

# --- Coefficient comparison ---
print(f"\n=== Coefficient Values (degree {degree} polynomial) ===")
print(f"{'Feature':>10}  {'MLE':>10}  {'Ridge':>10}  {'Lasso':>10}")
print("-" * 45)
feature_names = [f"x^{i+1}" for i in range(degree)]
for i, name in enumerate(feature_names):
    mle_c = mle_model.coef_[i]
    ridge_c = ridge_model.coef_[i]
    lasso_c = lasso_model.coef_[i]
    zero_marker = " *" if abs(lasso_c) < 1e-6 else ""
    print(f"{name:>10}  {mle_c:>10.4f}  {ridge_c:>10.4f}  {lasso_c:>10.4f}{zero_marker}")

n_zero_lasso = np.sum(np.abs(lasso_model.coef_) < 1e-6)
print(f"\nLasso zeros: {n_zero_lasso}/{degree} coefficients (L1 drives params to exactly 0)")

# --- MAP for Gaussian mean (analytical) ---
print("\n=== Analytical MAP: Gaussian Mean with Gaussian Prior ===")
# Prior: mu ~ N(mu_0, sigma_0^2)
# Likelihood: x_i ~ N(mu, sigma^2)  (sigma known)
# MAP: mu_MAP = (sigma^2 * mu_0 + n * sigma_0^2 * x_bar) / (sigma^2 + n * sigma_0^2)

mu_0 = 0.0       # prior mean
sigma_0 = 1.0    # prior std
sigma = 2.0      # known observation noise

data_gauss = np.random.normal(3.0, sigma, size=10)  # true mean = 3.0
x_bar = np.mean(data_gauss)
n = len(data_gauss)

mu_mle = x_bar
mu_map = (sigma**2 * mu_0 + n * sigma_0**2 * x_bar) / (sigma**2 + n * sigma_0**2)

# Weight on prior vs data
w_prior = sigma**2 / (sigma**2 + n * sigma_0**2)
w_data = n * sigma_0**2 / (sigma**2 + n * sigma_0**2)

print(f"Prior:           N({mu_0}, {sigma_0}^2)")
print(f"Data mean:       {x_bar:.4f} (n={n})")
print(f"MLE:             {mu_mle:.4f}")
print(f"MAP:             {mu_map:.4f}")
print(f"Weight on prior: {w_prior:.4f}")
print(f"Weight on data:  {w_data:.4f}")
print(f"\nMAP = {w_prior:.3f} * {mu_0:.1f} + {w_data:.3f} * {x_bar:.3f} = {mu_map:.4f}")
print(f"MAP is a weighted average of the prior mean and the MLE!")

# --- Effect of regularization strength ---
print("\n=== Effect of Regularization Strength (Ridge) ===")
print(f"{'Alpha':>10}  {'Test MSE':>10}  {'Coeff Norm':>11}  {'Prior interpretation':>25}")
print("-" * 62)
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    pred = model.predict(X_test_poly)
    test_mse = mean_squared_error(y_test_true, pred)
    norm = np.sqrt(np.sum(model.coef_ ** 2))
    sigma_prior = np.sqrt(1 / (2 * alpha)) if alpha > 0 else float('inf')
    print(f"{alpha:>10.3f}  {test_mse:>10.4f}  {norm:>11.4f}  sigma_prior={sigma_prior:.3f}")
```

---

## Key Takeaways

- **MAP estimation adds a prior to MLE.** It finds the posterior mode — the single most probable parameter value given both data and prior beliefs.
- **Regularization IS a prior, mathematically.** L2 regularization corresponds to a Gaussian prior; L1 regularization corresponds to a Laplace prior. The regularization strength is inversely related to the prior variance.
- **MAP prevents overfitting by penalizing extreme parameters.** The prior pulls parameters toward reasonable values (often zero), counteracting the tendency of MLE to overfit small datasets.
- **L1 (Lasso) drives parameters to exactly zero; L2 (Ridge) shrinks them toward zero.** This makes L1 useful for feature selection — it automatically identifies which features are unimportant.
- **MAP is a compromise between MLE and full Bayesian inference.** It is computationally cheap (just optimization) but only gives a point estimate, not a full posterior distribution.
