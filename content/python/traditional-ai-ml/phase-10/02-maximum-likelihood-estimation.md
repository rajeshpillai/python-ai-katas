# Maximum Likelihood Estimation

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.2

---

## Concept & Intuition

### What problem are we solving?

Given a statistical model and observed data, we need to find the parameter values that make the data most probable. **Maximum Likelihood Estimation (MLE)** does exactly this: it finds the parameters theta that maximize the **likelihood function** P(data | theta) — the probability of observing our data given those parameters.

MLE is the workhorse of classical statistics and connects deeply to machine learning. When you minimize mean squared error in linear regression, you are doing MLE under a Gaussian noise assumption. When you minimize cross-entropy in logistic regression, you are doing MLE under a Bernoulli model. Understanding MLE reveals that loss functions are not arbitrary — they emerge naturally from probabilistic assumptions about the data.

The practical procedure is: write down the likelihood (or more often the **log-likelihood**, since logarithms turn products into sums), take the derivative with respect to the parameters, set it to zero, and solve. For simple models this gives closed-form solutions. For complex models, we use gradient ascent (or equivalently, gradient descent on the negative log-likelihood).

### Why naive approaches fail

Without a principled estimation method, you might guess parameters, use heuristics, or try to minimize some ad-hoc error metric. MLE provides a principled, consistent, and asymptotically efficient estimator — meaning that with enough data, no other estimator can do better. However, MLE has no built-in regularization: with small samples, it can overfit (recall the 3-heads-in-3-flips example from the Bayesian thinking kata, where MLE gives P = 1.0).

### Mental models

- **MLE asks: "Which parameter makes my data least surprising?"** It maximizes the probability of what we actually observed.
- **Negative log-likelihood is a loss function**: Minimizing NLL is equivalent to maximizing likelihood. This is why cross-entropy loss = NLL for classification.
- **MSE = MLE + Gaussian noise**: If you assume y = f(x) + Gaussian noise, maximizing likelihood gives you the same answer as minimizing mean squared error. The connection is exact, not approximate.
- **Log-likelihood turns products into sums**: Since data points are independent, the total likelihood is a product. The log turns this into a sum, which is numerically stable and easier to differentiate.

### Visual explanations

```
MLE for a Gaussian (finding mean and variance):

  Data: [2.1, 3.5, 2.8, 3.1, 2.9]

  Likelihood: L(mu, sigma) = product of N(x_i | mu, sigma)
  Log-likelihood: LL = sum of log N(x_i | mu, sigma)
                     = -n/2 * log(2*pi*sigma^2) - sum((x_i - mu)^2) / (2*sigma^2)

  Setting dLL/d(mu) = 0:
    mu_MLE = mean(data) = 2.88

  Setting dLL/d(sigma^2) = 0:
    sigma^2_MLE = mean((x_i - mu)^2) = 0.176

  Likelihood surface:
        sigma
          |
    0.8   |  .  .  .  .  .
    0.6   |  .  .  *  .  .    * = MLE (peak)
    0.4   |  .  * *** *  .
    0.2   |  .  .  *  .  .
          +--+--+--+--+---> mu
           2.0 2.5 3.0 3.5

Connection to loss functions:
  Model        Noise assumption    MLE loss = NLL
  ─────        ────────────────    ───────────────
  Linear reg.  Gaussian            MSE
  Logistic     Bernoulli           Binary cross-entropy
  Poisson      Poisson             Poisson deviance
```

---

## Hands-on Exploration

1. Generate 20 samples from a Gaussian with known mu and sigma. Derive the MLE formulas by hand (take derivative of log-likelihood, set to zero). Verify with numpy.
2. Plot the log-likelihood surface for different values of mu and sigma. Find the peak — it should match your MLE estimates.
3. For logistic regression, write the log-likelihood and show it equals negative binary cross-entropy. Confirm that sklearn's LogisticRegression minimizes the same thing.
4. Generate data from a biased coin. Compare the MLE (sample proportion) with small samples (n=5) vs. large samples (n=1000). When does MLE become reliable?

---

## Live Code

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, minimize

np.random.seed(42)

# --- MLE for Gaussian Distribution ---
true_mu = 5.0
true_sigma = 2.0

# @param n_samples int 5 500 50
n_samples = 50

data = np.random.normal(true_mu, true_sigma, size=n_samples)

# Analytical MLE
mu_mle = np.mean(data)
sigma2_mle = np.mean((data - mu_mle) ** 2)  # MLE uses 1/n, not 1/(n-1)
sigma_mle = np.sqrt(sigma2_mle)

print("=== MLE for Gaussian Distribution ===")
print(f"True parameters:  mu={true_mu}, sigma={true_sigma}")
print(f"MLE estimates:    mu={mu_mle:.4f}, sigma={sigma_mle:.4f}")
print(f"Sample size:      {n_samples}\n")

# --- Log-Likelihood function ---
def log_likelihood_gaussian(mu, sigma, data):
    """Compute log-likelihood of data under N(mu, sigma)."""
    n = len(data)
    ll = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)
    return ll

ll_at_mle = log_likelihood_gaussian(mu_mle, sigma_mle, data)
ll_at_true = log_likelihood_gaussian(true_mu, true_sigma, data)

print(f"Log-likelihood at MLE:        {ll_at_mle:.4f}")
print(f"Log-likelihood at true params: {ll_at_true:.4f}")
print(f"MLE >= true? {ll_at_mle >= ll_at_true - 1e-10}  (MLE maximizes likelihood by definition)\n")

# --- Log-likelihood profile over mu ---
print("=== Log-Likelihood Profile (varying mu, sigma fixed at MLE) ===")
mus = np.linspace(mu_mle - 3, mu_mle + 3, 7)
print(f"{'mu':>8}  {'LL':>12}  {'bar':>30}")
max_ll = ll_at_mle
for mu in mus:
    ll = log_likelihood_gaussian(mu, sigma_mle, data)
    bar_len = max(0, int(30 + (ll - max_ll) * 2))
    marker = " <-- MLE" if abs(mu - mu_mle) < 0.5 else ""
    print(f"{mu:>8.2f}  {ll:>12.2f}  {'#' * bar_len}{marker}")

# --- MLE = MSE Connection ---
print("\n=== MLE-MSE Connection (Linear Regression) ===")
# Generate linear data with Gaussian noise
n_reg = 100
X = np.random.uniform(0, 10, n_reg)
y = 3 * X + 7 + np.random.normal(0, 2, n_reg)

# MLE via log-likelihood maximization
def neg_log_likelihood_linreg(params, X, y):
    m, b, log_sigma = params
    sigma = np.exp(log_sigma)
    predictions = m * X + b
    residuals = y - predictions
    nll = len(X)/2 * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / (2 * sigma**2)
    return nll

result = minimize(neg_log_likelihood_linreg, x0=[1, 1, 0], args=(X, y), method='Nelder-Mead')
m_mle, b_mle, log_sigma_mle = result.x
sigma_mle_reg = np.exp(log_sigma_mle)

# MSE solution (closed-form OLS)
X_design = np.column_stack([X, np.ones(n_reg)])
beta_ols = np.linalg.lstsq(X_design, y, rcond=None)[0]
m_ols, b_ols = beta_ols

print(f"True parameters:   m=3.0, b=7.0, sigma=2.0")
print(f"MLE (likelihood):  m={m_mle:.4f}, b={b_mle:.4f}, sigma={sigma_mle_reg:.4f}")
print(f"OLS (MSE):         m={m_ols:.4f}, b={b_ols:.4f}")
print(f"Slope match:       {abs(m_mle - m_ols) < 0.01}")
print("=> MLE with Gaussian noise gives the same slope/intercept as MSE!")

# --- MLE for Bernoulli (coin flip) ---
print("\n=== MLE for Bernoulli (Coin Flip) ===")
true_p = 0.7

sample_sizes = [5, 10, 30, 100, 500]
print(f"True P(heads): {true_p}\n")
print(f"{'n':>6}  {'Heads':>6}  {'MLE':>8}  {'|MLE - true|':>13}")
print("-" * 40)
for n in sample_sizes:
    flips = np.random.binomial(1, true_p, size=n)
    p_mle = np.mean(flips)
    print(f"{n:>6}  {np.sum(flips):>6}  {p_mle:>8.4f}  {abs(p_mle - true_p):>13.4f}")

# --- MLE = Cross-Entropy Connection ---
print("\n=== MLE-Cross-Entropy Connection (Logistic Regression) ===")
# Binary classification data
from sklearn.linear_model import LogisticRegression

n_cls = 200
X_cls = np.random.randn(n_cls, 1) * 2
true_w, true_b = 1.5, -0.5
prob = 1 / (1 + np.exp(-(true_w * X_cls.ravel() + true_b)))
y_cls = np.random.binomial(1, prob)

# Sklearn logistic regression (minimizes cross-entropy = NLL)
lr = LogisticRegression(penalty=None, solver='lbfgs')
lr.fit(X_cls, y_cls)

# Manual MLE via NLL minimization
def nll_logistic(params, X, y):
    w, b = params
    z = w * X.ravel() + b
    # Numerically stable log-likelihood
    ll = np.sum(y * z - np.log(1 + np.exp(z)))
    return -ll  # negative because we minimize

result = minimize(nll_logistic, x0=[0, 0], args=(X_cls, y_cls), method='Nelder-Mead')
w_mle, b_mle = result.x

print(f"True parameters:     w={true_w}, b={true_b}")
print(f"Sklearn (CE loss):   w={lr.coef_[0][0]:.4f}, b={lr.intercept_[0]:.4f}")
print(f"Manual MLE (NLL):    w={w_mle:.4f}, b={b_mle:.4f}")
print(f"Match: {abs(lr.coef_[0][0] - w_mle) < 0.05}")
print("\n=> Minimizing cross-entropy IS maximum likelihood estimation!")
```

---

## Key Takeaways

- **MLE finds parameters that maximize the probability of observed data.** It answers: "What parameter values make my data least surprising?"
- **Log-likelihood is preferred over likelihood for numerical stability.** The log turns products into sums and does not change the location of the maximum.
- **Common loss functions are negative log-likelihoods in disguise.** MSE assumes Gaussian noise; cross-entropy assumes Bernoulli outcomes. This is not a coincidence — it is the probabilistic foundation of machine learning.
- **MLE is consistent and asymptotically efficient.** With enough data, MLE converges to the true parameters and achieves the lowest possible variance among unbiased estimators.
- **MLE has no built-in regularization.** With small samples, it can overfit badly (e.g., estimating P = 1.0 from 3 heads in 3 flips). This motivates MAP estimation and Bayesian methods.
