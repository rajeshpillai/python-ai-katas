# Bayesian Optimization

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.5

---

## Concept & Intuition

### What problem are we solving?

Suppose you need to tune hyperparameters for a machine learning model — learning rate, regularization strength, number of layers, etc. Each evaluation is expensive: training the model might take hours. You cannot afford to try thousands of combinations. **Bayesian optimization** is a strategy for finding the optimum of an expensive-to-evaluate function using as few evaluations as possible.

The idea is elegant. First, build a cheap **surrogate model** (typically a Gaussian process) that approximates the expensive function based on the points you have already evaluated. The surrogate gives both a prediction and an uncertainty estimate for every point in the search space. Then, use an **acquisition function** to decide where to evaluate next — balancing exploitation (points where the surrogate predicts good values) with exploration (points where the surrogate is uncertain). Evaluate the expensive function at that point, update the surrogate, and repeat.

This approach is dramatically more efficient than grid search or random search for expensive black-box functions. While random search requires hundreds of evaluations to find good hyperparameters, Bayesian optimization often finds competitive settings in 20-50 evaluations.

### Why naive approaches fail

Grid search scales exponentially with the number of hyperparameters (curse of dimensionality). Random search is better but wasteful — it does not learn from previous evaluations. Each new random point is chosen without considering what was already tried. Bayesian optimization uses every previous evaluation to inform the next choice, focusing effort on the most promising regions of the search space.

### Mental models

- **Surrogate model as a map with fog**: Evaluated points are clear; unevaluated regions are foggy. The acquisition function decides whether to explore foggy areas (might be great!) or exploit clear areas (known to be good).
- **Expected Improvement**: "How much better than the current best can I expect this point to be?" High EI means either high predicted value or high uncertainty — naturally balancing exploration and exploitation.
- **Bayesian optimization as strategic tasting**: A wine judge with only 20 tastes to find the best bottle does not taste randomly. They use each taste to narrow down which bottles are worth trying next.

### Visual explanations

```
Bayesian Optimization Loop:

  1. Evaluate f(x) at a few initial points
  2. Fit surrogate model (Gaussian process) to observed data
  3. Compute acquisition function across search space
  4. Evaluate f(x) at the point that maximizes acquisition
  5. Go to step 2

  Surrogate model (Gaussian Process):
        f(x)
         |
     1.0 |          * (observed)        +---------+
         |     .....*.*....             | shaded  |
     0.5 |  ...     |    ...            | = uncertainty
         | .        |       ....        +---------+
     0.0 |..........|...........*.......
         |          |           * (observed)
    -0.5 |
         +--+--+--+--+--+--+--+--+---> x

  Acquisition function (Expected Improvement):
        EI(x)
         |
     0.3 |              *
         |            *   *
     0.2 |          *       *
         |        *           *
     0.1 |      *               *
         |    *                   *
     0.0 |***                       ***
         +--+--+--+--+--+--+--+--+---> x
                    ^
                    Next evaluation point
                    (max EI = high uncertainty OR high predicted value)

Comparison (20 evaluations budget):
  Grid search:  evaluates fixed grid, misses optima between grid points
  Random search: evaluates random points, no learning between evaluations
  Bayesian opt:  each point informed by all previous, converges faster
```

---

## Hands-on Exploration

1. Define a 1D function with multiple local optima (e.g., f(x) = -sin(3x) - x^2 + 0.7x). Evaluate it at 3 random points.
2. Fit a Gaussian process to those 3 points. Plot the mean prediction and uncertainty band. Where is the GP most uncertain?
3. Compute Expected Improvement across the search space. Where does it recommend evaluating next? Is it exploring or exploiting?
4. Run the full Bayesian optimization loop for 15 iterations. Compare the result to 15 random evaluations. Which finds a better optimum?

---

## Live Code

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize_scalar

np.random.seed(42)

# --- Expensive black-box function to optimize ---
def objective(x):
    """A multi-modal function (pretend this is expensive to evaluate)."""
    return -(np.sin(3 * x) + np.sin(5 * x) * 0.5 + 0.1 * x ** 2 - 0.5 * x)

# True optimum (found by dense evaluation — we would not do this in practice)
x_dense = np.linspace(-2, 6, 10000)
y_dense = objective(x_dense)
true_opt_x = x_dense[np.argmin(y_dense)]
true_opt_y = np.min(y_dense)

# @param n_iterations int 5 40 20
n_iterations = 20
# @param n_initial int 2 10 3
n_initial = 3

search_min, search_max = -2.0, 6.0

# --- Acquisition Function: Expected Improvement ---
def expected_improvement(X_candidates, gp, y_best, xi=0.01):
    """Compute Expected Improvement for candidate points."""
    mu, sigma = gp.predict(X_candidates.reshape(-1, 1), return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    Z = (y_best - mu) / sigma  # Note: we minimize, so improvement = y_best - mu
    ei = (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-8] = 0.0
    return ei

# --- Bayesian Optimization Loop ---
# Initialize with random points
X_observed = np.random.uniform(search_min, search_max, n_initial)
y_observed = np.array([objective(x) for x in X_observed])

print(f"=== Bayesian Optimization ===")
print(f"Search space: [{search_min}, {search_max}]")
print(f"True optimum: x={true_opt_x:.4f}, f(x)={true_opt_y:.4f}")
print(f"Initial points: {n_initial}, Iterations: {n_iterations}\n")

print(f"{'Iter':>5}  {'x_new':>8}  {'f(x_new)':>10}  {'Best f(x)':>10}  {'EI_max':>8}  {'Type':>8}")
print("-" * 58)

# Show initial points
for i, (x, y) in enumerate(zip(X_observed, y_observed)):
    best_so_far = np.min(y_observed[:i+1])
    print(f"{'init':>5}  {x:>8.4f}  {y:>10.4f}  {best_so_far:>10.4f}  {'---':>8}  {'init':>8}")

# Optimization loop
for iteration in range(n_iterations):
    # Fit Gaussian Process surrogate
    kernel = Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5,
                                   random_state=iteration)
    gp.fit(X_observed.reshape(-1, 1), y_observed)

    # Find point that maximizes Expected Improvement
    X_candidates = np.linspace(search_min, search_max, 500)
    y_best = np.min(y_observed)
    ei_values = expected_improvement(X_candidates, gp, y_best)

    # Select next point
    x_next = X_candidates[np.argmax(ei_values)]
    ei_max = np.max(ei_values)

    # Evaluate the expensive function
    y_next = objective(x_next)

    # Determine if this was exploration or exploitation
    mu_at_next, sigma_at_next = gp.predict([[x_next]], return_std=True)
    action = "explore" if sigma_at_next[0] > 0.3 else "exploit"

    # Update observations
    X_observed = np.append(X_observed, x_next)
    y_observed = np.append(y_observed, y_next)

    best_so_far = np.min(y_observed)
    if iteration % max(1, n_iterations // 10) == 0 or iteration == n_iterations - 1:
        print(f"{iteration+1:>5}  {x_next:>8.4f}  {y_next:>10.4f}  {best_so_far:>10.4f}  {ei_max:>8.4f}  {action:>8}")

# --- Compare with Random Search ---
np.random.seed(123)
X_random = np.random.uniform(search_min, search_max, n_initial + n_iterations)
y_random = np.array([objective(x) for x in X_random])

print(f"\n=== Comparison: Bayesian Opt vs Random Search ===")
print(f"Total evaluations: {n_initial + n_iterations}")
print(f"Bayesian Opt best: f(x) = {np.min(y_observed):.4f} at x = {X_observed[np.argmin(y_observed)]:.4f}")
print(f"Random Search best: f(x) = {np.min(y_random):.4f} at x = {X_random[np.argmin(y_random)]:.4f}")
print(f"True optimum:       f(x) = {true_opt_y:.4f} at x = {true_opt_x:.4f}")

# --- Convergence comparison ---
print(f"\n=== Convergence (best found so far) ===")
print(f"{'Evaluations':>12}  {'Bayes Opt':>10}  {'Random':>10}")
print("-" * 35)
checkpoints = [3, 5, 8, 10, 15, n_initial + n_iterations]
for cp in checkpoints:
    if cp <= len(y_observed):
        bo_best = np.min(y_observed[:cp])
        rs_best = np.min(y_random[:cp])
        print(f"{cp:>12}  {bo_best:>10.4f}  {rs_best:>10.4f}")

# --- Show GP surrogate at final state ---
print(f"\n=== Final GP Surrogate (sampled predictions) ===")
X_show = np.linspace(search_min, search_max, 9)
mu_show, std_show = gp.predict(X_show.reshape(-1, 1), return_std=True)
print(f"{'x':>8}  {'GP mean':>8}  {'GP std':>8}  {'True f(x)':>10}")
print("-" * 40)
for x, m, s in zip(X_show, mu_show, std_show):
    true_val = objective(x)
    print(f"{x:>8.2f}  {m:>8.4f}  {s:>8.4f}  {true_val:>10.4f}")

# --- Apply to Hyperparameter Tuning (practical example) ---
print(f"\n=== Practical Example: Hyperparameter Tuning ===")
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X_cls, y_cls = make_classification(n_samples=200, n_features=10, random_state=42)

def evaluate_svm(log_C):
    """Expensive: train SVM with given C and return negative CV accuracy."""
    C = 10 ** log_C
    svm = SVC(C=C, kernel='rbf', gamma='scale')
    scores = cross_val_score(svm, X_cls, y_cls, cv=3, scoring='accuracy')
    return -np.mean(scores)  # negative because we minimize

# Run small Bayesian optimization for C
X_bo = np.random.uniform(-3, 3, 3)  # log10(C) in [-3, 3]
y_bo = np.array([evaluate_svm(x) for x in X_bo])

for i in range(10):
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, random_state=i)
    gp.fit(X_bo.reshape(-1, 1), y_bo)
    X_cand = np.linspace(-3, 3, 200)
    ei = expected_improvement(X_cand, gp, np.min(y_bo))
    x_next = X_cand[np.argmax(ei)]
    y_next = evaluate_svm(x_next)
    X_bo = np.append(X_bo, x_next)
    y_bo = np.append(y_bo, y_next)

best_idx = np.argmin(y_bo)
best_C = 10 ** X_bo[best_idx]
best_acc = -y_bo[best_idx]
print(f"Best C found: {best_C:.4f} (log10(C) = {X_bo[best_idx]:.2f})")
print(f"Best CV accuracy: {best_acc:.4f}")
print(f"Total evaluations: {len(y_bo)}")
```

---

## Key Takeaways

- **Bayesian optimization finds optima of expensive functions with minimal evaluations.** It is ideal when each function evaluation costs significant time or resources.
- **The surrogate model (Gaussian process) provides predictions AND uncertainty.** This dual output is what enables intelligent exploration-exploitation trade-offs.
- **Acquisition functions like Expected Improvement balance exploration and exploitation.** They recommend points that are either predicted to be good or are highly uncertain.
- **Bayesian optimization dramatically outperforms grid and random search for small evaluation budgets.** With 20 evaluations, it finds solutions that random search might need 200+ evaluations to match.
- **Hyperparameter tuning is the most common ML application.** Libraries like Optuna and scikit-optimize implement Bayesian optimization for practical hyperparameter search.
