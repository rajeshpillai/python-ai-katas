# Hyperparameter Tuning

> Phase 4 — Model Evaluation & Selection | Kata 4.5

---

## Concept & Intuition

### What problem are we solving?

Every machine learning algorithm has **hyperparameters** -- settings you choose *before* training that control the model's behavior. KNN has k (number of neighbors). Decision trees have max_depth. SVMs have C and gamma. These are not learned from data; they are design decisions that dramatically affect performance. How do you find the best combination?

**Grid search** exhaustively tries every combination of specified values. If you have 5 values for C and 5 for gamma, grid search trains and evaluates 25 models. It is thorough but expensive -- adding another hyperparameter with 5 values makes it 125 models. **Random search** samples hyperparameter combinations randomly from specified distributions. Surprisingly, it often finds equally good or better configurations in far fewer iterations, because it explores the full range of each parameter rather than concentrating on a grid.

**Bayesian optimization** goes further by using the results of previous evaluations to decide where to search next. It builds a probabilistic model of the objective function and balances exploring unknown regions with exploiting promising areas. For expensive-to-evaluate models, Bayesian optimization can find excellent hyperparameters in a fraction of the evaluations grid search would require.

### Why naive approaches fail

Manual tuning is unreliable and biased. You might try max_depth=3 and max_depth=10, see that 10 is better, and conclude that deeper is always better -- missing that max_depth=5 was actually optimal. You also risk overfitting to the test set by repeatedly checking performance and adjusting.

Grid search scales exponentially with the number of hyperparameters. With 4 hyperparameters and 10 values each, you need 10,000 evaluations. Most of this computation is wasted -- research has shown that typically only 1-2 hyperparameters significantly affect performance. Grid search allocates equal effort to important and unimportant dimensions, while random search automatically focuses resolution on the important ones.

### Mental models

- **Grid search as a systematic sweep**: Like tuning two knobs on a radio by trying every combination of positions. Thorough but slow.
- **Random search as random sampling**: Randomly turning the knobs. More likely to find the right spot on the important knob because it tries more distinct values per dimension.
- **Bayesian optimization as a guided search**: Like a metal detector -- after getting a signal in one area, you concentrate your search there while occasionally checking distant spots.
- **Inner loop / outer loop**: Cross-validation is the inner loop (evaluate one hyperparameter set). The search strategy is the outer loop (which hyperparameter set to try next).

### Visual explanations

```
Grid Search vs Random Search (2D parameter space):

  Grid Search:               Random Search:
  (9 evaluations)            (9 evaluations)

  C  |  x   x   x           C  |    x       x
     |                          |  x     x
     |  x   x   x              |        x
     |                          |  x
     |  x   x   x              |     x    x
     +-----------> gamma        +-----------> gamma

  Grid: only 3 unique         Random: 9 unique values
  values per dimension!        per dimension!

  If gamma is the important parameter, random search
  tries 3x more gamma values in the same budget.


Bayesian Optimization Flow:

  1. Sample a few random points
  2. Fit a surrogate model (Gaussian Process)
  3. Find the next best point to evaluate (acquisition function)
  4. Evaluate and update the model
  5. Repeat until budget exhausted

  Iteration 1:  ??  ??  0.82  ??  ??  0.75  ??
  Iteration 3:  ??  0.88  0.82  ??  ??  0.75  ??
  Iteration 5:  ??  0.88  0.82  0.91  ??  0.75  ??
                              (concentrating near the peak)
```

---

## Hands-on Exploration

1. Perform grid search on an SVM (C and gamma) using `GridSearchCV`. Print the best parameters and the full results table. Visualize the accuracy heatmap over the C-gamma grid.
2. Perform random search with the same budget (same total number of evaluations). Compare the best score found by each method. Repeat 10 times and see which method is more consistent.
3. Plot the convergence curve: best-score-so-far vs number of evaluations for grid search and random search. Random search typically finds a good solution earlier.
4. Use `cross_val_score` inside a manual loop to implement Bayesian optimization with `scikit-optimize` or a simple gradient-free optimizer. Compare with grid/random search.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import loguniform, uniform
import time

# --- Generate dataset ---
X, y = make_classification(
    n_samples=500, n_features=10, n_redundant=2, n_informative=5,
    flip_y=0.05, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Grid Search ---
print("=== Grid Search ===")
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1, 10],
}

pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=42))

start = time.time()
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy',
                            return_train_score=True, n_jobs=-1)
grid_search.fit(X, y)
grid_time = time.time() - start

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
print(f"Total fits: {len(grid_search.cv_results_['mean_test_score'])}")
print(f"Time: {grid_time:.2f}s")

# --- Random Search (same budget) ---
print("\n=== Random Search ===")
param_dist = {
    'svc__C': loguniform(0.01, 100),
    'svc__gamma': loguniform(0.001, 10),
}

start = time.time()
random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=25, cv=cv,
                                     scoring='accuracy', random_state=42,
                                     return_train_score=True, n_jobs=-1)
random_search.fit(X, y)
random_time = time.time() - start

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
print(f"Total fits: {random_search.n_iter}")
print(f"Time: {random_time:.2f}s")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Grid search heatmap
ax = axes[0]
C_vals = param_grid['svc__C']
gamma_vals = param_grid['svc__gamma']
scores_grid = grid_search.cv_results_['mean_test_score'].reshape(len(C_vals), len(gamma_vals))
im = ax.imshow(scores_grid, cmap='YlOrRd', aspect='auto',
               extent=[0, len(gamma_vals)-1, 0, len(C_vals)-1])
ax.set_xticks(range(len(gamma_vals)))
ax.set_xticklabels([str(g) for g in gamma_vals])
ax.set_yticks(range(len(C_vals)))
ax.set_yticklabels([str(c) for c in C_vals])
ax.set_xlabel('Gamma')
ax.set_ylabel('C')
ax.set_title(f'Grid Search Heatmap\nBest={grid_search.best_score_:.3f}')
plt.colorbar(im, ax=ax, label='Accuracy')

# Add text annotations
for i in range(len(C_vals)):
    for j in range(len(gamma_vals)):
        ax.text(j, i, f'{scores_grid[i,j]:.2f}', ha='center', va='center', fontsize=8)

# 2. Random search scatter
ax = axes[1]
C_random = [p['svc__C'] for p in random_search.cv_results_['params']]
gamma_random = [p['svc__gamma'] for p in random_search.cv_results_['params']]
scores_random = random_search.cv_results_['mean_test_score']

scatter = ax.scatter(np.log10(gamma_random), np.log10(C_random),
                     c=scores_random, cmap='YlOrRd', s=100, edgecolors='black')
best_params = random_search.best_params_
ax.scatter(np.log10(best_params['svc__gamma']), np.log10(best_params['svc__C']),
           c='blue', s=200, marker='*', edgecolors='black', zorder=5, label='Best')
ax.set_xlabel('log10(Gamma)')
ax.set_ylabel('log10(C)')
ax.set_title(f'Random Search Points\nBest={random_search.best_score_:.3f}')
ax.legend()
plt.colorbar(scatter, ax=ax, label='Accuracy')

# 3. Convergence comparison
ax = axes[2]
# Grid search convergence
grid_scores = grid_search.cv_results_['mean_test_score']
grid_best_so_far = np.maximum.accumulate(grid_scores)

# Random search convergence
rand_scores = random_search.cv_results_['mean_test_score']
rand_best_so_far = np.maximum.accumulate(rand_scores)

ax.plot(range(1, len(grid_best_so_far)+1), grid_best_so_far, 'b-o', markersize=3,
        linewidth=2, label=f'Grid Search (n={len(grid_scores)})')
ax.plot(range(1, len(rand_best_so_far)+1), rand_best_so_far, 'r-o', markersize=3,
        linewidth=2, label=f'Random Search (n={len(rand_scores)})')
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Best Score So Far')
ax.set_title('Convergence: Grid vs Random')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Manual Bayesian-style optimization (simple) ---
print("\n=== Simple Sequential Optimization ===")
best_score = 0
best_params_manual = {}
history = []

np.random.seed(42)
for i in range(25):
    C = 10 ** np.random.uniform(-2, 2)
    gamma = 10 ** np.random.uniform(-3, 1)
    pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C, gamma=gamma, random_state=42))
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    history.append(mean_score)
    if mean_score > best_score:
        best_score = mean_score
        best_params_manual = {'C': C, 'gamma': gamma}

print(f"Best score: {best_score:.3f}")
print(f"Best params: C={best_params_manual['C']:.4f}, gamma={best_params_manual['gamma']:.4f}")
```

---

## Key Takeaways

- **Hyperparameters are choices YOU make; parameters are learned from data.** The quality of your model depends critically on these design decisions.
- **Grid search is exhaustive but scales exponentially.** With D hyperparameters and N values each, it requires N^D evaluations. It becomes impractical beyond 2-3 hyperparameters.
- **Random search is often as good or better than grid search.** It explores more unique values per dimension and is more likely to find good configurations for the important hyperparameters.
- **Always use cross-validation inside the search.** `GridSearchCV` and `RandomizedSearchCV` handle this automatically, preventing overfitting to a single validation split.
- **Log-uniform distributions are appropriate for scale parameters.** Parameters like C and gamma span orders of magnitude, so sampling uniformly in log-space (0.01 to 100) makes more sense than linear spacing.
