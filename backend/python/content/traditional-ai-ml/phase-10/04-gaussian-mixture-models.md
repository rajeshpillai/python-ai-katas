# Gaussian Mixture Models

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.4

---

## Concept & Intuition

### What problem are we solving?

Many datasets contain hidden subgroups. Customer purchase data might have distinct segments (budget shoppers, premium buyers, occasional splurgers). Gene expression data might reflect different cell types. A single Gaussian distribution cannot capture this structure — the data has multiple modes, and we need a model that can represent a **mixture** of distributions.

A **Gaussian Mixture Model (GMM)** models data as coming from K Gaussian distributions, each with its own mean, covariance, and mixing weight. The challenge is that we do not know which data point belongs to which Gaussian — these are **latent (hidden) variables**. The **Expectation-Maximization (EM) algorithm** solves this chicken-and-egg problem by alternating between two steps: the **E-step** computes soft assignments (probabilities that each point belongs to each cluster), and the **M-step** re-estimates the Gaussian parameters using those soft assignments. Each iteration is guaranteed to increase the likelihood, and the algorithm converges to a local maximum.

GMMs are a probabilistic generalization of k-means. Where k-means makes hard assignments (each point belongs to exactly one cluster), GMMs make soft assignments (each point has a probability of belonging to each cluster). This is more realistic and provides uncertainty estimates about cluster membership.

### Why naive approaches fail

K-means assumes clusters are spherical and equally sized — strong assumptions that are often violated. It also makes hard assignments, so a point on the boundary between two clusters gets no indication of ambiguity. GMMs handle elliptical clusters, clusters of different sizes, and provide probabilistic memberships. However, GMMs require choosing the number of components K, and EM can get stuck in local optima, so multiple random restarts are standard practice.

### Mental models

- **K-means is a special case of GMM**: When all covariances are equal and small, soft GMM assignments become hard k-means assignments.
- **EM as iterative refinement**: E-step says "Given these parameters, which cluster likely generated each point?" M-step says "Given these assignments, what are the best parameters?" Like alternating between adjusting the lens and adjusting the focus.
- **Soft clustering as uncertainty**: A point with 50% probability for cluster A and 50% for cluster B is genuinely ambiguous. GMMs quantify this; k-means hides it.

### Visual explanations

```
Gaussian Mixture Model with K=3:

  Component 1: N(mu1, Sigma1), weight = 0.3
  Component 2: N(mu2, Sigma2), weight = 0.5
  Component 3: N(mu3, Sigma3), weight = 0.2

  p(x) = 0.3 * N(x|mu1,S1) + 0.5 * N(x|mu2,S2) + 0.2 * N(x|mu3,S3)

EM Algorithm:
  Initialize: random means, identity covariances, equal weights

  E-step (soft assignments):
    r(i,k) = weight_k * N(x_i | mu_k, S_k) / sum_j weight_j * N(x_i | mu_j, S_j)

    Point x_i:  r(i,1)=0.1  r(i,2)=0.7  r(i,3)=0.2
    --> "70% likely from cluster 2"

  M-step (re-estimate parameters):
    N_k = sum_i r(i,k)                          (effective cluster size)
    mu_k = sum_i r(i,k) * x_i / N_k             (weighted mean)
    S_k = sum_i r(i,k) * (x_i-mu_k)(x_i-mu_k)^T / N_k  (weighted covariance)
    weight_k = N_k / N                           (mixing weight)

  Repeat until convergence.

Model selection (choosing K):
  K=1: underfitting (one big blob)
  K=3: good fit (matches true structure)
  K=10: overfitting (fitting noise)
  Use BIC or AIC to choose K.
```

---

## Hands-on Exploration

1. Generate 2D data from 3 Gaussians with different means and covariances. Visualize the raw data — can you see the clusters?
2. Run k-means with K=3 and plot hard assignments. Then run GMM with K=3 and plot soft assignments (color intensity = probability). Which is more informative?
3. Try K=2 and K=5 on 3-cluster data. Compare BIC scores. Does BIC correctly identify K=3 as best?
4. Initialize EM with bad starting points. Does it converge to the correct solution? How many restarts do you need?

---

## Live Code

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

np.random.seed(42)

# --- Generate 2D data from a mixture of Gaussians ---
# @param n_components int 2 8 3
n_components = 3
n_samples_per = 100

# True cluster parameters
true_means = [
    [2.0, 2.0],
    [7.0, 7.0],
    [2.0, 8.0],
    [8.0, 2.0],
    [5.0, 5.0],
    [0.0, 5.0],
    [5.0, 0.0],
    [9.0, 5.0],
][:n_components]

true_covs = [
    [[1.0, 0.5], [0.5, 1.0]],
    [[1.5, -0.3], [-0.3, 0.8]],
    [[0.6, 0.0], [0.0, 1.2]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.8, 0.4], [0.4, 0.8]],
    [[1.2, 0.0], [0.0, 0.6]],
    [[0.5, 0.0], [0.0, 0.5]],
    [[1.0, -0.5], [-0.5, 1.0]],
][:n_components]

# Generate data
X_parts = []
y_true = []
for k in range(n_components):
    samples = np.random.multivariate_normal(true_means[k], true_covs[k], n_samples_per)
    X_parts.append(samples)
    y_true.extend([k] * n_samples_per)

X = np.vstack(X_parts)
y_true = np.array(y_true)
N = len(X)

# Shuffle
idx = np.random.permutation(N)
X = X[idx]
y_true = y_true[idx]

print(f"=== Gaussian Mixture Models ===")
print(f"True components: {n_components}")
print(f"Total samples: {N}\n")

# --- Fit GMM ---
gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                       n_init=5, random_state=42)
gmm.fit(X)
y_gmm = gmm.predict(X)
probs = gmm.predict_proba(X)

# --- Fit K-Means for comparison ---
kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=42)
y_km = kmeans.fit_predict(X)

# --- Report ---
print("=== Learned GMM Parameters ===")
for k in range(n_components):
    print(f"\nComponent {k}:")
    print(f"  Weight: {gmm.weights_[k]:.3f} (true: {1/n_components:.3f})")
    print(f"  Mean:   [{gmm.means_[k][0]:.2f}, {gmm.means_[k][1]:.2f}]")
    print(f"  Covariance diagonal: [{gmm.covariances_[k][0,0]:.2f}, {gmm.covariances_[k][1,1]:.2f}]")

# --- Clustering accuracy ---
ari_gmm = adjusted_rand_score(y_true, y_gmm)
ari_km = adjusted_rand_score(y_true, y_km)

print(f"\n=== Clustering Quality (Adjusted Rand Index) ===")
print(f"GMM:     {ari_gmm:.4f}")
print(f"K-Means: {ari_km:.4f}")

# --- Soft vs Hard assignments ---
print(f"\n=== Soft Assignments (GMM) — Sample Points ===")
print(f"{'Point':>6}  ", end="")
for k in range(n_components):
    print(f"{'P(k=' + str(k) + ')':>8}", end="  ")
print(f"{'Hard':>6}  {'True':>6}")
print("-" * (16 + n_components * 10 + 14))

# Show 10 interesting points (those with ambiguous assignments)
ambiguity = 1.0 - np.max(probs, axis=1)
interesting_idx = np.argsort(ambiguity)[-5:]  # most ambiguous
certain_idx = np.argsort(ambiguity)[:5]        # most certain
sample_idx = np.concatenate([certain_idx, interesting_idx])

for i in sample_idx:
    print(f"{i:>6}  ", end="")
    for k in range(n_components):
        print(f"{probs[i, k]:>8.3f}", end="  ")
    print(f"{y_gmm[i]:>6}  {y_true[i]:>6}")

# --- Model Selection with BIC ---
print(f"\n=== Model Selection (BIC) ===")
print(f"{'K':>4}  {'BIC':>12}  {'AIC':>12}  {'Log-Lik':>12}")
print("-" * 46)
bic_scores = []
for k in range(1, min(8, n_components + 4)):
    gmm_k = GaussianMixture(n_components=k, covariance_type='full',
                             n_init=5, random_state=42)
    gmm_k.fit(X)
    bic = gmm_k.bic(X)
    aic = gmm_k.aic(X)
    ll = gmm_k.score(X) * N  # total log-likelihood
    bic_scores.append((k, bic))
    marker = " <-- best" if k == n_components else ""
    print(f"{k:>4}  {bic:>12.1f}  {aic:>12.1f}  {ll:>12.1f}{marker}")

best_k = min(bic_scores, key=lambda x: x[1])[0]
print(f"\nBIC selects K={best_k} (true K={n_components})")

# --- EM Convergence ---
print(f"\n=== EM Convergence ===")
gmm_trace = GaussianMixture(n_components=n_components, covariance_type='full',
                             max_iter=1, warm_start=True, n_init=1, random_state=42)

print(f"{'Iteration':>10}  {'Log-Likelihood':>15}")
print("-" * 28)
for i in range(20):
    gmm_trace.fit(X)
    ll = gmm_trace.score(X) * N
    print(f"{i+1:>10}  {ll:>15.2f}")
    if i > 0 and abs(ll - prev_ll) < 0.01:
        print(f"Converged at iteration {i+1}")
        break
    prev_ll = ll
```

---

## Key Takeaways

- **GMMs model data as a mixture of K Gaussians with latent cluster assignments.** Each data point is generated by one of the components, but we do not know which one.
- **EM alternates between soft assignment (E-step) and parameter estimation (M-step).** Each iteration increases the likelihood, guaranteed. Convergence is to a local maximum.
- **Soft clustering is more informative than hard clustering.** Knowing a point has 60/40 probability between two clusters is more useful than forcing a hard decision.
- **Model selection (choosing K) requires criteria like BIC or AIC.** These penalize complexity to prevent overfitting with too many components.
- **GMMs generalize k-means.** K-means is a special case where covariances are assumed equal and assignments are hard. GMMs are more flexible but more computationally expensive.
