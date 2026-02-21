# t-SNE Visualization

> Phase 5 — Unsupervised Learning | Kata 5.5

---

## Concept & Intuition

### What problem are we solving?

PCA is great for preserving global structure, but it struggles to reveal complex, non-linear patterns in high-dimensional data. **t-SNE** (t-distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique specifically designed for **visualization**. It excels at placing similar points near each other and dissimilar points far apart in 2D, creating stunning visualizations that reveal cluster structure invisible to PCA.

t-SNE works in two phases. First, it computes pairwise similarities between all points in the original high-dimensional space using Gaussian distributions -- nearby points get high similarity, distant points get near-zero similarity. Then, it finds a 2D arrangement of points that matches these similarities as closely as possible, using Student's t-distributions (heavier tails than Gaussian, preventing the "crowding problem" where points in 2D collapse into a blob). The algorithm optimizes the arrangement using gradient descent, minimizing the KL-divergence between the high-dimensional and low-dimensional similarity distributions.

The **perplexity** parameter controls how many neighbors each point "pays attention to." Low perplexity focuses on very local structure (tight clusters). High perplexity considers more global structure (broader patterns). Typical values range from 5 to 50.

### Why naive approaches fail

PCA projects data onto linear directions, which means it cannot "unfold" non-linear manifolds. Imagine data that lives on a Swiss roll -- PCA squashes it flat, overlapping points that are far apart along the roll surface. t-SNE can unroll the manifold and place points according to their true neighborhood relationships.

However, t-SNE has critical pitfalls that lead to misinterpretation. **Distances between clusters are meaningless** -- two clusters that appear far apart in a t-SNE plot might actually be close in the original space. **Cluster sizes are meaningless** -- t-SNE expands dense regions and compresses sparse ones. **Different runs produce different layouts** -- t-SNE is non-convex and random, so the same data can look very different. You must run t-SNE multiple times and never draw conclusions from a single plot.

### Mental models

- **t-SNE as neighborhood preservation**: "I don't care about exact distances. I just want nearby points to stay nearby." t-SNE sacrifices global geometry to perfectly preserve local neighborhoods.
- **Perplexity as neighborhood size**: Low perplexity = "only care about my 5 closest friends." High perplexity = "consider my 50 nearest acquaintances." Both perspectives are valid and reveal different structures.
- **t-SNE is a microscope, not a map**: It magnifies local structure beautifully but distorts global relationships. Use it for exploration, never for measurement.

### Visual explanations

```
PCA vs t-SNE:

  PCA (linear):                    t-SNE (non-linear):

  **** ****                         ****
  ** *** **                        *    *
  * * * * *   (overlapping)        *    *     ****
  ** *** **                                  *    *
  **** ****                         ****     *    *
                                             ****
  (clusters overlap in 2D)          (clusters clearly separated)


Perplexity Effect:

  Low (5):          Medium (30):       High (50):

  ** ** **          ****  ****         **** **** ****
  ** ** **          ****  ****
  ** ** **          ****  ****

  (many small       (balanced          (few large groups,
   tight groups)     structure)          global view)


DANGER: Things t-SNE Does NOT Tell You:

  Distance between clusters:  MEANINGLESS
  Cluster size:               MEANINGLESS
  Cluster shape:              PARTIALLY meaningful (local structure OK)
  Neighbors within a cluster: MEANINGFUL (the one thing to trust)
```

---

## Hands-on Exploration

1. Apply both PCA and t-SNE to the digits dataset. Color by digit label. Notice how t-SNE creates much cleaner cluster separation while PCA shows overlapping groups.
2. Run t-SNE with perplexity 5, 15, 30, and 50 on the same data. Observe how the visualization changes from many tight micro-clusters to fewer broader groups.
3. Run t-SNE 4 times with different `random_state` values. Notice that the overall layout changes each time (clusters might appear in different positions), but the local structure (which points cluster together) remains consistent.
4. Create a dataset where 3 clusters have very different densities. Apply t-SNE and observe how it equalizes the visual size of clusters, making the small dense cluster look the same size as the large sparse one.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# @param perplexity float 5 50 30

# --- Load dataset ---
digits = load_digits()
X, y = digits.data, digits.target
X_scaled = StandardScaler().fit_transform(X)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# --- Run t-SNE with selected perplexity ---
start = time.time()
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
            n_iter=1000, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X_scaled)
tsne_time = time.time() - start
print(f"t-SNE completed in {tsne_time:.1f}s (perplexity={perplexity})")
print(f"KL divergence: {tsne.kl_divergence_:.4f}")

# --- PCA for comparison ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- Visualization ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. PCA projection
ax = axes[0, 0]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
ax.set_title(f'PCA (2D)\n{pca.explained_variance_ratio_.sum()*100:.1f}% variance')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, ticks=range(10))

# 2. t-SNE projection
ax = axes[0, 1]
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
ax.set_title(f't-SNE (perplexity={perplexity})')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, ticks=range(10))

# 3. t-SNE with digit labels
ax = axes[0, 2]
for digit in range(10):
    mask = y == digit
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=15, alpha=0.5, label=str(digit))
ax.legend(markerscale=2, fontsize=8, ncol=2)
ax.set_title('t-SNE with Digit Labels')
ax.grid(True, alpha=0.3)

# 4-6. Perplexity comparison
perplexities = [5, 30, 50]
for idx, perp in enumerate(perplexities):
    ax = axes[1, idx]
    tsne_p = TSNE(n_components=2, perplexity=perp, random_state=42,
                  n_iter=1000, learning_rate='auto', init='pca')
    X_p = tsne_p.fit_transform(X_scaled)
    ax.scatter(X_p[:, 0], X_p[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
    ax.set_title(f'Perplexity = {perp}\nKL div = {tsne_p.kl_divergence_:.4f}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    if perp == perplexity:
        for spine in ax.spines.values():
            spine.set_color('red')
            spine.set_linewidth(3)

plt.suptitle('PCA vs t-SNE on Handwritten Digits (64D -> 2D)', fontsize=14)
plt.tight_layout()
plt.show()

# --- Reproducibility check ---
print("\n=== Reproducibility (different random seeds) ===")
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for idx, seed in enumerate([0, 1, 2, 3]):
    ax = axes[idx]
    tsne_r = TSNE(n_components=2, perplexity=30, random_state=seed,
                  n_iter=1000, learning_rate='auto', init='pca')
    X_r = tsne_r.fit_transform(X_scaled)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=y, cmap='tab10', s=8, alpha=0.5)
    ax.set_title(f'Random seed = {seed}')
    ax.grid(True, alpha=0.3)

plt.suptitle('t-SNE: Different Runs, Same Data (layout varies, neighborhoods preserved)', fontsize=12)
plt.tight_layout()
plt.show()

print("\nNote: Cluster positions change between runs, but which points")
print("cluster together (local structure) remains consistent.")
```

---

## Key Takeaways

- **t-SNE is the gold standard for high-dimensional data visualization.** It reveals cluster structure that PCA cannot by preserving local neighborhoods through non-linear projection.
- **Perplexity controls the scale of structure revealed.** Low perplexity shows fine-grained local clusters; high perplexity shows broader global patterns. Always try multiple values.
- **Distances and sizes in t-SNE plots are NOT meaningful.** Do not conclude that two clusters are "far apart" or "different sizes" based on a t-SNE plot. Only local neighborhood relationships are preserved.
- **t-SNE is for visualization only, not for downstream ML.** It is non-parametric (cannot transform new points), non-deterministic, and distorts global geometry. Use PCA for feature reduction before modeling.
- **Always run t-SNE multiple times.** Different random seeds produce different layouts. Only trust patterns that appear consistently across multiple runs.
