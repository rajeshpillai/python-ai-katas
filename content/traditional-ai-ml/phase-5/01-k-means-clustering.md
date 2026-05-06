# K-Means Clustering

> Phase 5 — Unsupervised Learning | Kata 5.1

---

## Concept & Intuition

### What problem are we solving?

You have a dataset with no labels. No one has told you which items are similar or what groups exist. How do you discover the natural structure? **K-Means clustering** is the most fundamental answer: it partitions data into K groups by finding K **centroids** (cluster centers) and assigning each point to its nearest centroid. It iterates between two steps -- assign points to the nearest centroid, then move each centroid to the mean of its assigned points -- until convergence.

K-Means is ubiquitous: customer segmentation, image compression, feature engineering, document grouping, anomaly detection (points far from any centroid). It is fast, scales to large datasets, and produces intuitive, interpretable results. The centroids themselves are meaningful -- they represent the "prototype" or "average member" of each cluster.

The critical question K-Means cannot answer on its own is: how many clusters should there be? The **elbow method** plots the total within-cluster distance (inertia) against K. As K increases, inertia always decreases, but the rate of decrease slows at the "elbow" -- the point of diminishing returns. The **silhouette score** provides a complementary measure: how similar each point is to its own cluster versus the nearest other cluster, ranging from -1 (wrong cluster) to 1 (clearly in the right cluster).

### Why naive approaches fail

K-Means assumes clusters are **spherical and equally sized**. It draws boundaries equidistant from centroids, creating Voronoi cells. When real clusters are elongated, ring-shaped, or vary dramatically in size, K-Means produces poor partitions. It will split a large cluster to balance sizes or merge overlapping non-spherical clusters.

K-Means is also sensitive to initialization. Bad starting centroids can lead to suboptimal convergence. The K-Means++ initialization scheme addresses this by spreading initial centroids apart, but the algorithm can still get stuck in local minima. Running it multiple times with different initializations (the default in scikit-learn with `n_init=10`) mitigates this.

### Mental models

- **Magnets and iron filings**: Place K magnets on a table covered with iron filings. Each filing goes to the nearest magnet. Then move each magnet to the center of its filings. Repeat until nothing moves.
- **The elbow method as diminishing returns**: Going from 1 to 2 clusters gives huge improvement. 2 to 3 helps a lot. 5 to 6 barely helps. 10 to 11 is negligible. The "elbow" is where the gains flatten.
- **Centroids as prototypes**: Each centroid is the "average customer" or "typical document" of its cluster. You can interpret clusters by examining their centroids.

### Visual explanations

```
K-Means Algorithm:

  Step 0: Initialize       Step 1: Assign          Step 2: Move centroids
  centroids randomly       points to nearest       to cluster means

  .  .  *1  . .            .  .  *1  . .           .  .  .  . .
  .  .  .  . .             1  1  1  . .            .  .  *1 . .
  .  .  .  . .    -->      1  1  .  2 .    -->     .  .  .  . .
  .  .  .  . .             .  .  2  2 2            .  .  .  . .
  .  .  . *2  .            .  .  2  2 .            .  .  . *2  .

  Repeat Steps 1 and 2 until centroids stop moving.


Elbow Method:

  Inertia
    |
    |  *
    |    *
    |      *
    |        *  <-- elbow (K=3)
    |          * *
    |              * * *
    +--+--+--+--+--+--+---> K
       1  2  3  4  5  6  7
```

---

## Hands-on Exploration

1. Generate blob data with 3 known clusters using `make_blobs`. Run K-Means with K=3 and plot the results with cluster centers. Verify it recovers the correct grouping.
2. Run K-Means with K=2 and K=6 on the same 3-cluster data. Observe how wrong K leads to splitting or merging natural clusters.
3. Plot the elbow curve (inertia vs K) and the silhouette score curve for K from 2 to 10. Find the K where both metrics agree.
4. Try K-Means on non-spherical data (e.g., `make_moons` or concentric circles). Observe the failure and understand why this is a fundamental limitation.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

# @param n_clusters int 2 10 3

# --- Generate dataset with known clusters ---
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Run K-Means with selected K ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

print(f"K = {n_clusters}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
print(f"Cluster sizes: {np.bincount(labels)}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Clustering result
ax = axes[0, 0]
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Set1', s=30, alpha=0.6)
ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X',
           edgecolors='white', linewidths=2, label='Centroids', zorder=5)
ax.set_title(f'K-Means Clustering (K={n_clusters})')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Elbow curve
ax = axes[0, 1]
K_range = range(2, 11)
inertias = []
silhouettes = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

ax.plot(K_range, inertias, 'bo-', linewidth=2, label='Inertia')
ax.axvline(x=n_clusters, color='red', linestyle='--', label=f'Selected K={n_clusters}')
ax.set_xlabel('K (Number of Clusters)')
ax.set_ylabel('Inertia (Within-cluster sum of squares)')
ax.set_title('Elbow Method')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Silhouette score
ax = axes[1, 0]
ax.plot(K_range, silhouettes, 'ro-', linewidth=2, label='Silhouette Score')
ax.axvline(x=n_clusters, color='blue', linestyle='--', label=f'Selected K={n_clusters}')
best_k = list(K_range)[np.argmax(silhouettes)]
ax.axvline(x=best_k, color='green', linestyle=':', label=f'Best K={best_k}')
ax.set_xlabel('K (Number of Clusters)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Analysis')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. K-Means failure on non-spherical data
ax = axes[1, 1]
X_moons, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
km_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_moons = km_moons.fit_predict(X_moons)
ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='Set1', s=30, alpha=0.7)
ax.scatter(km_moons.cluster_centers_[:, 0], km_moons.cluster_centers_[:, 1],
           c='black', s=200, marker='X', edgecolors='white', linewidths=2)
ax.set_title('K-Means Failure on Moon Data\n(expects spherical clusters)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Per-cluster summary ---
print("\n=== Per-Cluster Summary ===")
sil_samples = silhouette_samples(X_scaled, labels)
for c in range(n_clusters):
    mask = labels == c
    print(f"  Cluster {c}: n={mask.sum()}, "
          f"avg silhouette={sil_samples[mask].mean():.3f}, "
          f"centroid={centroids[c].round(2)}")
```

---

## Key Takeaways

- **K-Means partitions data into K groups by iteratively assigning points to the nearest centroid and updating centroids.** It converges quickly and works well on spherical, well-separated clusters.
- **Choosing K is the hardest part.** The elbow method (inertia) and silhouette score are complementary tools. Use domain knowledge to validate -- does the number of clusters make practical sense?
- **K-Means assumes spherical, equally sized clusters.** It fails on elongated, ring-shaped, or highly variable clusters. DBSCAN and hierarchical clustering handle these cases better.
- **Initialization matters.** K-Means++ (the default) spreads initial centroids apart, and running multiple initializations (`n_init=10`) reduces the chance of bad local minima.
- **Centroids are interpretable.** Each centroid represents the "average" member of its cluster, making K-Means results easy to explain to non-technical stakeholders.
