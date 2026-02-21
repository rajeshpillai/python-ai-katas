# Hierarchical Clustering

> Phase 5 — Unsupervised Learning | Kata 5.2

---

## Concept & Intuition

### What problem are we solving?

K-Means requires you to specify K upfront. But what if you do not know how many clusters exist, or the data has clusters at multiple scales -- cities within states within countries? **Hierarchical clustering** builds a tree of nested clusters, from individual points up to one giant cluster. You can then "cut" this tree at any level to get any number of clusters. The tree itself -- called a **dendrogram** -- is a rich visualization that reveals the hierarchical structure of your data.

**Agglomerative** (bottom-up) hierarchical clustering is the most common variant. It starts with each data point as its own cluster and repeatedly merges the two closest clusters until only one remains. The key decision is the **linkage criterion** -- how to measure the distance between two clusters. **Single linkage** uses the minimum distance between any pair of points. **Complete linkage** uses the maximum. **Average linkage** uses the mean. **Ward's method** minimizes the increase in total within-cluster variance, often producing the most compact, balanced clusters.

The dendrogram displays the entire merge history. The y-axis represents the distance at which each merge occurred. Large vertical jumps in the dendrogram indicate natural cluster boundaries -- cutting just below a big jump gives well-separated clusters.

### Why naive approaches fail

Single linkage suffers from the **chaining effect**: it can merge two clearly separate clusters because a single pair of points from each cluster happens to be close. This creates elongated, straggly clusters. Complete linkage avoids chaining but is sensitive to outliers -- one distant point can prevent two otherwise similar clusters from merging.

Hierarchical clustering is also computationally expensive. The standard algorithm is O(n^3) in time and O(n^2) in memory for storing the distance matrix. This makes it impractical for datasets larger than about 10,000 points. For large datasets, consider running K-Means first and then applying hierarchical clustering to the centroids.

### Mental models

- **Organizational chart building**: Start with individual employees. Merge the two most similar into a team. Keep merging teams into departments, then divisions, then the whole company. The org chart IS the dendrogram.
- **Dendrograms as family trees**: The height of each merge shows how "distant" the combined groups are. Siblings merged low on the tree are very similar; those merged high are quite different.
- **Cutting the tree**: Imagine drawing a horizontal line across the dendrogram. Every vertical line the cut crosses becomes a separate cluster. Moving the cut up gives fewer, larger clusters; moving it down gives more, smaller clusters.

### Visual explanations

```
Agglomerative Clustering (bottom-up):

  Step 1: {A} {B} {C} {D} {E}     (5 clusters)
  Step 2: {A,B} {C} {D} {E}       (merge A and B, closest pair)
  Step 3: {A,B} {C} {D,E}         (merge D and E)
  Step 4: {A,B} {C,D,E}           (merge C with D,E)
  Step 5: {A,B,C,D,E}             (one cluster)


Dendrogram:

  Distance
    6  |          ___|___
    5  |         |       |
    4  |     ___|___     |
    3  |    |       |    |
    2  |  __|__     |  __|__
    1  | |     |    | |     |
    0  | A     B    C D     E
       +--+--+--+--+--+--+--

  Cut at distance=3.5: --> {A,B,C} and {D,E} (2 clusters)
  Cut at distance=1.5: --> {A,B} {C} {D,E}  (3 clusters)


Linkage Methods:

  Single (min):    d(A,B) = min distance between any points
  Complete (max):  d(A,B) = max distance between any points
  Average:         d(A,B) = mean of all pairwise distances
  Ward:            d(A,B) = increase in total within-cluster variance
```

---

## Hands-on Exploration

1. Generate blob data and build a dendrogram using `scipy.cluster.hierarchy`. Identify the natural number of clusters from the largest vertical gaps.
2. Compare single, complete, average, and Ward linkage on the same dataset. Observe how single linkage creates chain-like clusters while Ward produces compact, balanced ones.
3. Try hierarchical clustering on the moon dataset. Compare single linkage (which handles this well) with Ward linkage (which fails). Understand why linkage choice matters for cluster shape.
4. Use `fcluster` to cut the dendrogram at different thresholds and compare the resulting clusterings with different numbers of groups.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# --- Generate dataset ---
X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=0.7, random_state=42)

# --- Dendrogram ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Full dendrogram with Ward linkage
ax = axes[0, 0]
Z = linkage(X, method='ward')
dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
           leaf_rotation=90, leaf_font_size=8, color_threshold=8)
ax.set_title('Dendrogram (Ward Linkage)')
ax.set_xlabel('Cluster Size')
ax.set_ylabel('Distance')
ax.axhline(y=8, color='red', linestyle='--', label='Cut for 4 clusters')
ax.legend()

# 2. Compare linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']
ax = axes[0, 1]

for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=4, linkage=method)
    labels = agg.fit_predict(X)
    sil = silhouette_score(X, labels)
    ax.bar(method, sil, edgecolor='black')

ax.set_ylabel('Silhouette Score')
ax.set_title('Linkage Method Comparison (K=4)')
ax.grid(True, alpha=0.3, axis='y')

# 3. Clustering results for Ward
ax = axes[1, 0]
agg_ward = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_ward = agg_ward.fit_predict(X)
ax.scatter(X[:, 0], X[:, 1], c=labels_ward, cmap='Set1', s=30, alpha=0.7)
ax.set_title(f'Ward Linkage (4 clusters)\nSilhouette={silhouette_score(X, labels_ward):.3f}')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.grid(True, alpha=0.3)

# 4. Moon dataset: single vs ward
ax = axes[1, 1]
X_moons, _ = make_moons(n_samples=200, noise=0.08, random_state=42)

# Single linkage handles moons well
agg_single = AgglomerativeClustering(n_clusters=2, linkage='single')
labels_single = agg_single.fit_predict(X_moons)

# Ward linkage fails on moons
agg_ward_moon = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels_ward_moon = agg_ward_moon.fit_predict(X_moons)

ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_single, cmap='Set1',
           s=40, alpha=0.7, marker='o', label='Single linkage')
ax.scatter(X_moons[:, 0] + 3, X_moons[:, 1], c=labels_ward_moon, cmap='Set1',
           s=40, alpha=0.7, marker='s', label='Ward linkage')
ax.set_title('Moon Data: Single (left) vs Ward (right)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Cutting the dendrogram at different levels ---
print("=== Cutting the Dendrogram ===")
Z = linkage(X, method='ward')

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
thresholds = [5, 8, 12, 20]

for ax, t in zip(axes, thresholds):
    clusters = fcluster(Z, t=t, criterion='distance')
    n_clusters = len(np.unique(clusters))
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='Set1', s=30, alpha=0.7)
    ax.set_title(f'Distance threshold={t}\n{n_clusters} clusters')
    ax.grid(True, alpha=0.3)

plt.suptitle('Different Cuts of the Same Dendrogram', fontsize=14)
plt.tight_layout()
plt.show()

# --- Summary ---
print("\n=== Linkage Method Summary ===")
for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=4, linkage=method)
    labels = agg.fit_predict(X)
    sizes = np.bincount(labels)
    sil = silhouette_score(X, labels)
    print(f"  {method:>8s}: Silhouette={sil:.3f}, Cluster sizes={sizes}")
```

---

## Key Takeaways

- **Hierarchical clustering builds a tree of nested clusters without requiring K in advance.** The dendrogram visualizes the entire merge history and reveals natural groupings at multiple scales.
- **Linkage criteria dramatically affect results.** Ward linkage produces compact, spherical clusters (similar to K-Means). Single linkage handles elongated shapes but is prone to chaining.
- **Cut the dendrogram where vertical gaps are largest.** Large gaps indicate clear separations between groups. The height of the cut determines the number of clusters.
- **Computational cost is O(n^2) to O(n^3).** Hierarchical clustering does not scale to large datasets. For more than ~10,000 points, use K-Means or other scalable methods.
- **The dendrogram is a powerful exploratory tool.** Even if you ultimately use K-Means, building a dendrogram first can reveal the natural number of clusters and hierarchical relationships in your data.
