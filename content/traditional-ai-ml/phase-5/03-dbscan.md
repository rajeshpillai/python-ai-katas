# DBSCAN

> Phase 5 — Unsupervised Learning | Kata 5.3

---

## Concept & Intuition

### What problem are we solving?

K-Means requires specifying the number of clusters and assumes spherical shapes. Hierarchical clustering does not scale. What if your clusters are **arbitrary shapes** -- crescents, rings, elongated blobs -- and you also need to **identify noise points** that do not belong to any cluster? **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) solves all of these problems by defining clusters as dense regions of points separated by sparser regions.

DBSCAN has two parameters: **eps** (the radius of a neighborhood) and **min_samples** (the minimum number of points required to form a dense region). A point with at least `min_samples` neighbors within radius `eps` is a **core point**. Points reachable from core points are part of the cluster. Points that are not reachable from any core point are labeled as **noise** (outliers). This density-based approach naturally discovers clusters of arbitrary shape, automatically determines the number of clusters, and explicitly identifies outlier points.

DBSCAN is particularly valuable in real-world applications like geographic data analysis (finding population centers), anomaly detection (noise points are anomalies), and any scenario where cluster shapes are unknown and irregular.

### Why naive approaches fail

K-Means splits the crescents in a moon dataset straight down the middle because it sees two regions equidistant from centroids. DBSCAN follows the density of each crescent and clusters them correctly. K-Means also assigns every point to a cluster, including outliers that should not belong anywhere. DBSCAN explicitly labels these as noise.

The challenge with DBSCAN is choosing eps and min_samples. If eps is too small, most points become noise (no cluster forms). If eps is too large, distinct clusters merge into one. The **k-distance plot** (distance to the k-th nearest neighbor, sorted) helps: the "elbow" in this plot suggests a good eps value. min_samples should typically be at least the dimensionality of the data plus one.

### Mental models

- **Epidemiological contact tracing**: A person (core point) who has been in close contact (within eps) with at least min_samples people forms a cluster. Anyone reachable through chains of close contacts is in the same cluster. Isolated individuals (noise) are not in any cluster.
- **Walking through a crowd**: You can walk from any point in a cluster to any other by stepping only to nearby points (within eps). If you have to cross an empty gap, you are moving to a different cluster.
- **eps as a magnifying glass**: Small eps = zoomed in, see fine structure, many small clusters. Large eps = zoomed out, merges nearby clusters into larger ones.

### Visual explanations

```
DBSCAN Concepts:

  eps=0.5, min_samples=3

  Core point:    has >= 3 neighbors within radius eps
  Border point:  within eps of a core point, but < 3 neighbors
  Noise point:   not within eps of any core point

  . . . . . . . . . .
  . . [C C C] . . . .     C = core points (dense region)
  . . [C C] . . . . .     B = border points (edge of cluster)
  . . B . . . . N . .     N = noise points (isolated)
  . . . . . . . . . .

  Each C has >= min_samples neighbors in its eps-neighborhood.
  B is within eps of a C but has few neighbors itself.
  N is far from any core point.


K-distance Plot (for choosing eps):

  k-distance
    |
    |                         *
    |                      *
    |                   *
    |               * *     <-- "knee" suggests eps ~ 0.5
    |           * *
    |     * * *
    | * *
    +----+----+----+-----> Points (sorted)
```

---

## Hands-on Exploration

1. Generate moon-shaped data with `make_moons`. Run K-Means (K=2) and DBSCAN side by side. DBSCAN should correctly separate the crescents while K-Means fails.
2. Add random outlier points to the dataset. Run DBSCAN and observe that outliers are labeled as noise (-1) while the clusters remain intact.
3. Create a k-distance plot: for each point, compute the distance to its 5th nearest neighbor. Sort these distances and plot them. The "knee" point suggests a good eps value.
4. Vary eps from 0.1 to 2.0 and plot the number of clusters and noise points at each setting. Find the range of eps values that produce stable, reasonable clustering.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# @param eps float 0.1 3.0 0.5

# --- Generate dataset with noise ---
X_moons, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
# Add some outlier noise
rng = np.random.RandomState(42)
noise_points = rng.uniform(low=[-1.5, -1], high=[2.5, 1.5], size=(30, 2))
X = np.vstack([X_moons, noise_points])

# --- Run DBSCAN ---
dbscan = DBSCAN(eps=eps, min_samples=5)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"eps = {eps}")
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({n_noise/len(X)*100:.1f}%)")
if n_clusters >= 2:
    mask = labels != -1
    print(f"Silhouette (excl. noise): {silhouette_score(X[mask], labels[mask]):.3f}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. DBSCAN result
ax = axes[0, 0]
unique_labels = set(labels)
colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 1)))
for k, col in zip(sorted(unique_labels), colors):
    if k == -1:
        col = 'gray'
        marker = 'x'
        label = f'Noise ({(labels == k).sum()})'
        alpha = 0.5
    else:
        marker = 'o'
        label = f'Cluster {k} ({(labels == k).sum()})'
        alpha = 0.7
    mask = labels == k
    ax.scatter(X[mask, 0], X[mask, 1], c=[col], s=30, marker=marker, alpha=alpha, label=label)

# Mark core points
core_mask = np.zeros(len(X), dtype=bool)
core_mask[dbscan.core_sample_indices_] = True
ax.scatter(X[core_mask, 0], X[core_mask, 1], facecolors='none',
           edgecolors='black', s=80, linewidths=0.5, alpha=0.3)
ax.set_title(f'DBSCAN (eps={eps}, min_samples=5)\n{n_clusters} clusters, {n_noise} noise')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# 2. K-Means comparison (fails on this data)
ax = axes[0, 1]
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X)
ax.scatter(X[:, 0], X[:, 1], c=km_labels, cmap='Set1', s=30, alpha=0.7)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='black', s=200, marker='X', edgecolors='white', linewidths=2)
ax.set_title('K-Means (K=2) - Fails on Non-Spherical Data')
ax.grid(True, alpha=0.3)

# 3. K-distance plot for eps selection
ax = axes[1, 0]
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
distances, _ = nn.kneighbors(X)
k_distances = np.sort(distances[:, k-1])[::-1]

ax.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
ax.axhline(y=eps, color='red', linestyle='--', linewidth=2, label=f'Current eps={eps}')
ax.set_xlabel('Points (sorted by distance)')
ax.set_ylabel(f'{k}-th Nearest Neighbor Distance')
ax.set_title('K-Distance Plot (for choosing eps)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Sensitivity to eps
ax = axes[1, 1]
eps_range = np.arange(0.1, 2.1, 0.05)
n_clusters_list = []
n_noise_list = []
silhouettes = []

for e in eps_range:
    db = DBSCAN(eps=e, min_samples=5)
    lbl = db.fit_predict(X)
    nc = len(set(lbl)) - (1 if -1 in lbl else 0)
    nn_count = (lbl == -1).sum()
    n_clusters_list.append(nc)
    n_noise_list.append(nn_count)

ax.plot(eps_range, n_clusters_list, 'b-o', markersize=2, linewidth=2, label='Clusters')
ax2 = ax.twinx()
ax2.plot(eps_range, n_noise_list, 'r-o', markersize=2, linewidth=2, label='Noise points')
ax.axvline(x=eps, color='green', linestyle='--', label=f'Current eps={eps}')
ax.set_xlabel('eps')
ax.set_ylabel('Number of Clusters', color='blue')
ax2.set_ylabel('Number of Noise Points', color='red')
ax.set_title('Sensitivity to eps')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Point type breakdown ---
print(f"\n=== Point Type Breakdown ===")
print(f"  Core points: {len(dbscan.core_sample_indices_)}")
border_count = ((labels != -1) & ~core_mask).sum()
print(f"  Border points: {border_count}")
print(f"  Noise points: {n_noise}")
```

---

## Key Takeaways

- **DBSCAN finds clusters of arbitrary shape based on density.** It does not assume spherical clusters and automatically determines the number of clusters.
- **Noise detection is a built-in feature.** Points that do not belong to any dense region are explicitly labeled as noise (-1), making DBSCAN useful for anomaly detection.
- **eps and min_samples control sensitivity.** The k-distance plot provides a principled way to choose eps. min_samples should be at least dimensionality + 1.
- **DBSCAN struggles with varying densities.** If one cluster is much denser than another, a single eps cannot capture both. HDBSCAN extends DBSCAN to handle this.
- **DBSCAN excels where K-Means fails.** Non-convex shapes, unknown number of clusters, and the presence of outliers are all scenarios where DBSCAN is the superior choice.
