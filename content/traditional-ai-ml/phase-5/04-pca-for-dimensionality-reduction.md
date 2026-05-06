# PCA for Dimensionality Reduction

> Phase 5 — Unsupervised Learning | Kata 5.4

---

## Concept & Intuition

### What problem are we solving?

Real-world datasets often have dozens or hundreds of features, but much of that information is redundant. Features are correlated -- income correlates with education, height correlates with weight. **Principal Component Analysis (PCA)** finds the directions in feature space that capture the most variation and projects the data onto those directions. It reduces dimensionality while preserving as much information as possible.

PCA works by computing the **eigenvectors** of the data's covariance matrix. Each eigenvector defines a "principal component" -- a new axis in the rotated coordinate system. The corresponding **eigenvalue** tells you how much variance that component explains. The first principal component captures the direction of maximum variance, the second captures the maximum variance orthogonal to the first, and so on. By keeping only the top K components (those with the largest eigenvalues), you get a K-dimensional representation that retains most of the information.

PCA is used everywhere: visualization (projecting high-dimensional data to 2D/3D), noise reduction (discarding low-variance components that are mostly noise), speeding up downstream algorithms (fewer features = faster training), and decorrelating features (principal components are orthogonal by construction).

### Why naive approaches fail

Simply dropping features (e.g., keeping only the first 10 columns) throws away information arbitrarily. You might drop a feature that is highly informative while keeping two features that are nearly identical. PCA is optimal in the sense that no other linear projection preserves more variance in the same number of dimensions.

Applying PCA without scaling features first is a common mistake. PCA maximizes variance, so a feature measured in millimeters (range 0-10000) will dominate over a feature measured in meters (range 0-10), purely due to scale differences. Always standardize features before PCA unless they are already on the same scale.

### Mental models

- **PCA as finding the "spine" of a cloud**: Imagine a 3D cloud of points shaped like a cigar. PC1 is the long axis of the cigar (most variance). PC2 is the widest perpendicular direction. PC3 is the thinnest -- discarding it loses almost nothing.
- **Scree plot as diminishing returns**: Each component explains less variance than the last. The scree plot shows when you have captured "enough" -- typically 90-95% of total variance.
- **Rotation, not distortion**: PCA rotates your coordinate system to align with the data's natural axes. No information is lost in the rotation; information is only lost when you drop components.

### Visual explanations

```
PCA in 2D -> 1D:

  Original data (2D):              After PCA:
                                   (projected onto PC1)
  Feature 2                        PC2 (less variance)
    |      * *                       |
    |    * * *                       |     * * *
    |  * * *          -->            | * * * * *
    | * *                            |* *
    +----------> Feature 1           +----------> PC1 (most variance)

  PC1 = direction of maximum variance (the "long axis")
  PC2 = perpendicular direction (less important)


Scree Plot:

  Variance
  Explained
    40% |  ####
    30% |  ####  ####
    20% |  ####  ####  ####
    10% |  ####  ####  ####  ####
     5% |  ####  ####  ####  ####  ####
        +--PC1---PC2---PC3---PC4---PC5--

  Cumulative: 40%, 70%, 90%, 95%, 100%
  --> Keep 3 components for 90% of variance
```

---

## Hands-on Exploration

1. Generate 3D data that lies mostly on a 2D plane (one feature is a linear combination of the other two plus noise). Apply PCA with 2 components and verify that the variance explained is near 100%.
2. Apply PCA to a high-dimensional dataset (e.g., digits with 64 features). Plot the scree plot (variance explained per component) and the cumulative variance plot. Find how many components capture 95% of variance.
3. Visualize the digits dataset projected to 2D using PCA. Color points by digit label and see which digits cluster together and which overlap.
4. Compare classifier accuracy (e.g., SVM) on the full 64-dimensional digits dataset vs PCA-reduced versions (5, 10, 20, 50 components). Find the sweet spot where accuracy is preserved with far fewer features.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

# --- Load high-dimensional dataset ---
digits = load_digits()
X, y = digits.data, digits.target
print(f"Original shape: {X.shape} ({X.shape[1]} features)")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Full PCA analysis ---
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scree plot and cumulative variance
ax = axes[0, 0]
var_explained = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(var_explained)

ax.bar(range(1, len(var_explained)+1), var_explained, alpha=0.6, label='Individual')
ax.step(range(1, len(cumulative_var)+1), cumulative_var, 'r-', where='mid',
        linewidth=2, label='Cumulative')
ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
n_95 = np.argmax(cumulative_var >= 0.95) + 1
ax.axvline(x=n_95, color='green', linestyle=':', alpha=0.7)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained')
ax.set_title(f'Scree Plot ({n_95} components for 95%)')
ax.set_xlim(0, 40)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 2D projection
ax = axes[0, 1]
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
plt.colorbar(scatter, ax=ax, ticks=range(10), label='Digit')
var_2d = pca_2d.explained_variance_ratio_.sum() * 100
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'Digits Projected to 2D ({var_2d:.1f}% variance)')
ax.grid(True, alpha=0.3)

# 3. Principal component images
ax = axes[1, 0]
n_show = 8
for i in range(n_show):
    ax_sub = fig.add_axes([0.05 + (i % 4) * 0.11, 0.32 - (i // 4) * 0.12, 0.09, 0.09])
    component = pca_full.components_[i].reshape(8, 8)
    ax_sub.imshow(component, cmap='RdBu_r')
    ax_sub.set_title(f'PC{i+1}', fontsize=8)
    ax_sub.axis('off')
axes[1, 0].set_title('First 8 Principal Components (as 8x8 images)')
axes[1, 0].axis('off')

# 4. Accuracy vs number of components
ax = axes[1, 1]
n_components_list = [2, 5, 10, 15, 20, 30, 40, 50, 64]
accuracies = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_comp in n_components_list:
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp),
        SVC(kernel='rbf', random_state=42)
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    accuracies.append(scores.mean())
    print(f"  PCA({n_comp:>2d}): Accuracy={scores.mean():.3f} +/- {scores.std():.3f}, "
          f"Variance retained={cumulative_var[n_comp-1]*100:.1f}%")

ax.plot(n_components_list, accuracies, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of PCA Components')
ax.set_ylabel('Cross-Validation Accuracy')
ax.set_title('SVM Accuracy vs PCA Components')
ax.grid(True, alpha=0.3)

# Add variance retained as secondary x-axis info
for n_comp, acc in zip(n_components_list, accuracies):
    var_pct = cumulative_var[n_comp-1]*100
    ax.annotate(f'{var_pct:.0f}%', (n_comp, acc), textcoords="offset points",
                xytext=(0, 10), fontsize=7, ha='center', color='red')

ax.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.show()

# --- Summary ---
print(f"\n=== Summary ===")
print(f"Original dimensionality: {X.shape[1]}")
print(f"Components for 90% variance: {np.argmax(cumulative_var >= 0.90) + 1}")
print(f"Components for 95% variance: {n_95}")
print(f"Components for 99% variance: {np.argmax(cumulative_var >= 0.99) + 1}")
```

---

## Key Takeaways

- **PCA finds the directions of maximum variance and projects data onto them.** It produces a lower-dimensional representation that retains the most information possible for any linear method.
- **Always standardize features before PCA.** Without scaling, PCA is dominated by features with the largest absolute values, not the most informative ones.
- **The scree plot guides dimension selection.** Look for the "elbow" or choose enough components to retain 90-95% of cumulative variance.
- **PCA enables visualization of high-dimensional data.** Projecting to 2D or 3D using the top principal components often reveals cluster structure that is invisible in individual feature plots.
- **Reduced dimensions can improve downstream models.** Fewer features means faster training, less overfitting, and sometimes even better accuracy by removing noise captured in low-variance components.
