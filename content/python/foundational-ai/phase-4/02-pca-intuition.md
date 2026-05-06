# PCA Intuition

> Phase 4 — Representation Learning | Kata 4.2

---

## Concept & Intuition

### What problem are we solving?

Real-world data often lives in high-dimensional spaces -- an image might have thousands of pixels, a dataset might have hundreds of columns. But much of that dimensionality is redundant. Many features are correlated, and the data often lives on or near a lower-dimensional surface. **Principal Component Analysis (PCA)** finds the directions in which your data varies the most and lets you project it down to fewer dimensions while preserving as much information as possible.

Think of it this way: if you have 2D data that forms an elongated cloud tilted at an angle, PCA finds the long axis of that cloud (the direction of maximum variance) and the short axis (the direction of minimum variance). By keeping only the long axis, you compress 2D data into 1D while losing very little information -- the short axis was mostly noise anyway.

PCA is one of the simplest and most foundational forms of representation learning. Unlike neural network hidden layers, PCA is linear -- it finds the best linear projection. But the core insight is the same: find a new coordinate system that captures what matters and discards what does not.

### Why naive approaches fail

A naive approach to dimensionality reduction would be to simply drop columns -- maybe remove features that "look" unimportant. But this throws away information blindly. Two features that individually seem weak might together encode a strong signal along a diagonal direction that neither axis captures alone.

Another naive approach is to pick dimensions at random or use the original axes. But the original axes (e.g., "height" and "weight") are rarely aligned with the directions where data actually varies most. PCA rotates the coordinate system to align with the data's natural axes of variation, ensuring you keep the most informative directions first.

### Mental models

| Analogy | Explanation |
|---------|-------------|
| **Shadow on a wall** | PCA is like finding the angle to hold a flashlight so the shadow of a 3D object on the wall looks as spread out (informative) as possible. |
| **Spinning a scatter plot** | Imagine rotating a 3D scatter plot until the data looks most spread out on your screen -- that rotation is PCA. |
| **Compressing a photo** | Keep the big shapes (high variance), discard the tiny pixel noise (low variance). |

### Visual explanations

```
  Original 2D data (correlated):      After PCA rotation:

       y                                  PC2 (small variance)
       |    . .                            |
       |   . . . .                         |  . .
       |  . . . .                      ----...... -------> PC1
       | . . .                             |  . .        (large variance)
       |. .                                |
       +-----------> x                 We can drop PC2 and keep
                                       only PC1 with minimal loss!

  Eigenvalues tell you HOW MUCH variance each PC captures:

  PC1: ========================================  (92%)
  PC2: ====                                      ( 8%)
       ^
       Most info is in PC1 -- safe to drop PC2
```

---

## Hands-on Exploration

1. Generate a 2D dataset with clear correlation, then compute PCA from scratch using the covariance matrix and eigendecomposition.
2. Project the data onto the principal components and see how much variance each component explains.
3. Reduce from 2D to 1D and reconstruct the original data -- measure the reconstruction error to see what was lost.

---

## Live Code

```python
import numpy as np
np.random.seed(42)

# --- Generate correlated 2D data ---
n = 200
x1 = np.random.randn(n)
x2 = 0.8 * x1 + 0.3 * np.random.randn(n)  # correlated with x1
X = np.column_stack([x1, x2])

print("=== PCA from Scratch ===")
print(f"Data shape: {X.shape}  (200 points in 2D)")

# Step 1: Center the data
mean = X.mean(axis=0)
Xc = X - mean
print(f"\nData mean:  [{mean[0]:.3f}, {mean[1]:.3f}]")

# Step 2: Covariance matrix
cov = (Xc.T @ Xc) / (n - 1)
print(f"\nCovariance matrix:")
print(f"  [{cov[0,0]:6.3f}  {cov[0,1]:6.3f}]")
print(f"  [{cov[1,0]:6.3f}  {cov[1,1]:6.3f}]")
print(f"  Off-diagonal = {cov[0,1]:.3f} --> features are correlated!")

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sort descending by eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nEigenvalues: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")
total_var = eigenvalues.sum()
print(f"Variance explained: PC1={eigenvalues[0]/total_var*100:.1f}%,"
      f" PC2={eigenvalues[1]/total_var*100:.1f}%")

print(f"\nPrincipal components (directions):")
print(f"  PC1: [{eigenvectors[0,0]:+.3f}, {eigenvectors[1,0]:+.3f}]")
print(f"  PC2: [{eigenvectors[0,1]:+.3f}, {eigenvectors[1,1]:+.3f}]")

# Step 4: Project data onto PCs
projected = Xc @ eigenvectors   # full projection (2D -> 2D rotated)
proj_1d = Xc @ eigenvectors[:, 0:1]  # keep only PC1 (2D -> 1D)

# Step 5: Reconstruct from 1D back to 2D
reconstructed = proj_1d @ eigenvectors[:, 0:1].T + mean

# Measure reconstruction error
error = np.mean(np.sum((X - reconstructed)**2, axis=1))
print(f"\n=== Dimensionality Reduction: 2D -> 1D ===")
print(f"Mean reconstruction error: {error:.4f}")
print(f"(Compare to total variance:  {total_var:.4f})")
print(f"Information retained: {(1 - error/total_var)*100:.1f}%")

# Visualize: text-based scatter of original vs PC axes
print("\n=== Text Visualization: First 10 points ===")
print(f"  {'Original (x1, x2)':<24} {'PC1 score':>10} {'Reconstructed (x1, x2)'}")
for i in range(10):
    orig = f"({X[i,0]:+5.2f}, {X[i,1]:+5.2f})"
    score = f"{proj_1d[i,0]:+6.3f}"
    recon = f"({reconstructed[i,0]:+5.2f}, {reconstructed[i,1]:+5.2f})"
    print(f"  {orig:<24} {score:>10}   {recon}")

# Show variance bar chart
print("\n=== Variance Explained (bar chart) ===")
for i, ev in enumerate(eigenvalues):
    pct = ev / total_var * 100
    bar = "#" * int(pct / 2)
    print(f"  PC{i+1}: {bar:<50} {pct:5.1f}%")
```

---

## Key Takeaways

- **PCA finds the axes of maximum variance.** It rotates your coordinate system to align with the directions where data spreads most.
- **Eigenvalues quantify importance.** Each eigenvalue tells you how much variance (information) its principal component captures.
- **Dimensionality reduction means dropping low-variance PCs.** If a component explains only 5% of variance, you can often discard it with minimal loss.
- **PCA is a linear method.** It can only find linear combinations of features -- neural networks learn nonlinear representations, which is why they are more powerful for complex data.
- **Reconstruction error measures what you lost.** The gap between original and reconstructed data quantifies the cost of reducing dimensions.
