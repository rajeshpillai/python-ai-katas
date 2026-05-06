# Support Vector Machines

> Phase 3 — Supervised Learning: Classification | Kata 3.4

---

## Concept & Intuition

### What problem are we solving?

Many algorithms can find *a* line that separates two classes, but which line is the *best*? Support Vector Machines (SVMs) answer this definitively: **the best boundary is the one that maximizes the margin** -- the distance between the boundary and the nearest data points from each class. These nearest points are called **support vectors**, and they alone determine the boundary. Every other data point could move or disappear without changing the model.

This maximum-margin principle gives SVMs excellent generalization. By pushing the decision boundary as far as possible from both classes, the model creates the largest "safety zone" for new, unseen data points. If a new point falls within this wide margin, it is still classified correctly even if it is slightly different from training examples.

The real power of SVMs comes from the **kernel trick**. When data is not linearly separable in the original space, a kernel function implicitly maps it to a higher-dimensional space where a linear separator exists. The radial basis function (RBF) kernel, for instance, can create complex, non-linear decision boundaries without explicitly computing the high-dimensional features.

### Why naive approaches fail

A simple linear classifier might find any separating line, but if it runs close to data points on one side, it has essentially zero margin there. New points in that tight area are likely to be misclassified. SVMs formalize the intuition that we want to be as far from danger as possible on all sides.

When data is not linearly separable (classes overlap), a hard-margin SVM fails completely -- it cannot find any separating hyperplane. The soft-margin SVM (controlled by the C parameter) allows some misclassifications in exchange for a wider margin. Setting C too high forces near-zero errors on training data (overfitting), while setting C too low allows too many errors (underfitting).

### Mental models

- **The widest road**: Imagine drawing a road between two groups of buildings. SVM finds the widest possible road. The buildings touching the road edges are the support vectors.
- **Kernel as a magnifying glass**: When points overlap in 2D, a kernel "lifts" them into 3D where they separate cleanly. The RBF kernel creates bumps around each point.
- **C as a strictness dial**: High C = strict teacher (no mistakes allowed, may overfit). Low C = lenient teacher (some mistakes OK, smoother boundary).

### Visual explanations

```
Hard Margin SVM:                   Soft Margin SVM (some violations):

  o         o                       o    |    o
  o    |    x  x                    o   x|    x  x
  o    |      x                     o    |  o   x
  o    |    x                       o    |    x
       |  x    x                         |  x    x
  ^----^----^                        (allows misclassified points)
  margin  margin

  |  = decision boundary
  o  = class 0
  x  = class 1
  ^  = support vectors (nearest points)

Kernel Trick (2D -> 3D):

  Original 2D (not separable):     Projected 3D (separable):
                                          |
  x x o o o x x                     x x  |  x x
                                     o o o|
                                          |
                                   (can now draw a flat plane)
```

---

## Hands-on Exploration

1. Generate linearly separable 2D data. Train a linear SVM and plot the decision boundary, margin lines, and support vectors (use `model.support_vectors_`). Observe how only a few points determine the boundary.
2. Create non-linearly separable data (e.g., `make_circles` or `make_moons`). Try a linear kernel and then an RBF kernel. Compare the decision boundaries.
3. Vary the C parameter from 0.01 to 1000. Plot how the margin width and number of support vectors change. Low C = wide margin + more support vectors; high C = narrow margin + fewer support vectors.
4. Experiment with the RBF `gamma` parameter. Low gamma = smooth, wide-reaching influence. High gamma = sharp, local influence (each point creates a tiny island). Plot boundaries for gamma = 0.1, 1, 10, 100.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Generate non-linear dataset ---
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Helper to plot decision boundary ---
def plot_svm_boundary(ax, model, X, y, title):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=30)
    # Highlight support vectors
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100, linewidths=2,
               facecolors='none', edgecolors='black', label=f'SVs ({len(sv)})')
    ax.set_title(title)
    ax.legend(fontsize=8)

# --- Compare kernels ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

kernels = [('linear', 'Linear Kernel'), ('rbf', 'RBF Kernel'), ('poly', 'Polynomial Kernel (d=3)')]
for ax, (kernel, title) in zip(axes, kernels):
    params = {'kernel': kernel, 'C': 1.0, 'random_state': 42}
    if kernel == 'poly':
        params['degree'] = 3
    svm = SVC(**params)
    svm.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test_s))
    plot_svm_boundary(ax, svm, X_train_s, y_train, f'{title}\nAcc={acc:.3f}')

plt.suptitle('SVM Kernel Comparison on Moon Dataset', fontsize=14)
plt.tight_layout()
plt.show()

# --- Effect of C parameter ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
C_values = [0.01, 0.1, 1.0, 100.0]

for ax, C in zip(axes, C_values):
    svm = SVC(kernel='rbf', C=C, random_state=42)
    svm.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test_s))
    plot_svm_boundary(ax, svm, X_train_s, y_train, f'C={C}\nAcc={acc:.3f}')

plt.suptitle('Effect of C (Regularization) on RBF SVM', fontsize=14)
plt.tight_layout()
plt.show()

# --- Effect of gamma ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
gamma_values = [0.1, 1.0, 10.0, 100.0]

for ax, gamma in zip(axes, gamma_values):
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
    svm.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test_s))
    plot_svm_boundary(ax, svm, X_train_s, y_train, f'gamma={gamma}\nAcc={acc:.3f}')

plt.suptitle('Effect of Gamma on RBF SVM', fontsize=14)
plt.tight_layout()
plt.show()

# --- Summary ---
print("--- Summary ---")
for kernel in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=kernel, C=1.0, random_state=42)
    svm.fit(X_train_s, y_train)
    print(f"  {kernel:>6s}: Accuracy={accuracy_score(y_test, svm.predict(X_test_s)):.3f}, "
          f"Support Vectors={len(svm.support_vectors_)}")
```

---

## Key Takeaways

- **SVMs find the maximum-margin boundary.** The widest possible gap between classes gives the best generalization to unseen data.
- **Support vectors are the only points that matter.** The boundary is determined entirely by the nearest points to it -- all other training data is irrelevant.
- **The kernel trick enables non-linear classification.** RBF, polynomial, and other kernels project data to higher dimensions where a linear separator exists, without the computational cost of explicitly computing the new features.
- **C controls the margin-error tradeoff.** High C = narrow margin, few errors (risk of overfitting). Low C = wide margin, more errors (risk of underfitting).
- **Gamma controls the influence radius of each point.** High gamma means each point has a tiny influence (complex, local boundaries). Low gamma means each point has a wide influence (smooth, global boundaries).
