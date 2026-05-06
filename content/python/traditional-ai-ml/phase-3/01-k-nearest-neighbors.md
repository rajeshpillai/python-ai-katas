# K-Nearest Neighbors

> Phase 3 — Supervised Learning: Classification | Kata 3.1

---

## Concept & Intuition

### What problem are we solving?

Given a new data point, how do we classify it based on what we already know? K-Nearest Neighbors (KNN) answers this with a beautifully simple idea: **look at the closest known data points and let them vote.** If most of your nearest neighbors are cats, you are probably a cat too.

KNN is a **lazy learner** -- it does not build an explicit model during training. Instead, it memorizes the entire training set and defers all computation to prediction time. This makes training instant but prediction expensive, especially with large datasets. Despite its simplicity, KNN can produce surprisingly accurate results and serves as an excellent baseline classifier.

The algorithm hinges on three critical decisions: how many neighbors to consider (k), how to measure "closeness" (distance metric), and how to weight the votes. Each of these choices dramatically affects the classifier's behavior, and understanding them is essential for applying KNN effectively.

### Why naive approaches fail

Using k=1 (just the single nearest neighbor) makes the classifier extremely sensitive to noise. A single mislabeled or outlier training point creates a pocket of wrong predictions around it. Conversely, setting k too high washes out local structure -- if k equals the entire dataset, every point gets classified as the majority class regardless of position.

The **curse of dimensionality** is KNN's fundamental weakness. In high-dimensional spaces, all points become nearly equidistant from each other. When every point is roughly the same distance away, the concept of "nearest" becomes meaningless. A dataset with hundreds of features will often produce poor KNN results unless dimensionality reduction is applied first.

### Mental models

- **Voting by neighbors**: You move to a new city and want to predict the local politics. You ask your 5 nearest neighbors how they vote -- majority wins.
- **Distance is identity**: KNN assumes that similar inputs produce similar outputs. Close in feature space means close in label space.
- **k as a smoothing dial**: Small k = jagged, noisy decision boundaries. Large k = smooth, blurry boundaries. You are tuning the resolution of the classifier.

### Visual explanations

```
k=1 (overfit, noisy)          k=7 (smoother boundary)

  . . . A A A A                . . . . A A A
  . . A A A A A                . . . A A A A
  . B B A A A A                . . B B A A A
  B B B B A A A       ->       B B B B B A A
  B B B B B . .                B B B B B . .
  B B B B . . .                B B B B . . .

  ^-- boundary is jagged        ^-- boundary is smooth


Distance metrics:

  Euclidean (L2):  sqrt((x1-x2)^2 + (y1-y2)^2)   -- "as the crow flies"
  Manhattan (L1):  |x1-x2| + |y1-y2|               -- "city block distance"
  Minkowski (Lp):  (|x1-x2|^p + |y1-y2|^p)^(1/p)  -- generalization

  Point A (1,1)  to  Point B (4,5):
    Euclidean: sqrt(9+16) = 5.0
    Manhattan: 3 + 4 = 7.0
```

---

## Hands-on Exploration

1. Generate a 2D dataset with two classes using `make_classification`. Train KNN with k=1, k=5, and k=20. Plot the decision boundaries for each and observe how they change from jagged to smooth.
2. Normalize your features using `StandardScaler` before fitting KNN. Compare accuracy with and without scaling on a dataset where features have very different ranges (e.g., age 0-100 vs salary 30000-200000).
3. Increase the number of features from 2 to 50 to 200 using `make_classification(n_features=...)`. Observe how KNN accuracy degrades as dimensionality grows -- this is the curse of dimensionality in action.
4. Try `weights='distance'` instead of `weights='uniform'`. This makes closer neighbors count more. Compare the two weighting schemes on a noisy dataset.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# @param n_neighbors int 1 20 5

# --- Generate a 2D classification dataset ---
X, y = make_classification(
    n_samples=300, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Scale features (critical for distance-based methods) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train KNN with the selected k ---
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"k = {n_neighbors}, Accuracy = {acc:.3f}")

# --- Plot decision boundary ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision boundary
ax = axes[0]
h = 0.05
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=30)
ax.set_title(f'Decision Boundary (k={n_neighbors})')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Accuracy vs k
ax = axes[1]
k_range = range(1, 21)
accuracies = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test_scaled)))

ax.plot(k_range, accuracies, 'bo-', linewidth=2)
ax.axvline(x=n_neighbors, color='red', linestyle='--', label=f'Current k={n_neighbors}')
ax.set_xlabel('k (Number of Neighbors)')
ax.set_ylabel('Test Accuracy')
ax.set_title('Accuracy vs k')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Curse of dimensionality demonstration ---
print("\n--- Curse of Dimensionality ---")
for n_feat in [2, 10, 50, 100, 200]:
    X_d, y_d = make_classification(n_samples=300, n_features=n_feat,
                                    n_informative=2, n_redundant=0,
                                    random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X_d, y_d, test_size=0.3, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    m = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
    print(f"  Features={n_feat:>3d}, Accuracy={accuracy_score(yte, m.predict(Xte)):.3f}")
```

---

## Key Takeaways

- **KNN classifies by majority vote of nearest neighbors.** No explicit model is built -- the training data IS the model.
- **Feature scaling is mandatory.** Distance metrics are dominated by features with larger ranges unless you normalize first.
- **k controls the bias-variance tradeoff.** Small k = low bias, high variance (noisy). Large k = high bias, low variance (smooth).
- **The curse of dimensionality is real.** In high dimensions, distances converge and KNN loses its discriminative power. Reduce dimensions first when working with many features.
- **KNN is a strong baseline.** Before reaching for complex models, try KNN -- it often performs surprisingly well on small to medium datasets.
