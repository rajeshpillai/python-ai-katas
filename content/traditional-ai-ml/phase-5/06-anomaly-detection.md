# Anomaly Detection

> Phase 5 — Unsupervised Learning | Kata 5.6

---

## Concept & Intuition

### What problem are we solving?

How do you find the unusual, the unexpected, the things that do not fit? **Anomaly detection** identifies data points that deviate significantly from the norm. This is different from classification: in anomaly detection, you typically have many "normal" examples but few or no "anomalous" examples to learn from. The model learns what "normal" looks like and flags anything that deviates.

Applications are everywhere: fraud detection (unusual transactions), network security (unusual traffic patterns), manufacturing quality control (defective products), medical diagnostics (unusual test results), and system monitoring (server failures). In all these cases, anomalies are rare, costly, and critically important to detect.

Three powerful approaches dominate. **Isolation Forest** isolates anomalies by randomly partitioning the feature space -- anomalies, being rare and different, get isolated in fewer partitions (shorter tree paths). **One-Class SVM** learns a boundary that encloses "normal" data; anything outside is anomalous. **Statistical methods** model the distribution of normal data and flag points in the low-probability tails. Each approach has different strengths depending on the data structure and dimensionality.

### Why naive approaches fail

Simple threshold rules (e.g., "flag if value > 3 standard deviations") only work for univariate data and assume Gaussian distributions. Real anomalies are often subtle, multivariate, and context-dependent. A transaction of $500 is normal for one customer but anomalous for another. A temperature of 100F is normal in summer but anomalous in winter.

Treating anomaly detection as supervised classification also fails because labeled anomaly data is scarce. You might have millions of normal transactions but only 50 known frauds. A classifier trained on such extreme imbalance will either ignore the minority class entirely or overfit to the few examples it has. Unsupervised anomaly detection learns the structure of "normal" without needing anomaly labels.

### Mental models

- **Isolation Forest as "easy to separate"**: If you randomly draw lines to split data, normal points need many splits to isolate (they are surrounded by similar points). Anomalies need very few splits (they sit alone in empty space). Shorter paths = more anomalous.
- **One-Class SVM as a flexible boundary**: It draws a contour around the normal data like an amoeba's membrane. Points outside the membrane are flagged as anomalous.
- **Anomaly score as "weirdness"**: Each method assigns a score. Highly negative scores (Isolation Forest) or far-from-boundary distances (One-Class SVM) indicate anomalies. You choose a threshold based on your tolerance for false alarms.

### Visual explanations

```
Isolation Forest:

  Normal point (many splits needed):    Anomaly (few splits needed):

  . . . . . . . . . .                  . . . . . . . . . .
  . . . .[. . .]. . .                  . . . . . . . . . .
  . . . .[. [X]]. . .                  . . . . . . . .[X].
  . . . .[. . .]. . .                  . . . . . . . . . .
  . . . . . . . . . .

  X needs 4 splits to isolate           X needs 2 splits to isolate
  (surrounded by neighbors)             (alone in empty space)
  --> Normal                            --> Anomaly


One-Class SVM:

  . . . . . . . . . . . .
  . .  ___________  . . .
  . . /  . . .    \ . . .
  . ./  . . . .  . \ . .
  . |  . . . . .  . | X  <-- outside boundary = anomaly
  . .\  . . . .  . / . .
  . . \___________/ . . .
  . . . . . . . . . . . .
```

---

## Hands-on Exploration

1. Generate 2D "normal" data from a Gaussian distribution, then add a few outlier points. Train an Isolation Forest and visualize which points get flagged as anomalies. Experiment with the `contamination` parameter.
2. Plot the anomaly score surface: for a grid of points, compute the Isolation Forest anomaly score and create a contour plot. Normal regions should be clearly separated from anomalous regions.
3. Compare Isolation Forest with One-Class SVM on the same dataset. One-Class SVM draws a smooth boundary; Isolation Forest creates a more complex, axis-aligned boundary.
4. Try anomaly detection on a real-world-like scenario: generate multivariate normal data representing normal server metrics, then inject a few "attack" patterns with different statistical properties. Measure detection performance.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Generate data with anomalies ---
np.random.seed(42)

# Normal data
n_normal = 300
X_normal = np.random.randn(n_normal, 2) * 0.8 + np.array([2, 2])

# Anomalies (scattered outliers)
n_anomalies = 20
X_anomalies = np.random.uniform(low=-2, high=6, size=(n_anomalies, 2))

X = np.vstack([X_normal, X_anomalies])
y_true = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])  # 1=normal, -1=anomaly

print(f"Dataset: {n_normal} normal + {n_anomalies} anomalies")
contamination = n_anomalies / len(X)
print(f"Contamination rate: {contamination:.2%}")

# --- Train anomaly detectors ---
detectors = {
    'Isolation Forest': IsolationForest(contamination=contamination, random_state=42),
    'One-Class SVM': OneClassSVM(kernel='rbf', gamma='scale', nu=contamination),
    'Elliptic Envelope': EllipticEnvelope(contamination=contamination, random_state=42),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

for idx, (name, detector) in enumerate(detectors.items()):
    detector.fit(X)
    y_pred = detector.predict(X)

    # Metrics
    p = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"  Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
    print(f"  Detected anomalies: {(y_pred == -1).sum()}")

    # Decision boundary
    ax = axes[0, idx]
    h = 0.1
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if hasattr(detector, 'decision_function'):
        Z = detector.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = detector.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    # Plot points
    normal_mask = y_pred == 1
    anomaly_mask = y_pred == -1
    ax.scatter(X[normal_mask, 0], X[normal_mask, 1], c='green', s=20, alpha=0.5, label='Normal')
    ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], c='red', s=60, marker='x',
               linewidths=2, label='Anomaly')
    ax.set_title(f'{name}\nP={p:.2f}, R={r:.2f}, F1={f1:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# --- Anomaly score analysis ---
# Use Isolation Forest for detailed analysis
iso = IsolationForest(contamination=contamination, random_state=42)
iso.fit(X)
scores = iso.decision_function(X)

# Score distribution
ax = axes[1, 0]
ax.hist(scores[y_true == 1], bins=30, alpha=0.6, color='green', density=True, label='Normal')
ax.hist(scores[y_true == -1], bins=15, alpha=0.6, color='red', density=True, label='True Anomaly')
ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision boundary')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Density')
ax.set_title('Isolation Forest Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Contamination sensitivity
ax = axes[1, 1]
contamination_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
precisions, recalls, f1s = [], [], []

for cont in contamination_values:
    iso_c = IsolationForest(contamination=cont, random_state=42)
    iso_c.fit(X)
    y_pred_c = iso_c.predict(X)
    precisions.append(precision_score(y_true, y_pred_c, pos_label=-1, zero_division=0))
    recalls.append(recall_score(y_true, y_pred_c, pos_label=-1, zero_division=0))
    f1s.append(f1_score(y_true, y_pred_c, pos_label=-1, zero_division=0))

ax.plot(contamination_values, precisions, 'b-o', label='Precision')
ax.plot(contamination_values, recalls, 'r-o', label='Recall')
ax.plot(contamination_values, f1s, 'g-o', label='F1')
ax.axvline(x=contamination, color='gray', linestyle='--', label=f'True rate={contamination:.2f}')
ax.set_xlabel('Contamination Parameter')
ax.set_ylabel('Score')
ax.set_title('Sensitivity to Contamination')
ax.legend()
ax.grid(True, alpha=0.3)

# Feature importance via path length
ax = axes[1, 2]
path_lengths = -scores  # Higher = more anomalous
sorted_idx = np.argsort(path_lengths)[::-1]
top_k = 30
colors = ['red' if y_true[i] == -1 else 'green' for i in sorted_idx[:top_k]]
ax.barh(range(top_k), path_lengths[sorted_idx[:top_k]], color=colors, edgecolor='black')
ax.set_xlabel('Anomaly Score (higher = more anomalous)')
ax.set_ylabel('Point Index (sorted)')
ax.set_title(f'Top {top_k} Most Anomalous Points\n(red=true anomaly, green=normal)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# --- Summary ---
print("\n=== Method Comparison ===")
print(f"{'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 55)
for name, detector in detectors.items():
    y_pred = detector.predict(X)
    p = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
    print(f"{name:<22} {p:>10.3f} {r:>10.3f} {f1:>10.3f}")
```

---

## Key Takeaways

- **Anomaly detection learns "normal" and flags deviations.** Unlike classification, it typically requires only normal examples for training, making it suitable when anomaly labels are scarce.
- **Isolation Forest is fast and effective.** It isolates anomalies through random partitioning -- anomalies have shorter average path lengths because they sit in sparse regions.
- **One-Class SVM provides smooth boundaries.** It learns a contour around normal data using the kernel trick, making it good for complex boundary shapes but slower on large datasets.
- **The contamination parameter is critical.** It controls the fraction of points flagged as anomalous. Setting it too low misses anomalies; too high generates excessive false alarms.
- **Anomaly detection is inherently unsupervised.** Even when some labeled anomalies exist, using them for validation (not training) is often more effective than trying to build a supervised classifier on extreme class imbalance.
