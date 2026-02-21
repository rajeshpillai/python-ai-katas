# Imbalanced Classes

> Phase 3 — Supervised Learning: Classification | Kata 3.7

---

## Concept & Intuition

### What problem are we solving?

In the real world, classes are rarely balanced. Fraud detection has 99.9% legitimate transactions and 0.1% fraud. Disease screening has 99% healthy patients and 1% sick. A classifier that simply predicts "not fraud" for everything achieves 99.9% accuracy -- and catches zero fraud. When classes are imbalanced, **accuracy is a misleading metric** and standard training procedures are biased toward the majority class.

Imbalanced classification requires a different mindset. We need to either rebalance the data (oversampling the minority, undersampling the majority, or generating synthetic examples with SMOTE), adjust the algorithm (class weights, cost-sensitive learning), or change our evaluation metrics (precision, recall, F1, PR-AUC instead of accuracy). Often, the best approach combines all three.

The precision-recall tradeoff becomes central. In fraud detection, we might accept many false alarms (low precision) to catch every fraud case (high recall). In spam filtering, we might prefer high precision (no legitimate email in spam) even if some spam gets through (lower recall). The right balance depends entirely on the application's cost structure.

### Why naive approaches fail

Standard classifiers optimize overall accuracy, which means they minimize total errors. When one class is 99% of the data, the cheapest way to minimize errors is to classify everything as the majority class. The algorithm is not broken -- it is correctly optimizing the wrong objective.

Similarly, random oversampling (duplicating minority examples) can cause overfitting because the classifier memorizes the exact duplicated points. Random undersampling throws away potentially valuable majority class data. SMOTE addresses both issues by creating synthetic examples that are similar-but-not-identical to existing minority points, but it can also create noisy examples in overlapping regions.

### Mental models

- **Needle in a haystack**: The minority class is the needle. Standard training optimizes for "classify hay correctly" because there is so much hay. We need to tell the model: "missing a needle is 100x worse than misclassifying a piece of hay."
- **SMOTE as interpolation**: SMOTE picks a minority example, finds its minority neighbor, and creates a new point somewhere between them. It is "imagining" what other minority examples might look like.
- **Class weights as cost adjustment**: Setting `class_weight='balanced'` tells the model: "A mistake on a minority example costs N times more" (where N is the imbalance ratio).

### Visual explanations

```
Imbalanced Dataset:                  After SMOTE:

  . . . . . . . . .                  . . . . . . . . .
  . . . . . . . . .                  . . . . . . . . .
  . . . . . . . . .                  . . . * * . . . .
  . . . . * * . . .                  . . * * * * . . .
  . . . . . . . . .                  . . . * * . . . .
  . . . . . . . . .                  . . . . . . . . .

  . = majority class                  * = minority (original + synthetic)
  * = minority class (rare)

Precision-Recall Tradeoff:

  High threshold (0.9):  Precision HIGH, Recall LOW
    --> Only flag very confident cases. Few false alarms, but miss many.

  Low threshold (0.1):   Precision LOW, Recall HIGH
    --> Flag everything suspicious. Catch most fraud, but many false alarms.

  Threshold    Precision    Recall
   0.1          0.05        0.98     <-- catch almost all fraud
   0.3          0.15        0.90
   0.5          0.40        0.70     <-- balanced
   0.7          0.75        0.40
   0.9          0.95        0.10     <-- very few false alarms
```

---

## Hands-on Exploration

1. Create a highly imbalanced dataset (95/5 split). Train a default classifier and observe that it achieves high accuracy while completely ignoring the minority class. Print the confusion matrix to see the problem.
2. Apply `class_weight='balanced'` to Logistic Regression and compare. Notice how recall for the minority class improves dramatically, often at the cost of majority class precision.
3. Implement SMOTE from `imblearn` (or manually by interpolating between minority points). Compare the training set before and after SMOTE. Visualize the synthetic points.
4. Plot precision-recall curves at different thresholds. Find the threshold that maximizes F1 score. Compare this with the default 0.5 threshold.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              precision_recall_curve, f1_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- Create imbalanced dataset ---
X, y = make_classification(
    n_samples=2000, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, weights=[0.95, 0.05],
    flip_y=0.02, random_state=42
)

print(f"Class distribution: {Counter(y)}")
print(f"Imbalance ratio: {Counter(y)[0] / Counter(y)[1]:.1f}:1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Manual SMOTE implementation ---
def simple_smote(X_minority, n_synthetic, k=5, random_state=42):
    """Generate synthetic minority samples by interpolation."""
    rng = np.random.RandomState(random_state)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_minority)
    neighbors = nn.kneighbors(X_minority, return_distance=False)[:, 1:]

    synthetic = []
    for _ in range(n_synthetic):
        idx = rng.randint(len(X_minority))
        neighbor_idx = rng.choice(neighbors[idx])
        lam = rng.random()
        new_point = X_minority[idx] + lam * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)
    return np.array(synthetic)

# Generate synthetic minority points
X_minority = X_train_s[y_train == 1]
X_majority = X_train_s[y_train == 0]
n_synthetic = len(X_majority) - len(X_minority)
X_synthetic = simple_smote(X_minority, n_synthetic)

X_train_smote = np.vstack([X_train_s, X_synthetic])
y_train_smote = np.hstack([y_train, np.ones(n_synthetic, dtype=int)])

# --- Train three models ---
models = {
    'Default': LogisticRegression(random_state=42),
    'Class Weights': LogisticRegression(class_weight='balanced', random_state=42),
    'SMOTE': LogisticRegression(random_state=42),
}

train_data = {
    'Default': (X_train_s, y_train),
    'Class Weights': (X_train_s, y_train),
    'SMOTE': (X_train_smote, y_train_smote),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (name, model) in enumerate(models.items()):
    Xt, yt = train_data[name]
    model.fit(Xt, yt)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1_min = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}, Minority F1: {f1_min:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    # Decision boundary
    ax = axes[0, idx]
    h = 0.05
    x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
    y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_train_s[y_train == 0, 0], X_train_s[y_train == 0, 1],
               c='blue', s=10, alpha=0.3, label='Majority')
    ax.scatter(X_train_s[y_train == 1, 0], X_train_s[y_train == 1, 1],
               c='red', s=40, edgecolors='k', label='Minority')
    if name == 'SMOTE':
        ax.scatter(X_synthetic[:, 0], X_synthetic[:, 1],
                   c='orange', s=20, alpha=0.5, marker='x', label='Synthetic')
    ax.set_title(f'{name}\nAcc={acc:.3f}, F1(min)={f1_min:.3f}')
    ax.legend(fontsize=7)

    # Precision-Recall curve
    ax = axes[1, idx]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.scatter(recall[best_idx], precision[best_idx], c='red', s=100, zorder=5,
               label=f'Best F1={f1_scores[best_idx]:.2f} @ thresh={thresholds[best_idx]:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve ({name})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

- **Accuracy is meaningless on imbalanced data.** A 99% accurate model on 99/1 data might just be predicting the majority class every time. Always check the confusion matrix.
- **Class weights are the simplest fix.** Setting `class_weight='balanced'` tells the algorithm to penalize minority misclassification proportionally to the imbalance ratio.
- **SMOTE creates synthetic minority examples by interpolation.** It generates new points between existing minority neighbors, expanding the minority class without exact duplication.
- **Precision-recall tradeoff is application-dependent.** Fraud detection needs high recall (catch all fraud). Spam filtering needs high precision (never lose real email). There is no universal right answer.
- **Stratified splitting preserves class ratios.** Always use `stratify=y` in `train_test_split` when dealing with imbalanced data to ensure both train and test sets have the same class proportions.
