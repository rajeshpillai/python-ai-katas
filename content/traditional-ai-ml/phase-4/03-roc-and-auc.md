# ROC and AUC

> Phase 4 — Model Evaluation & Selection | Kata 4.3

---

## Concept & Intuition

### What problem are we solving?

How do you compare two classifiers when each could be tuned to operate at any point on the precision-recall spectrum? The **Receiver Operating Characteristic (ROC) curve** provides a comprehensive answer by plotting the classifier's True Positive Rate (recall) against its False Positive Rate at every possible threshold. The **Area Under the Curve (AUC)** condenses this entire curve into a single number: 1.0 for a perfect classifier, 0.5 for random guessing.

The ROC curve shows you the full range of tradeoffs a classifier can make. At threshold=0 (predict everything as positive), you get TPR=1 but FPR=1 (top-right corner). At threshold=1 (predict everything as negative), you get TPR=0 and FPR=0 (bottom-left corner). The curve traces how TPR and FPR change as you sweep the threshold between these extremes. A good classifier hugs the top-left corner -- high TPR with low FPR.

AUC has an elegant probabilistic interpretation: it equals the probability that a randomly chosen positive example is ranked higher (given a higher predicted probability) than a randomly chosen negative example. AUC of 0.5 means the classifier's ranking is no better than random coin flips.

### Why naive approaches fail

Comparing classifiers by accuracy at a single threshold can be misleading. Model A might have lower accuracy at threshold=0.5 but a much better ROC curve overall. By comparing ROC curves, you evaluate classifiers across ALL possible operating points, making the comparison fair regardless of the threshold each happens to use.

However, ROC curves have their own limitation: on highly imbalanced data, the ROC curve can look overly optimistic. When there are 10,000 negatives and 100 positives, a jump from 0 to 500 false positives changes FPR by only 0.05 -- the ROC barely moves while 500 innocent items are being flagged. For imbalanced data, precision-recall curves are often more informative.

### Mental models

- **ROC as a dial sweep**: Imagine slowly turning a sensitivity dial from "detect nothing" to "detect everything." The ROC curve plots what happens at each dial setting -- how many true positives you gain vs how many false positives you accumulate.
- **AUC as ranking quality**: A higher AUC means the classifier assigns higher scores to positives than negatives more consistently. It is a measure of separability between classes.
- **The diagonal = random guessing**: Any point on the diagonal line (TPR = FPR) could be achieved by random coin flipping. A useful classifier must be above this line.

### Visual explanations

```
ROC Curve:

  TPR (Recall)
  1.0 |        ___------*  Perfect (AUC=1.0)
      |      /
      |    /   Good classifier (AUC=0.85)
  0.5 |  /  /
      | / /   Random (AUC=0.5)
      |//
  0.0 +--+--+--+--+---> FPR
      0  0.2 0.5 0.8 1.0

  FPR = FP / (FP + TN)  = "false alarm rate"
  TPR = TP / (TP + FN)  = "detection rate" (recall)


AUC Interpretation:

  AUC = P(score(random positive) > score(random negative))

  AUC = 1.0:  Perfect separation. All positives scored higher than all negatives.
  AUC = 0.9:  Excellent. 90% of the time, a random positive scores higher.
  AUC = 0.7:  Fair. Substantial overlap between positive and negative scores.
  AUC = 0.5:  Random. The model cannot distinguish positives from negatives.

Score Distributions:

  AUC ~ 0.95 (good separation):     AUC ~ 0.65 (poor separation):

  Neg: |||||||                       Neg:    ||||||||
  Pos:          |||||||              Pos:       ||||||||
       0   0.5   1.0                      0   0.5   1.0
```

---

## Hands-on Exploration

1. Train two classifiers (e.g., Logistic Regression and KNN) on the same dataset. Plot both ROC curves on the same axes and compare their AUC values. The higher curve dominates at all operating points.
2. Plot the score distributions (predicted probabilities) for positive and negative classes side by side. Notice how the overlap between these distributions corresponds to the AUC -- less overlap means higher AUC.
3. Pick a specific point on the ROC curve and compute the corresponding threshold, confusion matrix, precision, and recall. Understand that each point on the ROC curve represents a complete classifier configuration.
4. Compare ROC-AUC vs PR-AUC on an imbalanced dataset. Create a scenario where ROC-AUC looks good (0.95) but the classifier still has poor practical performance on the minority class.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, roc_auc_score,
                              precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler

# --- Generate dataset ---
X, y = make_classification(
    n_samples=1000, n_features=10, n_redundant=2, n_informative=5,
    weights=[0.7, 0.3], flip_y=0.05, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train multiple classifiers ---
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. ROC Curves
ax = axes[0]
colors = ['blue', 'red', 'green']
for (name, clf), color in zip(classifiers.items(), colors):
    clf.fit(X_train_s, y_train)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# 2. Score distributions for the best model
ax = axes[1]
best_model = classifiers['Logistic Regression']
y_prob = best_model.predict_proba(X_test_s)[:, 1]

ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6, color='blue', density=True, label='Negative class')
ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6, color='red', density=True, label='Positive class')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Default threshold')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title(f'Score Distributions (AUC={roc_auc_score(y_test, y_prob):.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. ROC vs PR curve comparison
ax = axes[2]
fpr, tpr, _ = roc_curve(y_test, y_prob)
prec, rec, _ = precision_recall_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)
pr_auc_val = average_precision_score(y_test, y_prob)

ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={roc_auc_val:.3f})')
ax.plot(rec, prec, 'r-', linewidth=2, label=f'PR (AP={pr_auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('FPR / Recall')
ax.set_ylabel('TPR / Precision')
ax.set_title('ROC vs Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Finding the optimal threshold from ROC ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# Youden's J statistic: maximize TPR - FPR
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"\n=== Optimal Threshold (Youden's J) ===")
print(f"Threshold: {best_threshold:.3f}")
print(f"TPR: {tpr[best_idx]:.3f}, FPR: {fpr[best_idx]:.3f}")
print(f"J statistic: {j_scores[best_idx]:.3f}")

# --- Summary comparison ---
print(f"\n=== Model Comparison ===")
for name, clf in classifiers.items():
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    print(f"  {name:>25s}: ROC-AUC={roc_auc_score(y_test, y_prob):.3f}, "
          f"PR-AUC={average_precision_score(y_test, y_prob):.3f}")
```

---

## Key Takeaways

- **The ROC curve shows TPR vs FPR across all thresholds.** It visualizes the full range of operating points a classifier can achieve, independent of any single threshold choice.
- **AUC measures ranking quality.** An AUC of 0.9 means that a random positive gets a higher score than a random negative 90% of the time.
- **AUC = 0.5 is random; AUC = 1.0 is perfect.** Any classifier worth using should be well above the diagonal baseline.
- **ROC can be overly optimistic on imbalanced data.** When negatives vastly outnumber positives, FPR changes slowly even with many false positives. Use PR curves for imbalanced problems.
- **Youden's J statistic (TPR - FPR) finds the optimal ROC threshold.** It maximizes the distance from the random-guessing diagonal, giving a principled threshold selection method.
