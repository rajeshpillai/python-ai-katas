# Precision, Recall, and F1

> Phase 4 — Model Evaluation & Selection | Kata 4.2

---

## Concept & Intuition

### What problem are we solving?

When accuracy fails us on imbalanced data, we need metrics that focus specifically on the class we care about. **Precision** answers: "Of all items I flagged as positive, how many actually were?" **Recall** answers: "Of all actual positives, how many did I catch?" These two metrics pull in opposite directions, creating a fundamental tradeoff that the **F1 score** attempts to balance.

Precision = TP / (TP + FP). A model with high precision rarely cries wolf -- when it says "positive," you can trust it. Recall = TP / (TP + FN). A model with high recall rarely misses real positives -- it catches almost everything. The F1 score is the harmonic mean of precision and recall: 2 * P * R / (P + R). It requires both to be high for the F1 to be high -- a model with 100% precision and 0% recall gets F1 = 0.

The decision threshold is the hidden lever that controls this tradeoff. Most classifiers output a probability, and we choose a cutoff (default 0.5) above which we predict positive. Lowering the threshold catches more positives (higher recall) but increases false alarms (lower precision). Raising the threshold reduces false alarms (higher precision) but misses more positives (lower recall).

### Why naive approaches fail

Optimizing for just one metric leads to degenerate solutions. A model that predicts positive for everything achieves 100% recall but terrible precision. A model that only predicts positive when it is absolutely certain achieves near-100% precision but catches very few cases. Neither extreme is useful.

Using the default 0.5 threshold is also often wrong. On imbalanced data, the optimal threshold might be 0.1 or 0.01 because the model learns to assign low probabilities to rare events. Tuning the threshold to maximize F1 (or another target metric) on a validation set can dramatically improve practical performance without changing the model at all.

### Mental models

- **Precision as a purity filter**: Of everything the model puts in the "positive" bucket, what fraction actually belongs there? High precision = pure bucket.
- **Recall as a dragnet**: Of all the fish in the ocean, what fraction does the net catch? High recall = thorough net.
- **F1 as a compromise judge**: F1 punishes imbalance between precision and recall. P=0.9, R=0.9 gives F1=0.9. But P=1.0, R=0.1 gives F1=0.18. It demands both be good.
- **Threshold as a sensitivity dial**: Low threshold = sensitive (catch everything, many false alarms). High threshold = specific (fewer catches, almost no false alarms).

### Visual explanations

```
Precision vs Recall:

  Precision = TP / (TP + FP)     "How pure are my positive predictions?"
  Recall    = TP / (TP + FN)     "How complete is my positive detection?"

  All items:  [P P P P P N N N N N N N N N N N N N N N]
  Model flags:        ^^^^^^^^^^^^
                    [P P P P N N N]
                     TP=4   FP=3        FN=1

  Precision = 4/7 = 0.57    (flagged 7, only 4 were right)
  Recall    = 4/5 = 0.80    (there were 5 positives, caught 4)

Threshold Effect:

  threshold=0.3:  flag 15 items -> TP=4, FP=11 -> P=0.27, R=0.80
  threshold=0.5:  flag  8 items -> TP=4, FP= 4 -> P=0.50, R=0.80
  threshold=0.7:  flag  5 items -> TP=3, FP= 2 -> P=0.60, R=0.60
  threshold=0.9:  flag  2 items -> TP=2, FP= 0 -> P=1.00, R=0.40

  P     1.0 |*
  r         | *
  e         |  *
  c     0.5 |    *
  i         |      *
  s         |        *
  n     0.0 +-----------> Recall
            0   0.5   1.0
```

---

## Hands-on Exploration

1. Compute precision, recall, and F1 by hand from a confusion matrix. Verify your answers against `sklearn.metrics.precision_score`, `recall_score`, and `f1_score`.
2. Train a classifier on imbalanced data and sweep the decision threshold from 0.0 to 1.0. Plot precision and recall as functions of threshold. Find where they cross (often a good operating point).
3. Compare F1 score with F-beta scores. `fbeta_score(beta=2)` weighs recall higher (good for medical screening). `fbeta_score(beta=0.5)` weighs precision higher (good for search ranking).
4. Create a scenario where precision matters more (spam filter) and one where recall matters more (cancer screening). Choose the appropriate threshold for each.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              precision_recall_curve, confusion_matrix,
                              classification_report, fbeta_score)
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- Create imbalanced dataset ---
X, y = make_classification(
    n_samples=2000, n_features=10, n_redundant=2, n_informative=5,
    weights=[0.9, 0.1], flip_y=0.02, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Test set distribution: {Counter(y_test)}")

# --- Train model ---
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

# --- Metrics at default threshold ---
print("\n=== Default Threshold (0.5) ===")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")
print(f"F2 (recall-weighted): {fbeta_score(y_test, y_pred, beta=2):.3f}")
print(f"F0.5 (precision-weighted): {fbeta_score(y_test, y_pred, beta=0.5):.3f}")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Precision-Recall vs Threshold
ax = axes[0]
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_prob)
ax.plot(thresholds, precision_curve[:-1], 'b-', linewidth=2, label='Precision')
ax.plot(thresholds, recall_curve[:-1], 'r-', linewidth=2, label='Recall')

# F1 at each threshold
f1_at_thresh = 2 * precision_curve[:-1] * recall_curve[:-1] / (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
ax.plot(thresholds, f1_at_thresh, 'g--', linewidth=2, label='F1')

best_f1_idx = np.argmax(f1_at_thresh)
best_threshold = thresholds[best_f1_idx]
ax.axvline(x=best_threshold, color='green', linestyle=':', alpha=0.7)
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Default (0.5)')
ax.scatter(best_threshold, f1_at_thresh[best_f1_idx], c='green', s=100, zorder=5,
           label=f'Best F1={f1_at_thresh[best_f1_idx]:.2f} @ {best_threshold:.2f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision, Recall, F1 vs Threshold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
ax = axes[1]
ax.plot(recall_curve, precision_curve, 'b-', linewidth=2)
ax.fill_between(recall_curve, precision_curve, alpha=0.1, color='blue')
ax.scatter(recall_curve[best_f1_idx], precision_curve[best_f1_idx], c='red', s=100, zorder=5,
           label=f'Best F1 point')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

# 3. Confusion matrices at different thresholds
ax = axes[2]
thresholds_to_show = [0.2, best_threshold, 0.5, 0.8]
table_data = []
for t in thresholds_to_show:
    y_t = (y_prob >= t).astype(int)
    p = precision_score(y_test, y_t, zero_division=0)
    r = recall_score(y_test, y_t, zero_division=0)
    f = f1_score(y_test, y_t, zero_division=0)
    cm = confusion_matrix(y_test, y_t)
    tn, fp, fn, tp = cm.ravel()
    table_data.append([f'{t:.2f}', f'{p:.2f}', f'{r:.2f}', f'{f:.2f}', str(tp), str(fp), str(fn)])

ax.axis('off')
table = ax.table(
    cellText=table_data,
    colLabels=['Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'],
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
ax.set_title('Metrics at Different Thresholds', pad=20)

plt.tight_layout()
plt.show()

# --- Apply optimal threshold ---
print(f"\n=== Optimal Threshold ({best_threshold:.3f}) ===")
y_pred_optimal = (y_prob >= best_threshold).astype(int)
print(f"Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_optimal):.3f}")
print(f"F1:        {f1_score(y_test, y_pred_optimal):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_optimal, target_names=['Negative', 'Positive']))
```

---

## Key Takeaways

- **Precision measures prediction purity; recall measures detection completeness.** They are both needed to understand classifier performance on the positive class.
- **F1 is the harmonic mean of precision and recall.** It requires both to be high -- if either is near zero, F1 is near zero. Use F-beta to weight one over the other.
- **The decision threshold is a powerful, free parameter.** Tuning it on validation data can significantly improve F1 without retraining the model.
- **Lower thresholds increase recall at the cost of precision; higher thresholds do the opposite.** The right tradeoff depends on the application's cost structure.
- **Always report precision and recall separately, not just F1.** F1=0.6 could mean P=0.9, R=0.43 or P=0.5, R=0.75 -- very different operational profiles.
