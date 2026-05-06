# Accuracy and Its Limits

> Phase 4 — Model Evaluation & Selection | Kata 4.1

---

## Concept & Intuition

### What problem are we solving?

You trained a classifier and it says "95% accuracy." Is that good? It depends entirely on the problem. If you are classifying cats vs dogs with equal class sizes, 95% is solid. If you are detecting a rare disease that affects 1% of patients, a model that always says "healthy" achieves 99% accuracy while being completely useless. **Accuracy measures overall correctness, but it hides critical failures when classes are imbalanced or when different types of errors have different costs.**

Accuracy is defined as (correct predictions) / (total predictions). It treats every prediction equally -- a true positive and a true negative count the same, and a false positive and false negative are equally bad. In reality, these errors almost never carry equal weight. Missing a cancer diagnosis (false negative) is far worse than a false alarm (false positive). Accuracy cannot capture these asymmetries.

Understanding when accuracy works and when it misleads is the first step toward becoming a thoughtful model evaluator. This kata builds the foundation for all the evaluation metrics that follow.

### Why naive approaches fail

Reporting a single accuracy number creates a dangerous illusion of understanding. A model with 98% accuracy on imbalanced fraud data might detect 0% of actual fraud. A model with 70% accuracy on a 3-class problem might be perfect on two classes and random on the third. Without breaking accuracy down by class and error type, you cannot diagnose what is actually happening.

Another common failure is comparing accuracy across different datasets or class distributions. A model with 80% accuracy on a hard, balanced problem might be far better than a model with 95% accuracy on an easy, imbalanced problem. Accuracy numbers are not comparable unless the underlying distributions are identical.

### Mental models

- **Accuracy as a grade point average**: A student with a 3.5 GPA might have all B+'s or a mix of A's and C's. The average hides the distribution. Accuracy is the GPA of your classifier.
- **The majority class baseline**: Before celebrating accuracy, ask: "What would a model that always predicts the majority class achieve?" If your model only beats that by a few percent, it has barely learned anything useful.
- **Error types as different currencies**: A false positive (false alarm) and a false negative (missed detection) are different currencies. Accuracy treats them as the same, but in medical screening, a missed cancer is worth 1000 false alarms.

### Visual explanations

```
The Accuracy Paradox:

  Dataset: 1000 patients, 10 actually sick (1%)

  Model A (always predicts "healthy"):
    Correct: 990/1000 = 99.0% accuracy
    Sick patients caught: 0/10 = 0% recall
    --> USELESS despite high accuracy

  Model B (actual classifier):
    Correct: 950/1000 = 95.0% accuracy
    Sick patients caught: 9/10 = 90% recall
    --> USEFUL despite lower accuracy

Confusion Matrix (binary):

                    Predicted
                  Pos        Neg
  Actual Pos  [ TP=9    |  FN=1  ]   <-- False Negative (missed!)
         Neg  [ FP=41   |  TN=949]   <-- False Positive (false alarm)

  Accuracy = (TP + TN) / Total = (9 + 949) / 1000 = 95.8%
  But: 41 healthy people were incorrectly flagged
       1 sick person was missed
```

---

## Hands-on Exploration

1. Create a 95/5 imbalanced dataset. Train a classifier and a "dummy" classifier that always predicts the majority class. Compare their accuracies -- the dummy might win.
2. Build confusion matrices for both models. Compute accuracy, then manually compute TP, FP, TN, FN. Notice how the numbers tell a completely different story from accuracy alone.
3. Create a 3-class problem where one class is much rarer. Compute overall accuracy and per-class accuracy. Find the class the model struggles with that overall accuracy hides.
4. Implement a baseline comparison: always report accuracy alongside the majority-class baseline (no-information rate) to provide context for your accuracy numbers.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              ConfusionMatrixDisplay, classification_report)
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- Create imbalanced dataset ---
X, y = make_classification(
    n_samples=2000, n_features=10, n_redundant=2, n_informative=5,
    n_clusters_per_class=1, weights=[0.95, 0.05],
    flip_y=0.01, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Class distribution: {Counter(y_test)}")
majority_rate = Counter(y_test).most_common(1)[0][1] / len(y_test)
print(f"Majority class baseline (no-information rate): {majority_rate:.3f}")

# --- Train models ---
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_s, y_train)

lr_default = LogisticRegression(random_state=42, max_iter=1000)
lr_default.fit(X_train_s, y_train)

lr_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_balanced.fit(X_train_s, y_train)

models = {
    'Always Majority': dummy,
    'Default LR': lr_default,
    'Balanced LR': lr_balanced,
}

# --- Compare ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    # Compute detailed metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Minority recall (sensitivity): {tp/(tp+fn) if (tp+fn) > 0 else 0:.3f}")
    print(f"  Minority precision: {tp/(tp+fp) if (tp+fp) > 0 else 0:.3f}")

    ax = axes[idx]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Majority', 'Minority'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{name}\nAccuracy={acc:.3f}')

plt.tight_layout()
plt.show()

# --- Accuracy vs class balance ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
imbalance_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
baseline_accs = []
model_accs = []

for ratio in imbalance_ratios:
    X_i, y_i = make_classification(n_samples=2000, n_features=10, n_redundant=2,
                                    n_informative=5, weights=[ratio, 1-ratio],
                                    flip_y=0.01, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X_i, y_i, test_size=0.3, stratify=y_i, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    baseline_accs.append(max(Counter(yte).values()) / len(yte))

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(Xtr, ytr)
    model_accs.append(accuracy_score(yte, lr.predict(Xte)))

ax.plot(imbalance_ratios, baseline_accs, 'r--o', linewidth=2, label='Majority Class Baseline')
ax.plot(imbalance_ratios, model_accs, 'b-o', linewidth=2, label='Logistic Regression')
ax.fill_between(imbalance_ratios, baseline_accs, model_accs, alpha=0.2, color='green',
                label='Model value-add')
ax.set_xlabel('Majority Class Proportion')
ax.set_ylabel('Accuracy')
ax.set_title('The Accuracy Paradox: Model vs Baseline')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.02)
plt.tight_layout()
plt.show()

print("\nLesson: As imbalance increases, accuracy becomes less informative.")
print("Always compare your model accuracy to the majority-class baseline.")
```

---

## Key Takeaways

- **Accuracy is (TP+TN) / Total.** It treats all correct predictions equally and all errors equally, which is rarely appropriate in practice.
- **The accuracy paradox: high accuracy can coexist with zero usefulness.** On imbalanced data, always predicting the majority class gives high accuracy but catches nothing.
- **Always report accuracy alongside the majority-class baseline.** If your model is only 2% better than always guessing the majority, it has learned very little.
- **The confusion matrix tells the full story.** Break accuracy down into TP, FP, TN, FN to understand what the model actually gets right and wrong.
- **Different errors have different costs.** Missing a disease (FN) vs false alarm (FP) are not equivalent -- accuracy cannot capture this asymmetry, which is why we need precision, recall, and other metrics.
