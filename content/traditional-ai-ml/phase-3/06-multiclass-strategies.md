# Multiclass Strategies

> Phase 3 — Supervised Learning: Classification | Kata 3.6

---

## Concept & Intuition

### What problem are we solving?

Many classification algorithms are inherently binary -- they separate two classes. But real-world problems often involve multiple classes: classifying handwritten digits (0-9), categorizing news articles, identifying animal species. How do we extend binary classifiers to handle multiple classes? There are two main strategies: **One-vs-Rest (OvR)**, which trains one binary classifier per class, and **One-vs-One (OvO)**, which trains one classifier per pair of classes. Some algorithms like softmax regression handle multiclass natively.

One-vs-Rest trains N classifiers for N classes. Each classifier learns to distinguish "class k vs everything else." At prediction time, we run all N classifiers and pick the one with the highest confidence. One-vs-One trains N*(N-1)/2 classifiers, one for each pair. At prediction time, each classifier votes for one of its two classes, and the class with the most votes wins.

Understanding these strategies is critical because they affect training time, prediction time, and the quality of probability estimates. The confusion matrix becomes an essential tool for understanding where a multiclass classifier struggles -- which classes get confused with which.

### Why naive approaches fail

Simply thresholding continuous outputs from a binary classifier for multiclass problems creates inconsistent probability estimates. If class A gets score 0.6 and class B gets score 0.7, both classifiers claim their class is "more likely than not" -- but they cannot both be right. OvR with proper calibration or softmax normalization addresses this by ensuring probabilities sum to 1.

Ignoring the multiclass structure can also lead to missing important error patterns. Overall accuracy of 90% might mean one class is being classified perfectly while another is misclassified 50% of the time. The confusion matrix reveals these hidden failures.

### Mental models

- **One-vs-Rest as spotlight**: Each classifier shines a spotlight on one class. "Is it a cat? Is it a dog? Is it a bird?" The brightest spotlight wins.
- **One-vs-One as tournament**: Every pair of classes plays a match. "Cat vs dog? Cat wins. Cat vs bird? Cat wins. Dog vs bird? Bird wins." The one with the most match wins takes the title.
- **Confusion matrix as a report card**: Rows are actual classes, columns are predictions. Diagonal = correct, off-diagonal = mistakes. It tells you exactly where the model struggles.

### Visual explanations

```
One-vs-Rest (OvR) for 3 classes:

  Classifier 1: A vs {B, C}     Classifier 2: B vs {A, C}     Classifier 3: C vs {A, B}
  +--------+                     +--------+                     +--------+
  | A  A  |                      |   B    |                     |      C |
  | A     | not-A                | B  B   | not-B               |    C C | not-C
  +--------+                     +--------+                     +--------+

  Prediction: pick the classifier with highest confidence

One-vs-One (OvO) for 3 classes:

  Classifier 1: A vs B    Classifier 2: A vs C    Classifier 3: B vs C
  A wins: 1 vote           A wins: 1 vote           B wins: 1 vote
  Total: A=2, B=1, C=0 --> Predict A

Confusion Matrix:

                  Predicted
              Cat   Dog   Bird
  Actual Cat [ 45     3     2 ]   <-- Cat sometimes confused with Dog
       Dog [   5    40     5 ]   <-- Dog confused with both
       Bird [  1     4    45 ]   <-- Bird sometimes called Dog
```

---

## Hands-on Exploration

1. Train a Logistic Regression on the Iris dataset (3 classes). Print `model.coef_.shape` to see that scikit-learn automatically creates 3 sets of weights (OvR). Compare with `multi_class='multinomial'` (softmax).
2. Build a confusion matrix for a multiclass classifier. Identify which classes are most confused with each other. Use `ConfusionMatrixDisplay` for visualization.
3. Train an SVM with `decision_function_shape='ovr'` and then with `'ovo'`. Compare the number of internal classifiers and prediction agreement.
4. Compute per-class precision, recall, and F1 using `classification_report`. Notice how overall accuracy can hide poor performance on minority classes.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              ConfusionMatrixDisplay, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# --- Load multiclass dataset ---
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Compare strategies ---
print("\n=== Multiclass Strategies ===")

# 1. OvR with Logistic Regression
lr_ovr = LogisticRegression(multi_class='ovr', max_iter=5000, random_state=42)
lr_ovr.fit(X_train_s, y_train)
acc_ovr = accuracy_score(y_test, lr_ovr.predict(X_test_s))
print(f"Logistic Regression (OvR): Accuracy={acc_ovr:.3f}, Coef shape={lr_ovr.coef_.shape}")

# 2. Multinomial (Softmax) Logistic Regression
lr_softmax = LogisticRegression(multi_class='multinomial', max_iter=5000, random_state=42)
lr_softmax.fit(X_train_s, y_train)
acc_softmax = accuracy_score(y_test, lr_softmax.predict(X_test_s))
print(f"Logistic Regression (Softmax): Accuracy={acc_softmax:.3f}")

# 3. SVM with OvO (default for SVC)
svm_ovo = SVC(kernel='rbf', random_state=42)
svm_ovo.fit(X_train_s, y_train)
acc_svm = accuracy_score(y_test, svm_ovo.predict(X_test_s))
n_classifiers_ovo = 10 * 9 // 2  # N*(N-1)/2
print(f"SVM (OvO): Accuracy={acc_svm:.3f}, Internal classifiers={n_classifiers_ovo}")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Confusion matrix
ax = axes[0]
y_pred = lr_softmax.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix (Softmax LR)')

# 2. Per-class accuracy
ax = axes[1]
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors = plt.cm.RdYlGn(per_class_acc)
bars = ax.bar(range(10), per_class_acc, color=colors, edgecolor='black')
ax.set_xlabel('Digit Class')
ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Accuracy')
ax.set_xticks(range(10))
ax.axhline(y=acc_softmax, color='red', linestyle='--', label=f'Overall={acc_softmax:.3f}')
ax.set_ylim(0.8, 1.02)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Softmax probability distribution for a test sample
ax = axes[2]
sample_idx = 0
probs = lr_softmax.predict_proba(X_test_s[sample_idx:sample_idx+1])[0]
true_label = y_test[sample_idx]
bar_colors = ['green' if i == true_label else 'steelblue' for i in range(10)]
ax.bar(range(10), probs, color=bar_colors, edgecolor='black')
ax.set_xlabel('Class')
ax.set_ylabel('Probability')
ax.set_title(f'Softmax Probabilities (True={true_label})')
ax.set_xticks(range(10))
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# --- Most confused pairs ---
print("\n=== Most Confused Pairs ===")
np.fill_diagonal(cm, 0)
confused_pairs = []
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            confused_pairs.append((cm[i, j], i, j))
confused_pairs.sort(reverse=True)
for count, actual, predicted in confused_pairs[:5]:
    print(f"  {actual} misclassified as {predicted}: {count} times")

print(f"\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=[str(d) for d in digits.target_names]))
```

---

## Key Takeaways

- **One-vs-Rest trains N classifiers; One-vs-One trains N*(N-1)/2.** OvR is faster to train with many classes; OvO can be more accurate but scales poorly.
- **Softmax provides calibrated multiclass probabilities.** Unlike OvR which can produce inconsistent probabilities, softmax ensures all class probabilities sum to 1.
- **The confusion matrix is essential for multiclass problems.** It reveals which specific classes are confused with each other, guiding targeted improvements.
- **Per-class metrics matter more than overall accuracy.** A 95% accurate model might completely fail on one rare class -- always check per-class precision, recall, and F1.
- **Most sklearn classifiers handle multiclass automatically.** Logistic Regression, Decision Trees, KNN, and others have built-in multiclass support, but understanding the underlying strategy helps diagnose problems.
