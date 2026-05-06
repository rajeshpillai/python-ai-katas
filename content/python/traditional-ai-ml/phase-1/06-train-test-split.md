# Train-Test Split

> Phase 1 — Data Wrangling | Kata 1.6

---

## Concept & Intuition

### What problem are we solving?

The fundamental question in machine learning is: "Will this model work on data it has never seen before?" A model that memorizes the training data perfectly but fails on new data is useless. The **train-test split** is the simplest and most important technique for estimating how well a model will generalize. By holding out a portion of the data (the test set) that the model never sees during training, we can evaluate performance on truly unseen data.

The idea is simple: split your dataset into two parts. Train the model on one part (typically 70-80%) and evaluate it on the other (20-30%). The test set score is your best estimate of real-world performance. Without this discipline, you are flying blind — a model that reports 99% accuracy on training data might only achieve 60% on new data due to overfitting.

Beyond the basic holdout split, there are critical nuances: **stratification** ensures that the class distribution in each split matches the original data (essential for imbalanced datasets), **random state** controls reproducibility, and **data leakage** — the most insidious mistake in ML — occurs when information from the test set contaminates the training process, producing misleading results that collapse in production.

### Why naive approaches fail

The most common mistake is evaluating on the training data itself. A sufficiently complex model (like a deep decision tree) can memorize every training example, achieving 100% training accuracy while being completely useless on new data. This is overfitting, and the train-test split is the first line of defense against it.

A subtler mistake is performing data preprocessing (like feature scaling, encoding, or imputation) on the entire dataset before splitting. When you compute the mean salary across ALL data and use it to impute missing values, the imputed values in the "test set" contain information from the "training set" — and vice versa. This data leakage makes your test scores optimistically biased. The correct practice is to split FIRST, then preprocess using only training data.

### Mental models

- **The exam analogy**: The training set is the textbook you study from. The test set is the exam. If you have seen the exam questions while studying (data leakage), your exam score does not reflect what you actually learned
- **The time machine problem**: In many real applications, you are predicting the future from the past. Your test set should represent "future" data, and your training set "past" data. Any information from the "future" leaking into the "past" is a violation of causality
- **Stratification as fairness**: If 5% of your data is class A and 95% is class B, a random split might put 0% of class A in the test set. Stratification ensures both sets reflect the true proportions
- **The reproducibility contract**: Setting a random seed means anyone can reproduce your exact split. This is essential for debugging, comparison, and scientific reproducibility

### Visual explanations

```
Basic Train-Test Split
========================

  Full Dataset (1000 samples)
  ┌────────────────────────────────────────────────────────────┐
  │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■│
  └────────────────────────────────────────────────────────────┘

  After 80/20 split:
  ┌──────────────────────────────────────────────┐┌───────────┐
  │         Training Set (800 samples)           ││ Test Set   │
  │   Model learns from this data                ││ (200)      │
  │   fit(), train()                             ││ score()    │
  └──────────────────────────────────────────────┘└───────────┘


Stratification (for imbalanced classes)
========================================

  Original:  90% Class 0 (■), 10% Class 1 (●)
  ┌────────────────────────────────────────────────────────────┐
  │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ● ■ ■ ■ ■ ■ ■ ■ ■ ● ■ ■ ■ ■ ■ ● ■│
  └────────────────────────────────────────────────────────────┘

  Without stratification (RISKY):
  Train: 93% Class 0, 7% Class 1    <- Imbalanced!
  Test:  80% Class 0, 20% Class 1   <- Different distribution!

  With stratification (SAFE):
  Train: 90% Class 0, 10% Class 1   <- Matches original
  Test:  90% Class 0, 10% Class 1   <- Matches original


Data Leakage
==============

  WRONG ORDER:                    CORRECT ORDER:
  1. Load all data                1. Load all data
  2. Scale all data  ← LEAK!     2. Split into train/test
  3. Split into train/test        3. Scale train data (fit_transform)
  4. Train model                  4. Scale test data (transform only!)
  5. Evaluate on test             5. Train model on scaled train
                                  6. Evaluate on scaled test
```

---

## Hands-on Exploration

1. **Basic split**: Run the code below to perform a train-test split. Verify that the sizes match the specified ratio and that the class distribution is preserved with stratification.

2. **Overfitting detection**: Compare training accuracy vs. test accuracy. A large gap indicates overfitting. Try different model complexities to see how the gap changes.

3. **Stratification matters**: Remove the `stratify` parameter and re-split multiple times with different random seeds. Observe how the class distribution varies in each split — especially problematic for rare classes.

4. **Data leakage experiment**: Run the leakage demonstration in the code. Compare the "leaked" test score vs. the "honest" test score. Even a small difference means your production performance will be worse than expected.

---

## Live Code

```python
"""
Train-Test Split — Holdout validation, stratification, and data leakage.

This code demonstrates proper train-test splitting, the importance of
stratification, and the insidious effects of data leakage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

# ============================================================
# Part 1: Basic Train-Test Split
# ============================================================

np.random.seed(42)

# Create an imbalanced classification dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
    n_clusters_per_class=2, weights=[0.85, 0.15], random_state=42,
)

print("=" * 60)
print("TRAIN-TEST SPLIT — Fundamentals")
print("=" * 60)

print(f"\nFull dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Class 1 proportion: {y.mean():.2%}")

# Split WITHOUT stratification
X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None,
)

# Split WITH stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

print(f"\n--- Split Sizes ---")
print(f"Training: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.0%})")
print(f"Test:     {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.0%})")

print(f"\n--- Class Distribution Comparison ---")
print(f"{'Set':<25} {'Class 0':>10} {'Class 1':>10} {'Class 1 %':>12}")
print("-" * 60)
for name, labels in [("Original", y),
                     ("Train (no stratify)", y_train_ns),
                     ("Test (no stratify)", y_test_ns),
                     ("Train (stratified)", y_train),
                     ("Test (stratified)", y_test)]:
    c0 = (labels == 0).sum()
    c1 = (labels == 1).sum()
    print(f"{name:<25} {c0:>10} {c1:>10} {c1/len(labels):>11.2%}")

# ============================================================
# Part 2: Overfitting Detection
# ============================================================

print("\n" + "=" * 60)
print("OVERFITTING DETECTION: Train vs Test Accuracy")
print("=" * 60)

depths = [1, 2, 3, 5, 7, 10, 15, 20, 30, None]
train_scores = []
test_scores = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))
    depth_str = str(depth) if depth is not None else "None"
    print(f"  max_depth={depth_str:>4}  Train: {train_scores[-1]:.4f}  "
          f"Test: {test_scores[-1]:.4f}  Gap: {train_scores[-1] - test_scores[-1]:.4f}")

# ============================================================
# Part 3: Data Leakage Demonstration
# ============================================================

print("\n" + "=" * 60)
print("DATA LEAKAGE DEMONSTRATION")
print("=" * 60)

# WRONG: Scale before splitting (leakage!)
scaler_wrong = StandardScaler()
X_scaled_all = scaler_wrong.fit_transform(X)
X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
    X_scaled_all, y, test_size=0.2, random_state=42, stratify=y,
)
clf_leak = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_leak.fit(X_train_leak, y_train_leak)
score_leak = clf_leak.score(X_test_leak, y_test_leak)

# CORRECT: Scale after splitting (no leakage)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
scaler_correct = StandardScaler()
X_train_scaled = scaler_correct.fit_transform(X_train_c)  # Fit on train
X_test_scaled = scaler_correct.transform(X_test_c)         # Transform test
clf_correct = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_correct.fit(X_train_scaled, y_train_c)
score_correct = clf_correct.score(X_test_scaled, y_test_c)

print(f"\n  WRONG (scale then split): Test accuracy = {score_leak:.4f}")
print(f"  RIGHT (split then scale): Test accuracy = {score_correct:.4f}")
print(f"  Leakage bias: {score_leak - score_correct:+.4f}")
print(f"\n  Note: Even a small bias compounds across many preprocessing steps!")

# BEST: Use a Pipeline (prevents leakage automatically)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
])
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"\n  BEST (Pipeline + cross-validation): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ============================================================
# Part 4: Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Overfitting curve
ax = axes[0, 0]
depth_labels = [str(d) if d is not None else "None" for d in depths]
ax.plot(range(len(depths)), train_scores, "bo-", linewidth=2, markersize=8, label="Train")
ax.plot(range(len(depths)), test_scores, "ro-", linewidth=2, markersize=8, label="Test")
ax.fill_between(range(len(depths)), train_scores, test_scores, alpha=0.2, color="gray",
                label="Overfitting gap")
ax.set_xticks(range(len(depths)))
ax.set_xticklabels(depth_labels, fontsize=9)
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Overfitting: Train vs Test Accuracy", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# Stratification comparison
ax = axes[0, 1]
n_trials = 50
class1_ratios_unstrat = []
class1_ratios_strat = []
for seed in range(n_trials):
    _, _, _, yt_ns = train_test_split(X, y, test_size=0.2, random_state=seed)
    _, _, _, yt_s = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    class1_ratios_unstrat.append(yt_ns.mean())
    class1_ratios_strat.append(yt_s.mean())

ax.hist(class1_ratios_unstrat, bins=15, alpha=0.7, label="Unstratified", color="#e74c3c")
ax.hist(class1_ratios_strat, bins=15, alpha=0.7, label="Stratified", color="#2ecc71")
ax.axvline(y.mean(), color="blue", linestyle="--", linewidth=2, label=f"True ratio: {y.mean():.2%}")
ax.set_xlabel("Class 1 Ratio in Test Set")
ax.set_ylabel("Count (over 50 splits)")
ax.set_title("Stratification Reduces Variance", fontweight="bold")
ax.legend(fontsize=9)

# Data leakage impact
ax = axes[1, 0]
methods = ["Leaked\n(wrong)", "No Leak\n(correct)", "Pipeline\n(best)"]
scores_compare = [score_leak, score_correct, cv_scores.mean()]
colors = ["#e74c3c", "#2ecc71", "#3498db"]
bars = ax.bar(methods, scores_compare, color=colors, edgecolor="black", alpha=0.8)
ax.set_ylabel("Accuracy")
ax.set_title("Data Leakage Impact", fontweight="bold")
for bar, score in zip(bars, scores_compare):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{score:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)

# Correct workflow
ax = axes[1, 1]
ax.axis("off")
workflow = [
    "Correct ML Workflow",
    "",
    "1. Load raw data",
    "2. Split into train/test (stratify!)",
    "3. EDA on training data only",
    "4. Preprocess training data (fit_transform)",
    "5. Preprocess test data (transform only!)",
    "6. Train model on preprocessed train data",
    "7. Evaluate on preprocessed test data",
    "",
    "Or better: use sklearn Pipeline!",
    "Pipeline handles steps 4-7 automatically",
    "and prevents leakage by design.",
]
for i, line in enumerate(workflow):
    weight = "bold" if i == 0 or "Pipeline" in line else "normal"
    color = "black" if "WRONG" not in line else "red"
    ax.text(0.1, 0.95 - i * 0.07, line, fontsize=11, fontweight=weight,
            color=color, family="monospace", transform=ax.transAxes)
ax.set_title("The Right Way", fontweight="bold")

plt.tight_layout()
plt.show()

print("\nKey insight: The train-test split is your first line of defense")
print("against overfitting. Stratification preserves class balance,")
print("and pipelines prevent data leakage by design.")
```

---

## Key Takeaways

- **Never evaluate on training data**: The gap between training and test accuracy is your primary indicator of overfitting — a large gap means the model memorized rather than learned
- **Stratification preserves class distribution**: For imbalanced datasets, always use `stratify=y` to ensure both train and test sets reflect the true class proportions
- **Data leakage is the silent killer**: Any preprocessing that uses information from the test set (even computing a mean) contaminates your evaluation and produces overoptimistic results
- **Use sklearn Pipelines**: Pipelines automatically ensure that preprocessing is fit on training data only and applied consistently to test data, eliminating leakage by design
- **Reproducibility requires a random seed**: Always set `random_state` so that your splits are reproducible, enabling fair comparisons and debugging
