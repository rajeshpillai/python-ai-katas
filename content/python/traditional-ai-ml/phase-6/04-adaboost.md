# AdaBoost

> Phase 6 — Ensemble Methods | Kata 6.4

---

## Concept & Intuition

### What problem are we solving?

AdaBoost (Adaptive Boosting) is the original boosting algorithm that turns a collection of weak classifiers into a strong one. The key mechanism is sample reweighting: after each round, misclassified examples get higher weights so the next weak learner pays more attention to them. Correctly classified examples get lower weights. Over time, the ensemble develops specialized sub-models for the hardest cases.

Each weak learner also gets a vote weight proportional to its accuracy. A learner that is 90% accurate gets a loud vote; one that is barely better than random gets a whisper. The final prediction is a weighted majority vote across all weak learners. This weighting scheme ensures that the best learners dominate the final decision while poor learners contribute minimally.

AdaBoost has a beautiful theoretical property: as long as each weak learner is slightly better than random guessing (>50% accuracy for binary classification), the training error decreases exponentially with the number of rounds. In practice, AdaBoost can achieve near-zero training error with enough rounds, and its test error often continues to decrease even after training error hits zero -- a phenomenon that puzzled researchers for years and is related to margin theory.

### Why naive approaches fail

Giving all training examples equal importance forever means the model never focuses on its mistakes. It keeps getting the easy examples right and the hard ones wrong. Uniform weighting is wasteful -- the model spends equal effort on examples it already handles perfectly and ones it consistently misclassifies. AdaBoost's adaptive reweighting is like a student who identifies their weak topics and studies those harder, rather than re-reading material they already know.

### Mental models

- **Adaptive study plan**: after each practice test, you identify which questions you got wrong and spend more time on those topics. Easy topics get less review time.
- **Weighted jury**: each juror (weak learner) gets a vote proportional to their track record. Reliable jurors have more influence than unreliable ones.
- **Exponential focusing**: sample weights grow exponentially for repeatedly misclassified points, creating an intense focus on the hardest cases.

### Visual explanations

```
Round 1:  All samples equal weight
  o o o o x x o o        (o = correct, x = misclassified)
  w: [1 1 1 1 1 1 1 1]   --> alpha_1 = high (good accuracy)

Round 2:  Misclassified get higher weight
  o o x o o o x o
  w: [. . 3 . 3 . . .]   --> alpha_2 = medium

Round 3:  Focus shifts again
  o x o o o o o o
  w: [. 5 . . . . . .]   --> alpha_3 = high

Final: alpha_1 * h1(x) + alpha_2 * h2(x) + alpha_3 * h3(x) > 0 ?

Weight Update Rule:
  Correct:   w_i *= exp(-alpha)   (weight decreases)
  Wrong:     w_i *= exp(+alpha)   (weight increases)
```

---

## Hands-on Exploration

1. Train AdaBoost on a binary classification dataset with a few clearly hard-to-classify examples near the decision boundary. Visualize how sample weights change over the first 5 rounds.
2. Extract the alpha (learner weight) for each weak learner and verify that more accurate learners get higher alpha values.
3. Plot training and test error as a function of the number of boosting rounds. Observe how training error drops rapidly and test error follows.
4. Compare AdaBoost to a single decision stump and to a single deep tree to see how many weak learners it takes to match a complex model.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

# @param n_estimators int 10 200 50

# --- Dataset ---
X, y = make_classification(
    n_samples=400, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=2, flip_y=0.05, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Train AdaBoost ---
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=n_estimators,
    learning_rate=1.0,
    algorithm="SAMME",
    random_state=42
)
ada.fit(X_train, y_train)

print("=== AdaBoost Results ===")
print(f"Number of estimators: {n_estimators}")
print(f"Train accuracy: {ada.score(X_train, y_train):.4f}")
print(f"Test accuracy:  {ada.score(X_test, y_test):.4f}\n")

# --- Baselines ---
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)
deep = DecisionTreeClassifier(max_depth=10, random_state=42)
deep.fit(X_train, y_train)

print("=== Baselines ===")
print(f"Single stump:  train={stump.score(X_train, y_train):.4f}, "
      f"test={stump.score(X_test, y_test):.4f}")
print(f"Deep tree:     train={deep.score(X_train, y_train):.4f}, "
      f"test={deep.score(X_test, y_test):.4f}\n")

# --- Staged performance (error vs rounds) ---
train_errors = []
test_errors = []
for i, (y_pred_train, y_pred_test) in enumerate(
    zip(ada.staged_predict(X_train), ada.staged_predict(X_test))
):
    train_errors.append(1 - accuracy_score(y_train, y_pred_train))
    test_errors.append(1 - accuracy_score(y_test, y_pred_test))

# --- Estimator weights (alpha values) ---
alphas = ada.estimator_weights_[:min(20, n_estimators)]
print("=== Estimator Weights (first 20) ===")
for i, a in enumerate(alphas):
    bar = "|" * int(a * 10)
    print(f"  Round {i+1:>2}: alpha = {a:.3f}  {bar}")

# --- Visualize sample weights after training ---
# Retrain with small n to show weight evolution
ada_small = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=5, algorithm="SAMME", random_state=42
)
ada_small.fit(X_train, y_train)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Error vs rounds
axes[0].plot(range(1, len(train_errors)+1), train_errors, 'b-', label="Train error")
axes[0].plot(range(1, len(test_errors)+1), test_errors, 'r-', label="Test error")
axes[0].set_xlabel("Boosting Round")
axes[0].set_ylabel("Error Rate")
axes[0].set_title("AdaBoost Error vs Rounds")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Decision boundary
h = 0.05
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = ada.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu",
                edgecolors="k", s=20)
axes[1].set_title(f"AdaBoost Decision Boundary ({n_estimators} stumps)")

# Estimator weights
axes[2].bar(range(1, len(alphas)+1), alphas, color="steelblue")
axes[2].set_xlabel("Round")
axes[2].set_ylabel("Alpha (estimator weight)")
axes[2].set_title("Weak Learner Weights")

plt.tight_layout()
plt.savefig("adaboost.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to adaboost.png")
```

---

## Key Takeaways

- **AdaBoost reweights samples to focus on mistakes.** Misclassified examples get exponentially higher weights, forcing subsequent learners to prioritize them.
- **Each weak learner gets a vote proportional to its accuracy.** The alpha weight ensures that better learners have more influence on the final prediction.
- **Training error decreases exponentially with rounds.** As long as each weak learner is slightly better than random, AdaBoost converges to high accuracy.
- **AdaBoost is sensitive to noisy data.** Outliers and mislabeled examples get increasingly high weights, which can degrade performance. This is a known weakness.
- **The algorithm is beautifully simple.** The core loop is: train weak learner, compute error, update weights, repeat. Yet the result is a powerful classifier.
