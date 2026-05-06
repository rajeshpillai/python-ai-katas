# Cross-Validation

> Phase 4 — Model Evaluation & Selection | Kata 4.4

---

## Concept & Intuition

### What problem are we solving?

A single train/test split is a gamble. Depending on which data points land in the test set, your accuracy estimate could be optimistically high or pessimistically low. **Cross-validation** eliminates this luck by systematically using every data point for both training and testing. In K-fold cross-validation, the data is split into K equal parts. Each part takes a turn being the test set while the remaining K-1 parts form the training set. The final score is the average across all K folds.

This gives us two critical things that a single split cannot: a **more reliable estimate** of model performance (the average across folds) and a **confidence interval** (the standard deviation across folds). If accuracy varies from 85% to 95% across folds, we know the model is sensitive to the training data. If it is consistently 89-91%, we can trust the estimate.

Cross-validation is also the foundation for model selection and hyperparameter tuning. When comparing two models or choosing a hyperparameter value, we need reliable performance estimates to make the right choice. A single split might favor the wrong model by chance; cross-validation makes the comparison robust.

### Why naive approaches fail

Evaluating on the training set gives an absurdly optimistic estimate -- the model has seen these exact points. A single train/test split is better but noisy -- the estimate depends heavily on which points happen to fall into the test set. With small datasets, a single split can be off by 10% or more from the true performance.

A subtle but dangerous mistake is using the test set for model selection. If you try 50 hyperparameter combinations and pick the one with the best test accuracy, you have implicitly trained on the test set. The reported accuracy is now optimistic. The correct approach uses three sets: train (fit the model), validation (choose hyperparameters), and test (final, untouched evaluation) -- or equivalently, cross-validation for selection and a held-out test set for final evaluation.

### Mental models

- **K-fold as rotation**: Imagine 5 students taking turns being the teacher's assistant while the other 4 study. Each student gets evaluated by a different "teacher." The average of all 5 evaluations is more fair than any single one.
- **Standard deviation as confidence**: If cross-validation scores are [88, 90, 87, 91, 89], the std is ~1.5%. You can be confident the true performance is near 89%. If scores are [70, 95, 80, 65, 92], the std is ~12% -- something is unstable.
- **Stratified as representative**: In stratified K-fold, each fold has the same class proportions as the full dataset. This prevents a fold from accidentally having no minority class examples.

### Visual explanations

```
5-Fold Cross-Validation:

  Data: [========================================]

  Fold 1: [TEST ][  train   |  train   |  train  ]  -> Score: 0.88
  Fold 2: [train][  TEST    |  train   |  train  ]  -> Score: 0.91
  Fold 3: [train][  train   |  TEST    |  train  ]  -> Score: 0.87
  Fold 4: [train][  train   |  train   |  TEST   ]  -> Score: 0.90
  Fold 5: [train][  train   |  train   |  ] TEST ]  -> Score: 0.89

  Average: 0.89 +/- 0.015

  Every point is tested exactly once and trained on exactly 4 times.


Stratified K-Fold (preserves class balance):

  Original:    70% A, 30% B

  Fold 1 test: 70% A, 30% B    (same proportions)
  Fold 2 test: 70% A, 30% B    (same proportions)
  ...

  vs Regular K-Fold (might get):

  Fold 1 test: 90% A, 10% B    (unrepresentative!)
  Fold 2 test: 50% A, 50% B    (unrepresentative!)
```

---

## Hands-on Exploration

1. Compare a single train/test split vs 5-fold CV vs 10-fold CV. Run each 20 times with different random seeds. Plot the distribution of accuracy estimates. Notice how CV produces tighter, more reliable estimates.
2. Use `StratifiedKFold` on an imbalanced dataset. Print the class distribution in each fold and verify it matches the overall distribution. Compare with regular `KFold`.
3. Implement leave-one-out cross-validation (LOOCV) on a small dataset. Notice it uses N-1 points for training and 1 for testing, repeated N times. Compare its estimate with K-fold.
4. Use cross-validation to compare two models (e.g., Logistic Regression vs KNN). Report mean and std for both. Use a paired t-test to determine if the difference is statistically significant.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                      KFold, LeaveOneOut, train_test_split)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# @param n_folds int 2 20 5

# --- Generate dataset ---
X, y = make_classification(
    n_samples=500, n_features=10, n_redundant=2, n_informative=5,
    weights=[0.7, 0.3], flip_y=0.05, random_state=42
)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {np.bincount(y)}")

# --- Compare single split vs K-fold CV ---
print(f"\n=== Single Split vs {n_folds}-Fold CV ===")

# Single splits with different seeds
single_scores = []
for seed in range(50):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(Xtr, ytr)
    single_scores.append(accuracy_score(yte, lr.predict(Xte)))

# K-fold cross-validation
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

print(f"Single split: {np.mean(single_scores):.3f} +/- {np.std(single_scores):.3f} (50 random splits)")
print(f"{n_folds}-Fold CV:    {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Distribution of estimates
ax = axes[0]
ax.hist(single_scores, bins=15, alpha=0.6, color='blue', label=f'Single split (n=50)', density=True)
ax.axvline(x=np.mean(single_scores), color='blue', linestyle='--', linewidth=2)

# Plot CV folds as individual points
for i, score in enumerate(cv_scores):
    ax.axvline(x=score, color='red', linestyle='-', alpha=0.4, linewidth=1)
ax.axvline(x=cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'{n_folds}-Fold CV mean')

ax.set_xlabel('Accuracy')
ax.set_ylabel('Density')
ax.set_title(f'Stability: Single Split vs {n_folds}-Fold CV')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. CV scores per fold
ax = axes[1]
fold_numbers = range(1, n_folds + 1)
colors = plt.cm.Set2(np.linspace(0, 1, n_folds))
bars = ax.bar(fold_numbers, cv_scores, color=colors, edgecolor='black')
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={cv_scores.mean():.3f}')
ax.fill_between([0.5, n_folds + 0.5],
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.2, color='red', label=f'+/- 1 std={cv_scores.std():.3f}')
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy')
ax.set_title(f'{n_folds}-Fold CV Scores')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Compare models using CV
ax = axes[2]
models = {
    'Logistic Reg': make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)),
    'KNN (k=5)': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
}

model_scores = {}
positions = range(len(models))
for i, (name, model) in enumerate(models.items()):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    model_scores[name] = scores
    ax.boxplot(scores, positions=[i], widths=0.5)
    ax.scatter([i] * len(scores), scores, c='red', s=30, zorder=5)

ax.set_xticks(list(positions))
ax.set_xticklabels(list(models.keys()), rotation=15)
ax.set_ylabel('Accuracy')
ax.set_title(f'Model Comparison ({n_folds}-Fold CV)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# --- Summary ---
print(f"\n=== Model Comparison ({n_folds}-Fold CV) ===")
for name, scores in model_scores.items():
    print(f"  {name:>15s}: {scores.mean():.3f} +/- {scores.std():.3f}  "
          f"[{', '.join(f'{s:.3f}' for s in scores)}]")

# --- Effect of K on CV estimate ---
print(f"\n=== Effect of Number of Folds ===")
for k in [2, 3, 5, 10, 20]:
    cv_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores_k = cross_val_score(pipeline, X, y, cv=cv_k, scoring='accuracy')
    print(f"  {k:>2d}-fold: {scores_k.mean():.3f} +/- {scores_k.std():.3f}")
```

---

## Key Takeaways

- **Cross-validation uses every data point for both training and testing.** This gives a more reliable and less noisy performance estimate than a single train/test split.
- **The mean score estimates performance; the standard deviation estimates reliability.** A large std means the model is sensitive to the specific training data and may not generalize stably.
- **Stratified K-fold preserves class proportions in each fold.** This is essential for imbalanced datasets and generally recommended for all classification tasks.
- **Use cross-validation for model selection, a held-out test set for final evaluation.** Never report cross-validation scores on data used to choose hyperparameters as your final metric.
- **5 or 10 folds is the standard choice.** More folds give slightly less biased estimates but take longer and have higher variance. Leave-one-out is N-fold and is computationally expensive but useful for very small datasets.
