# Stacking

> Phase 6 — Ensemble Methods | Kata 6.7

---

## Concept & Intuition

### What problem are we solving?

Bagging and boosting combine models of the *same type* (usually decision trees). Stacking combines models of *different types* -- for example, a random forest, a logistic regression, and a k-NN model -- by training a meta-learner that learns how to best combine their predictions. The meta-learner figures out which base model to trust in which situation.

The idea is that different algorithms have different strengths. A linear model might be excellent in one region of the feature space while a tree-based model excels in another. Rather than choosing one algorithm, stacking lets a meta-learner learn the optimal blend. The base models (level-0) produce predictions, and those predictions become features for the meta-learner (level-1), which makes the final prediction.

A critical detail is that the base model predictions used to train the meta-learner must be out-of-fold predictions (generated via cross-validation), not predictions on the training data. If you train the base models on the full training set and then use their training predictions as meta-features, the meta-learner will be trained on overfitted signals and will not generalize. Cross-validated predictions simulate how the base models will behave on unseen data.

### Why naive approaches fail

Simple averaging gives each model equal weight regardless of quality. Weighted averaging is better but assumes the optimal weight is constant across all inputs. Stacking goes further: the meta-learner can learn that "trust the random forest when feature X is large, trust the linear model otherwise." This input-dependent combination is strictly more flexible than any fixed weighting scheme, though it comes with the risk of overfitting the meta-learner if not done carefully with proper cross-validation.

### Mental models

- **Panel of experts with a moderator**: each expert (base model) gives their opinion. The moderator (meta-learner) has learned which experts to trust for which types of questions.
- **Two-stage pipeline**: stage 1 generates diverse opinions, stage 2 synthesizes them. The separation ensures the synthesizer learns from honest (out-of-fold) opinions.
- **Blending as a simpler alternative**: instead of full cross-validation, you can use a holdout set for generating meta-features. Simpler but wastes data.

### Visual explanations

```
Level 0 (Base Models):                    Level 1 (Meta-Learner):

X_train -----> RandomForest  --> pred_rf  \
X_train -----> LogisticReg   --> pred_lr   +--> [pred_rf, pred_lr, pred_knn] --> Meta-Model --> Final
X_train -----> KNN           --> pred_knn /

CRITICAL: Base model predictions must be cross-validated (out-of-fold)!

Cross-validation stacking:
  Fold 1: Train on folds 2-5, predict fold 1  --> meta_features[fold 1]
  Fold 2: Train on folds 1,3-5, predict fold 2 --> meta_features[fold 2]
  ...
  Fold 5: Train on folds 1-4, predict fold 5  --> meta_features[fold 5]

  Meta-features = stack of out-of-fold predictions (no data leakage)
  Meta-learner trains on these honest predictions
```

---

## Hands-on Exploration

1. Train three diverse base models (Random Forest, Logistic Regression, k-NN) on a classification dataset. Compare their individual accuracies.
2. Generate out-of-fold predictions for each base model using cross-validation. Stack these predictions as meta-features and train a logistic regression meta-learner.
3. Compare the stacked ensemble accuracy to each individual model and to a simple majority vote.
4. Experiment with different meta-learners (Logistic Regression vs Decision Tree vs another Random Forest) to see which combination works best.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

np.random.seed(42)

# --- Dataset ---
X, y = make_classification(
    n_samples=600, n_features=10, n_informative=6,
    n_redundant=2, n_clusters_per_class=2,
    flip_y=0.05, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --- Base models ---
base_models = {
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
    "LogisticReg": LogisticRegression(max_iter=1000, random_state=42),
    "KNN(k=7)": KNeighborsClassifier(n_neighbors=7),
    "SVM(rbf)": SVC(probability=True, random_state=42),
}

print("=== Individual Model Performance (5-fold CV) ===\n")
individual_scores = {}
for name, model in base_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    individual_scores[name] = scores.mean()
    print(f"  {name:<15}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# --- Manual stacking with cross-validated predictions ---
print("\n=== Manual Stacking ===\n")

# Generate out-of-fold predictions (meta-features)
meta_train = np.zeros((len(X_train), len(base_models)))
meta_test = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    # Out-of-fold predictions for training set
    meta_train[:, i] = cross_val_predict(
        model, X_train, y_train, cv=5, method="predict"
    )
    # Full-training predictions for test set
    model.fit(X_train, y_train)
    meta_test[:, i] = model.predict(X_test)

print(f"  Meta-feature matrix shape: {meta_train.shape}")
print(f"  (each column is one base model's out-of-fold predictions)\n")

# Train meta-learner on out-of-fold predictions
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(meta_train, y_train)
stack_pred = meta_learner.predict(meta_test)
stack_acc = accuracy_score(y_test, stack_pred)

# Meta-learner coefficients show how much it trusts each base model
print("  Meta-learner coefficients (model trust weights):")
for name, coef in zip(base_models.keys(), meta_learner.coef_[0]):
    bar = "|" * int(abs(coef) * 20)
    print(f"    {name:<15}: {coef:+.3f}  {bar}")

# --- Simple majority vote baseline ---
majority_vote = (meta_test.sum(axis=1) > len(base_models) / 2).astype(int)
vote_acc = accuracy_score(y_test, majority_vote)

# --- sklearn StackingClassifier for comparison ---
sklearn_stack = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
        ("svm", SVC(probability=True, random_state=42)),
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)
sklearn_stack.fit(X_train, y_train)
sklearn_acc = accuracy_score(y_test, sklearn_stack.predict(X_test))

# --- Summary ---
print("\n=== Final Comparison ===\n")
results = {}
for name, model in base_models.items():
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"  {name:<20}: {acc:.4f}")
results["Majority Vote"] = vote_acc
print(f"  {'Majority Vote':<20}: {vote_acc:.4f}")
results["Manual Stack"] = stack_acc
print(f"  {'Manual Stack':<20}: {stack_acc:.4f}")
results["sklearn Stack"] = sklearn_acc
print(f"  {'sklearn Stack':<20}: {sklearn_acc:.4f}")

best = max(results, key=results.get)
print(f"\n  Best: {best} ({results[best]:.4f})")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Accuracy comparison
names = list(results.keys())
accs = [results[n] for n in names]
colors = ["steelblue" if "Stack" not in n else "coral" for n in names]
axes[0].barh(names, accs, color=colors)
axes[0].set_xlabel("Test Accuracy")
axes[0].set_title("Individual vs Stacked Models")
axes[0].set_xlim(min(accs) - 0.02, max(accs) + 0.02)
axes[0].invert_yaxis()

# Meta-learner coefficients
coefs = meta_learner.coef_[0]
model_names = list(base_models.keys())
colors2 = ["steelblue" if c >= 0 else "salmon" for c in coefs]
axes[1].barh(model_names, coefs, color=colors2)
axes[1].set_xlabel("Meta-learner Coefficient")
axes[1].set_title("How Much the Meta-Learner Trusts Each Model")
axes[1].axvline(0, color="k", linewidth=0.5)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("stacking.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to stacking.png")
```

---

## Key Takeaways

- **Stacking combines diverse model types via a trained meta-learner.** Instead of averaging, a second-level model learns the optimal combination.
- **Out-of-fold predictions are essential.** Using training predictions to train the meta-learner causes data leakage. Cross-validated predictions prevent this.
- **Diversity is key.** Stacking works best when base models make different kinds of errors. Combining five random forests gives little benefit; combining a forest, linear model, and k-NN gives much more.
- **The meta-learner should be simple.** Logistic regression or ridge regression work well as meta-learners. A complex meta-learner risks overfitting the base model predictions.
- **sklearn's StackingClassifier handles the cross-validation automatically.** In practice, use this rather than implementing the loop manually.
