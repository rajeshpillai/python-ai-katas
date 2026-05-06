# Decision Trees

> Phase 3 — Supervised Learning: Classification | Kata 3.3

---

## Concept & Intuition

### What problem are we solving?

Imagine sorting your email: "Is the sender in my contacts? If yes, inbox. If no, does it contain the word 'invoice'? If yes, important. If no, spam." You just built a decision tree. Decision trees classify data by learning a series of **if-then rules** directly from the data, splitting the feature space into rectangular regions, each assigned a class label.

At each node, the tree asks a question about one feature (e.g., "Is income > 50k?") and branches based on the answer. The algorithm chooses which question to ask by measuring how much each potential split **purifies** the groups -- ideally, each split should separate the classes as cleanly as possible. The two main measures of impurity are **Gini impurity** (how often a randomly chosen element would be misclassified) and **entropy** (the information-theoretic uncertainty in the group).

Decision trees are beloved for their interpretability -- you can literally draw the decision process as a flowchart that anyone can understand. They handle both numerical and categorical features, require minimal preprocessing, and can capture non-linear relationships. However, they are prone to overfitting, which is why controlling tree depth and pruning are essential.

### Why naive approaches fail

An unrestricted decision tree will keep splitting until every leaf node contains a single training point -- achieving 100% training accuracy but catastrophically overfitting. It memorizes every quirk and noise pattern in the training data. A tree grown to full depth on a noisy dataset might have hundreds of nodes, each capturing random fluctuations rather than genuine patterns.

Without pruning or depth limits, decision trees are also unstable: small changes in the training data can produce completely different trees. This high variance makes individual trees unreliable, which is why ensemble methods like Random Forests were invented to stabilize them.

### Mental models

- **Twenty questions game**: The tree tries to classify items by asking the most informative yes/no questions first, narrowing down possibilities with each answer.
- **Cutting a pizza**: Each split cuts the feature space with a line parallel to one axis. Multiple cuts create rectangular slices, each assigned to a class.
- **Purity as a goal**: Imagine sorting colored marbles into cups. A pure cup has all one color. The tree's job is to find the questions that produce the purest cups with the fewest questions.

### Visual explanations

```
Decision Tree Structure:

         [Income > 50k?]
          /            \
        Yes             No
        /                \
  [Age > 35?]        [Student?]
   /       \          /       \
  Yes      No       Yes       No
  /         \       /           \
 BUY     MAYBE   MAYBE       NO BUY


Gini Impurity:

  Pure node:   [A A A A A]   Gini = 0.00  (perfect)
  Mixed:       [A A A B B]   Gini = 0.48  (impure)
  Worst case:  [A A B B]     Gini = 0.50  (maximum for binary)

  Gini(node) = 1 - sum(p_i^2)

Feature Space Splits:

  Feature 2
    |     |  B B B
    | A A |------
    | A A | B B B
    |-----|
    | A A | B B
    +-------------> Feature 1
    (axis-aligned rectangular regions)
```

---

## Hands-on Exploration

1. Train a decision tree with no depth limit on a noisy dataset. Print `tree.get_depth()` and `tree.get_n_leaves()`. Then restrict `max_depth=3` and compare. Plot both trees using `plot_tree()`.
2. Compare Gini impurity vs entropy as splitting criteria. Train two trees (one with `criterion='gini'`, one with `criterion='entropy'`) and see if the resulting trees differ.
3. Visualize the decision boundary of a decision tree on a 2D dataset. Notice the axis-aligned rectangular regions -- this is a fundamental limitation of single decision trees.
4. Experiment with pruning parameters: `min_samples_split`, `min_samples_leaf`, and `max_depth`. Plot training vs test accuracy as you vary `max_depth` from 1 to 20 to find the sweet spot.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# @param max_depth int 1 20 3

# --- Generate dataset ---
X, y = make_classification(
    n_samples=300, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1,
    flip_y=0.1, random_state=42  # 10% label noise
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Train decision tree with controlled depth ---
tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
tree.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree.predict(X_train))
test_acc = accuracy_score(y_test, tree.predict(X_test))

print(f"max_depth = {max_depth}")
print(f"Tree depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}")
print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test  Accuracy: {test_acc:.3f}")
print(f"Overfit gap: {train_acc - test_acc:.3f}")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 1. Decision boundary
ax = axes[0]
h = 0.05
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=30)
ax.set_title(f'Decision Boundary (depth={max_depth})')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# 2. Training vs Test accuracy across depths
ax = axes[1]
depths = range(1, 21)
train_accs, test_accs = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, t.predict(X_train)))
    test_accs.append(accuracy_score(y_test, t.predict(X_test)))

ax.plot(depths, train_accs, 'b-o', label='Train', markersize=4)
ax.plot(depths, test_accs, 'r-o', label='Test', markersize=4)
ax.axvline(x=max_depth, color='green', linestyle='--', label=f'Current depth={max_depth}')
ax.set_xlabel('Max Depth')
ax.set_ylabel('Accuracy')
ax.set_title('Overfitting vs Underfitting')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Tree structure
ax = axes[2]
plot_tree(tree, filled=True, feature_names=['Feature 1', 'Feature 2'],
          class_names=['Class 0', 'Class 1'], ax=ax, fontsize=7,
          rounded=True, proportion=True)
ax.set_title(f'Tree Structure (depth={max_depth})')

plt.tight_layout()
plt.show()

# --- Gini vs Entropy comparison ---
print("\n--- Gini vs Entropy ---")
for criterion in ['gini', 'entropy']:
    t = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    t.fit(X_train, y_train)
    acc = accuracy_score(y_test, t.predict(X_test))
    print(f"  {criterion:>7s}: Accuracy={acc:.3f}, Leaves={t.get_n_leaves()}")
```

---

## Key Takeaways

- **Decision trees learn if-then rules automatically.** They split the feature space into axis-aligned rectangular regions, making them intuitive and interpretable.
- **Unrestricted trees overfit severely.** Without `max_depth` or pruning constraints, a tree will memorize the training data, including its noise.
- **Gini and entropy measure impurity differently but usually produce similar trees.** Gini is slightly faster to compute; entropy tends to produce slightly more balanced trees.
- **max_depth is the primary overfitting control.** Plot training vs test accuracy across depths to find the sweet spot where the gap is small and test accuracy is highest.
- **Axis-aligned boundaries are a limitation.** Decision trees cannot easily capture diagonal or curved boundaries, which is why ensemble methods and other algorithms are often preferred.
