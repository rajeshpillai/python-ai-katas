# Logistic Regression

> Phase 3 — Supervised Learning: Classification | Kata 3.2

---

## Concept & Intuition

### What problem are we solving?

Linear regression predicts continuous values, but what if we need to predict a **probability** -- is this email spam or not? Will this patient develop diabetes? Logistic regression solves binary classification by wrapping a linear model inside the **sigmoid function**, squashing any real number into the range (0, 1). The output is not just a class label but a calibrated probability.

Despite its name, logistic regression is a **classification** algorithm, not a regression one. It finds a linear decision boundary in feature space -- a line in 2D, a plane in 3D, a hyperplane in higher dimensions -- that best separates the two classes. Points on one side get classified as 0, the other side as 1. The sigmoid function provides a smooth transition between the two, giving us a principled way to interpret "how confident" the model is.

Logistic regression is one of the most important algorithms in machine learning. It is fast, interpretable, well-calibrated probabilistically, and serves as the building block for neural networks (each neuron is essentially a logistic regression unit). Understanding it deeply pays dividends across all of ML.

### Why naive approaches fail

You might think: "Why not just use linear regression and threshold at 0.5?" The problem is that linear regression outputs can be any real number -- predicting -3.2 or 7.8 for a probability makes no sense. Worse, linear regression minimizes squared error, which is not the right loss function for classification. It gets distorted by points far from the boundary, pulling the line toward outliers in a way that hurts classification accuracy.

Using a hard threshold without probabilities also discards valuable information. In medical diagnosis, there is a huge difference between "60% likely cancer" and "99% likely cancer," but a hard classifier treats both as "positive." Logistic regression preserves this nuance through its probabilistic output.

### Mental models

- **The sigmoid as a dimmer switch**: Linear regression is a light switch (on/off at a sharp threshold). Sigmoid is a dimmer -- it smoothly transitions from off (0) to on (1), with the steepness controlled by the model's confidence.
- **Decision boundary as a fence**: Logistic regression builds a straight fence through your data. Everything on one side is class 0, the other is class 1. The fence's position and angle are learned from training data.
- **Log-odds as the hidden linear model**: Logistic regression is actually linear in log-odds space. The sigmoid just converts log-odds to probabilities.

### Visual explanations

```
The Sigmoid Function:

  1.0  |                    --------
       |                 /
       |               /
  0.5  |    - - - - -X- - - - - - -    (decision boundary at 0.5)
       |           /
       |        /
  0.0  |  ------
       +----+----+----+----+----+----> z (linear combination)
           -4   -2    0    2    4

  sigma(z) = 1 / (1 + e^(-z))

Decision Boundary in 2D:

  Feature 2
       |    o o o
       |  o o o /
       |  o o / x x
       |  o / x x x
       |  / x x x x
       +-------------> Feature 1

  / = decision boundary (w1*x1 + w2*x2 + b = 0)
  o = class 0,  x = class 1
```

---

## Hands-on Exploration

1. Plot the sigmoid function for z in [-10, 10]. Then multiply z by 0.5, 1, 2, and 5 to see how the steepness changes. Relate this to model confidence.
2. Train logistic regression on a 2D dataset. Extract the coefficients and intercept, then manually compute the decision boundary line equation. Plot it on top of the scatter plot to verify.
3. Use `model.predict_proba()` to get probability outputs instead of hard labels. Plot a histogram of predicted probabilities for each class. A well-calibrated model should show two separated peaks.
4. Add polynomial features (`PolynomialFeatures`) to logistic regression to create non-linear decision boundaries. Compare the linear vs polynomial boundaries visually.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# --- Generate dataset ---
X, y = make_classification(
    n_samples=300, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train logistic regression ---
model = LogisticRegression(random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Log Loss: {log_loss(y_test, y_prob):.3f}")
print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]:.3f}")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Sigmoid function
ax = axes[0]
z = np.linspace(-7, 7, 200)
sigmoid = 1 / (1 + np.exp(-z))
ax.plot(z, sigmoid, 'b-', linewidth=2, label='sigmoid(z)')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('z (linear combination)')
ax.set_ylabel('P(y=1)')
ax.set_title('The Sigmoid Function')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Decision boundary
ax = axes[1]
h = 0.05
x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=30)

# Draw the explicit decision boundary line
w = model.coef_[0]
b = model.intercept_[0]
x_boundary = np.linspace(x_min, x_max, 100)
y_boundary = -(w[0] * x_boundary + b) / w[1]
ax.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision Boundary')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title('Decision Boundary')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

# 3. Probability distribution
ax = axes[2]
prob_class1 = y_prob[:, 1]
ax.hist(prob_class1[y_test == 0], bins=20, alpha=0.6, label='Actual Class 0', color='blue')
ax.hist(prob_class1[y_test == 1], bins=20, alpha=0.6, label='Actual Class 1', color='red')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax.set_xlabel('Predicted P(class=1)')
ax.set_ylabel('Count')
ax.set_title('Predicted Probability Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

- **Logistic regression outputs probabilities, not just labels.** The sigmoid function maps any linear combination to (0, 1), giving calibrated confidence scores.
- **The decision boundary is linear.** Logistic regression can only draw straight lines (or hyperplanes) to separate classes. For non-linear boundaries, you need feature engineering or other algorithms.
- **Coefficients are directly interpretable.** Each coefficient tells you the change in log-odds per unit change in that feature -- positive means it pushes toward class 1.
- **Log loss is the right metric.** Cross-entropy (log loss) penalizes confident wrong predictions heavily, unlike accuracy which treats all errors equally.
- **It is the foundation of neural networks.** Each neuron in a neural network is essentially a logistic regression unit. Mastering this algorithm unlocks deeper understanding of deep learning.
