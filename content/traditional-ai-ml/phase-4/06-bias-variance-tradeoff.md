# Bias-Variance Tradeoff

> Phase 4 — Model Evaluation & Selection | Kata 4.6

---

## Concept & Intuition

### What problem are we solving?

Every prediction error can be decomposed into three components: **bias** (systematic error from wrong assumptions), **variance** (sensitivity to training data fluctuations), and **irreducible noise** (randomness in the data itself). Understanding this decomposition is perhaps the single most important concept in machine learning because it explains why models fail and what to do about it.

**High bias** means the model is too simple to capture the underlying pattern. A linear model trying to fit a curved relationship will consistently underpredict in some regions and overpredict in others, regardless of how much training data you provide. This is **underfitting**. **High variance** means the model is too complex and memorizes the training data, including its noise. A deep decision tree will produce wildly different predictions if you change the training set slightly. This is **overfitting**.

The tradeoff arises because reducing one often increases the other. Making a model more complex (adding features, increasing depth, lowering regularization) reduces bias but increases variance. Making it simpler does the opposite. The sweet spot -- minimal total error -- lies somewhere in between. **Learning curves** are the primary diagnostic tool for identifying where your model sits on this spectrum.

### Why naive approaches fail

Without understanding the bias-variance tradeoff, practitioners often chase the wrong solution. If your model underfits (high bias), adding more training data will not help -- the model cannot capture the pattern no matter how much data it sees. You need a more complex model. If your model overfits (high variance), making it more complex only makes things worse. You need more data, regularization, or a simpler model.

Training error alone is useless for diagnosis. A model with zero training error could have either perfect generalization (unlikely) or severe overfitting (likely). You must compare training error with validation/test error. When training error is low but test error is high, you have high variance. When both are high, you have high bias.

### Mental models

- **Darts analogy**: Bias is how far the average dart is from the bullseye. Variance is how spread out the darts are. Low bias + low variance = tight cluster on the bullseye. High bias = cluster off-center. High variance = darts everywhere.
- **Model complexity as a dial**: Turn it up = lower bias, higher variance. Turn it down = higher bias, lower variance. The optimal setting minimizes total error.
- **Learning curves as a diagnostic**: Plot training and validation error vs training set size. If they converge at a high error, the model is too simple (bias). If there is a persistent gap, the model is too complex (variance).

### Visual explanations

```
Bias-Variance Decomposition:

  Total Error = Bias^2 + Variance + Noise
                |          |          |
                |          |          +-- irreducible (data is noisy)
                |          +-- model is sensitive to training data
                +-- model makes systematic errors


  Error
    |                        Total Error
    |  \                   /
    |    \  Bias^2       /  Variance
    |      \           /
    |        \       /
    |          \   /
    |           \/  <-- Sweet spot (minimum total error)
    +----------+-------> Model Complexity
      simple               complex


Learning Curves (diagnostic):

  High Bias (underfitting):       High Variance (overfitting):

  Error                           Error
    |  -------- train               |
    |  -------- valid               |  -------- valid
    |                               |
    |  (both high, small gap)       |          -------- train
    +----------> train size         |  (big gap)
    Fix: more complex model         +----------> train size
                                    Fix: more data, regularization
```

---

## Hands-on Exploration

1. Fit polynomial regression with degrees 1, 3, 5, 10, and 20 to a noisy sine wave. Plot each fit on top of the true function. Watch the progression from underfitting (degree 1) to good fit (degree 3-5) to wild overfitting (degree 20).
2. Generate 20 different training sets from the same distribution. Fit a simple model and a complex model to each. Plot all 20 predictions on the same axes. The simple model's predictions cluster together (low variance) but are systematically off (high bias). The complex model's predictions scatter wildly (high variance).
3. Plot learning curves using `sklearn.model_selection.learning_curve`. Identify whether your model has a bias or variance problem by examining the gap between training and validation curves.
4. Apply regularization (L2 penalty) to a high-variance model. Plot how increasing regularization strength moves the model from high variance toward high bias, and find the optimal middle ground.

---

## Live Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# Part 1: Bias-Variance on Polynomial Regression
# ==========================================
np.random.seed(42)

# True function + noise
def true_function(x):
    return np.sin(2 * x)

n_samples = 50
X = np.sort(np.random.uniform(0, 2 * np.pi, n_samples))
y = true_function(X) + np.random.normal(0, 0.3, n_samples)
X_plot = np.linspace(0, 2 * np.pi, 200)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
degrees = [1, 3, 10, 20]

for ax, degree in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-6))
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X_plot.reshape(-1, 1))
    y_train_pred = model.predict(X.reshape(-1, 1))

    train_mse = np.mean((y - y_train_pred) ** 2)

    ax.scatter(X, y, c='steelblue', s=20, alpha=0.6, label='Data')
    ax.plot(X_plot, true_function(X_plot), 'g--', linewidth=2, label='True function')
    ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
    ax.set_title(f'Degree={degree}\nTrain MSE={train_mse:.3f}')
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Underfitting -> Good Fit -> Overfitting', fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# Part 2: Variance Visualization (multiple training sets)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, degree, label in zip(axes, [1, 15], ['Low Variance (High Bias)', 'High Variance (Low Bias)']):
    for i in range(20):
        X_sample = np.sort(np.random.uniform(0, 2 * np.pi, 30))
        y_sample = true_function(X_sample) + np.random.normal(0, 0.3, 30)
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-6))
        model.fit(X_sample.reshape(-1, 1), y_sample)
        y_pred = model.predict(X_plot.reshape(-1, 1))
        ax.plot(X_plot, y_pred, 'r-', alpha=0.15, linewidth=1)

    ax.plot(X_plot, true_function(X_plot), 'g--', linewidth=2, label='True function')
    ax.set_title(f'Degree={degree}: {label}')
    ax.set_ylim(-3, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('20 Models from Different Training Sets', fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# Part 3: Learning Curves for Diagnosis
# ==========================================
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_redundant=5,
    n_informative=10, flip_y=0.1, random_state=42
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_diag = [
    ('High Bias\n(max_depth=1)', DecisionTreeClassifier(max_depth=1, random_state=42)),
    ('Good Fit\n(max_depth=5)', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('High Variance\n(max_depth=None)', DecisionTreeClassifier(max_depth=None, random_state=42)),
]

for ax, (title, model) in zip(axes, models_diag):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_clf, y_clf, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    ax.plot(train_sizes, train_mean, 'b-o', markersize=4, linewidth=2, label='Training')
    ax.plot(train_sizes, val_mean, 'r-o', markersize=4, linewidth=2, label='Validation')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    gap = train_mean[-1] - val_mean[-1]
    ax.text(train_sizes[-1] * 0.5, 0.55, f'Gap: {gap:.3f}', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Learning Curves: Diagnosing Bias vs Variance', fontsize=14)
plt.tight_layout()
plt.show()

# --- Summary ---
print("=== Bias-Variance Diagnostic Guide ===")
print("  High Bias (underfitting):")
print("    - Both train and val error are high")
print("    - Fix: more complex model, more features, less regularization")
print("  High Variance (overfitting):")
print("    - Train error low, val error high (big gap)")
print("    - Fix: more data, regularization, simpler model, dropout")
print("  Good Fit:")
print("    - Train error reasonable, small gap to val error")
print("    - The sweet spot we're aiming for!")
```

---

## Key Takeaways

- **Total error = bias^2 + variance + irreducible noise.** You can reduce bias and variance but never the noise. The goal is to minimize the sum of bias^2 and variance.
- **High bias = underfitting; high variance = overfitting.** These are the two fundamental failure modes of machine learning models.
- **Learning curves are the primary diagnostic tool.** They show whether adding more data or changing model complexity will help, saving you from wasting time on the wrong fix.
- **Regularization trades variance for bias.** L1/L2 penalties, dropout, early stopping, and pruning all reduce variance at the cost of some bias.
- **More data reduces variance but not bias.** If your model is too simple, no amount of data will fix it. If your model is too complex, more data is the best remedy.
