# Handcrafted vs Learned Features

> Phase 4 — Representation Learning | Kata 4.1

---

## Concept & Intuition

### What problem are we solving?

When we build a machine learning model, the raw input data is rarely in the best form for making predictions. Traditionally, engineers would spend enormous effort designing **handcrafted features** -- manually computed transformations of the raw data that encode domain knowledge. For example, to classify whether an email is spam, you might manually count exclamation marks, compute the ratio of uppercase letters, or check for known spam phrases. Each of these is a feature you designed by hand.

The problem is that handcrafted features are limited by human imagination and domain expertise. For every new problem, you need a new expert to design new features. This approach does not scale and often misses subtle patterns that humans cannot easily articulate. What if, instead of telling the model what patterns to look for, we let it **discover** useful patterns on its own?

This is exactly what neural networks do. A hidden layer in a neural network learns its own internal representation of the input -- a set of **learned features** that are optimized end-to-end for the task at hand. The network automatically figures out which combinations of inputs are useful, without any human feature engineering.

### Why naive approaches fail

Handcrafted features fail in subtle ways. First, they require you to anticipate every relevant pattern in the data. If you miss an important feature, your model has a hard ceiling on performance. Second, handcrafted features are often **too rigid** -- they encode one specific view of the data and cannot adapt when the underlying patterns change.

A simple linear model on raw inputs also fails because many real-world problems are not linearly separable. The classic example is XOR: no single line can separate the two classes. But a neural network with a hidden layer can learn a nonlinear transformation that makes the problem separable -- effectively inventing new features that solve the problem.

### Mental models

| Analogy | Explanation |
|---------|-------------|
| **Chef vs recipe book** | Handcrafted features are like following a fixed recipe. Learned features are like a chef who tastes as they go and adjusts. |
| **Translation phrasebook vs fluency** | A phrasebook gives you fixed patterns. True fluency lets you construct any sentence you need for the situation. |
| **Manual lens vs autofocus** | You can manually focus a camera for each shot, or let autofocus find the sharpest image automatically. |

### Visual explanations

```
  HANDCRAFTED FEATURES              LEARNED FEATURES

  Raw Input                         Raw Input
  [x1, x2]                         [x1, x2]
     |                                 |
     v                                 v
  Human designs:                    Hidden layer learns:
  f1 = x1 * x2                     h1 = relu(w11*x1 + w12*x2 + b1)
  f2 = x1^2 + x2^2                 h2 = relu(w21*x1 + w22*x2 + b2)
     |                                 |
     v                                 v
  [f1, f2] --> classifier           [h1, h2] --> classifier
     |                                 |
  Fixed forever                     Weights update via backprop!


  XOR Problem -- why learned features matter:

  Input space:             After hidden layer:
  (0,1)=1    (1,1)=0      Points are rearranged so
     +----------o          a straight line CAN
     |          |          separate them!
     |          |
     o----------+          o--------+--------
  (0,0)=0    (1,0)=1           separable!
  Not linearly separable
```

---

## Hands-on Exploration

1. Build a small dataset (XOR) where handcrafted features would require knowing the exact pattern, then watch a 2-layer network learn it automatically.
2. Compare accuracy: a linear model on raw inputs vs. a neural network that learns its own hidden representation.
3. Inspect the learned hidden activations to see what internal features the network invented.

---

## Live Code

```python
import numpy as np
np.random.seed(42)

# --- XOR dataset ---
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
y = np.array([[0],[1],[1],[0]], dtype=np.float64)

# --- Approach 1: Handcrafted feature (must KNOW the pattern) ---
# XOR = (x1 + x2) == 1, so we manually engineer: f = |x1 - x2|
handcrafted = np.abs(X[:,0:1] - X[:,1:2])
print("=== Handcrafted Feature: |x1 - x2| ===")
for i in range(4):
    pred = 1.0 if handcrafted[i,0] > 0.5 else 0.0
    print(f"  Input {X[i]} -> feature={handcrafted[i,0]:.1f} -> pred={pred:.0f} (true={y[i,0]:.0f})")

# --- Approach 2: Neural network learns its own features ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 2-layer network: 2 inputs -> 4 hidden (ReLU) -> 1 output (sigmoid)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))
lr = 1.0

print("\n=== Training 2-layer network (learns its own features) ===")
for epoch in range(3001):
    # Forward
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)            # ReLU hidden layer
    z2 = h @ W2 + b2
    out = sigmoid(z2)

    # Loss (binary cross-entropy)
    loss = -np.mean(y * np.log(out + 1e-8) + (1 - y) * np.log(1 - out + 1e-8))

    # Backward
    dz2 = out - y
    dW2 = h.T @ dz2 / 4
    db2 = dz2.mean(axis=0, keepdims=True)
    dh = dz2 @ W2.T
    dz1 = dh * (z1 > 0).astype(float)
    dW1 = X.T @ dz1 / 4
    db1 = dz1.mean(axis=0, keepdims=True)

    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 1000 == 0:
        preds = (out > 0.5).astype(int).flatten()
        acc = np.mean(preds == y.flatten()) * 100
        print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.0f}%")

# --- Inspect the learned hidden features ---
print("\n=== Learned Hidden Representations ===")
z1 = X @ W1 + b1
h = np.maximum(0, z1)
print(f"  {'Input':<10} {'Hidden activations':<36} {'Output'}")
for i in range(4):
    h_str = " ".join(f"{v:5.2f}" for v in h[i])
    pred = sigmoid(h[i:i+1] @ W2 + b2)[0,0]
    print(f"  {str(X[i]):<10} [{h_str}]  ->  {pred:.3f} (true={y[i,0]:.0f})")

print("\nThe network invented hidden features that make XOR linearly separable!")
```

---

## Key Takeaways

- **Handcrafted features require domain knowledge.** You must already understand the problem well enough to manually design useful transformations -- this is fragile and does not scale.
- **Learned features are discovered automatically.** A hidden layer finds whatever internal representation best serves the task, via gradient descent.
- **Neural networks solve non-linear problems.** A single hidden layer can rearrange the input space so that previously inseparable classes become separable.
- **The hidden layer IS the learned representation.** Each neuron's activation pattern encodes a feature the network decided was useful.
- **This is the core idea of deep learning.** Stack more layers to learn increasingly abstract features, each building on the last.
