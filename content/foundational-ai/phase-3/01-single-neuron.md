# The Single Neuron

> Phase 3 — Artificial Neural Networks | Kata 3.1

---

## Concept & Intuition

### What problem are we solving?

At the heart of every neural network lies a simple computational unit: the **neuron**. A single artificial neuron takes multiple inputs, multiplies each by a learned weight, adds a bias term, and passes the result through an activation function. The formula is: `output = activation(w1*x1 + w2*x2 + ... + bias)`.

This is the building block from which all deep learning is constructed. A single neuron can learn to draw a **linear decision boundary** -- a straight line (or hyperplane) that separates two classes. This makes it capable of learning simple logical functions like AND and OR, where the positive and negative examples can be separated by a single line.

However, a single neuron has a fundamental limitation: it can only represent **linearly separable** functions. The classic example of a function it cannot learn is XOR (exclusive or), where the positive examples (0,1) and (1,0) are diagonally opposite the negative examples (0,0) and (1,1). No single straight line can separate them. This limitation was famously pointed out by Minsky and Papert in 1969 and temporarily slowed neural network research for over a decade.

### Why naive approaches fail

You might think: "Why not just use a lookup table?" For two binary inputs, a lookup table works fine. But real-world data has continuous inputs with thousands of dimensions. A lookup table would need to store every possible combination -- an impossibly large space. The neuron's power is **generalization**: it learns a compact rule (weights + bias) that works for inputs it has never seen.

You might also try a simple threshold on a single feature (e.g., "if x1 > 0.5, predict 1"). But this ignores relationships between features. The AND function requires BOTH inputs to be high -- no single-feature threshold can capture that. The neuron's weighted sum naturally combines multiple features into a single decision.

### Mental models

| Analogy | Explanation |
|---------|-------------|
| **Voting system** | Each input is a voter, each weight is how much their vote counts, the bias is the threshold needed to pass a motion, and the activation decides yes/no. |
| **Balance scale** | Inputs are objects placed on the scale, weights determine how heavy each object feels, and the bias shifts the tipping point. |
| **Light dimmer** | The weighted sum controls how much "signal" reaches the dimmer (activation), which decides how much light (output) to let through. |

### Visual explanations

```
Single Neuron Architecture:

  x1 ---w1--\
              \
  x2 ---w2--->(SUM + bias)--->[activation]---> output
              /
  x3 ---w3--/

  output = activation( w1*x1 + w2*x2 + w3*x3 + bias )


AND gate (learnable):          OR gate (learnable):

  x2                            x2
  1 | 0   1                     1 | 1   1
    |                             |
  0 | 0   0                     0 | 0   1
    +------                       +------
      0   1  x1                    0   1  x1
    Line: x1+x2 = 1.5            Line: x1+x2 = 0.5


XOR gate (NOT learnable by single neuron):

  x2
  1 | 1   0     <-- No single line can
    |               separate 0s from 1s!
  0 | 0   1
    +------
      0   1  x1
```

---

## Hands-on Exploration

1. **Implement a single neuron from scratch** using only numpy -- weighted sum, bias, and sigmoid activation. Manually set weights to solve AND and OR gates and verify with all four input combinations.

2. **Try to solve XOR** with a single neuron by experimenting with different weights and biases. Observe that no combination of weights produces the correct outputs for all four XOR inputs.

3. **Watch the neuron learn** by implementing a simple gradient descent loop. Train on AND data and watch the weights converge. Then try XOR and watch the loss plateau -- the neuron cannot find a solution.

---

## Live Code

```python
import numpy as np
np.random.seed(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def neuron(X, w, b):
    """Single neuron: weighted sum + bias + sigmoid."""
    return sigmoid(X @ w + b)

# --- Truth tables ---
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
AND_y = np.array([0, 0, 0, 1], dtype=float)
OR_y  = np.array([0, 1, 1, 1], dtype=float)
XOR_y = np.array([0, 1, 1, 0], dtype=float)

# --- Manually crafted weights ---
print("=== Hand-crafted neuron weights ===\n")
w_and, b_and = np.array([5.0, 5.0]), -7.5   # Both inputs must be high
w_or,  b_or  = np.array([5.0, 5.0]), -2.5   # Either input suffices

for name, w, b, y in [("AND", w_and, b_and, AND_y), ("OR", w_or, b_or, OR_y)]:
    preds = neuron(X, w, b)
    print(f"{name} gate (w={w}, b={b}):")
    for i in range(4):
        print(f"  {X[i]} -> {preds[i]:.4f} (target: {y[i]})")
    print()

# --- Train a neuron on AND, OR, and XOR ---
print("=== Training a neuron via gradient descent ===\n")
lr = 0.5
for name, y_true in [("AND", AND_y), ("OR", OR_y), ("XOR", XOR_y)]:
    w = np.random.randn(2) * 0.5
    b = 0.0
    for epoch in range(2000):
        # Forward pass
        y_pred = neuron(X, w, b)
        # Gradient of binary cross-entropy through sigmoid
        error = y_pred - y_true
        grad_w = X.T @ error / 4
        grad_b = np.mean(error)
        w -= lr * grad_w
        b -= lr * grad_b

    y_pred = neuron(X, w, b)
    correct = np.sum((y_pred > 0.5) == y_true)
    print(f"{name}: predictions={np.round(y_pred, 2)}, targets={y_true}, "
          f"accuracy={correct}/4")

print("\n--> A single neuron solves AND and OR but FAILS on XOR.")
```

---

## Key Takeaways

- **A neuron = weighted sum + bias + activation.** This is the atomic unit of all neural networks. Understand this deeply, and everything else is composition.
- **Linear separability is the limit.** A single neuron can only classify data that can be separated by a straight line (or hyperplane). AND and OR are linearly separable; XOR is not.
- **Weights encode importance.** Each weight tells the neuron how much to "listen" to a particular input feature. The bias shifts the decision threshold.
- **The activation function introduces nonlinearity.** Without it, the neuron is just linear regression -- stacking more linear functions still gives a linear function.
- **XOR requires more than one neuron.** This fundamental limitation motivates the move to multi-layer networks in the next kata.
