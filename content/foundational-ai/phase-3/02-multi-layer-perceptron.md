# Multi-Layer Perceptron

> Phase 3 — Artificial Neural Networks | Kata 3.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 3.1, we saw that a single neuron fails on XOR because XOR is not linearly separable. The solution is to **stack neurons into layers**. A Multi-Layer Perceptron (MLP) has an input layer, one or more **hidden layers**, and an output layer. Each neuron in a layer connects to every neuron in the next layer -- this is called a "fully connected" or "dense" architecture.

The hidden layer performs a critical function: it **transforms the input space** into a new representation where the problem becomes linearly separable. For XOR, two hidden neurons can learn to create an intermediate representation where the four input points are rearranged so that a single output neuron can draw a line between the classes. This is the fundamental insight of deep learning -- each layer transforms data into increasingly useful representations.

Training an MLP requires **backpropagation**: computing how much each weight contributed to the error, then adjusting all weights simultaneously using gradient descent. The chain rule of calculus lets us propagate error gradients backward from the output layer through each hidden layer. This algorithm, combined with the universal approximation theorem (which proves that a single hidden layer with enough neurons can approximate any continuous function), is what makes neural networks so powerful.

### Why naive approaches fail

Without hidden layers, we are stuck with linear decision boundaries. You could try feature engineering -- manually creating new features like `x1 * x2` to make XOR linearly separable. But this requires human insight for each new problem. Hidden layers **automate feature engineering**: they learn whatever intermediate representations are needed, without human guidance.

A common misconception is that adding more neurons to a single layer solves everything. While a wide single layer has theoretical universal approximation power, in practice **depth** (more layers) is far more efficient than width. A deep network can represent certain functions exponentially more compactly than a shallow one. This is why modern architectures go deep rather than just wide.

### Mental models

| Analogy | Explanation |
|---------|-------------|
| **Assembly line** | Each layer is a station that transforms the raw material a bit more. Raw inputs enter, refined features emerge, and the final station makes the decision. |
| **Translation chain** | Layer 1 translates the problem from "hard" coordinates to "easier" coordinates. Layer 2 solves the now-easy problem. Like translating a foreign document through an intermediate language. |
| **Committee of experts** | Hidden neurons are specialists. One might detect "both inputs high", another detects "both inputs low". The output neuron combines their opinions for the final answer. |

### Visual explanations

```
2-Layer MLP for XOR:

  Input      Hidden        Output
  Layer      Layer         Layer

  x1 ---w1-->[h1]--v1--\
      \  /               \
       \/                 >[out]---> prediction
       /\                 /
      /  \               /
  x2 ---w2-->[h2]--v2--/

  4 weights (w) from input to hidden
  2 weights (v) from hidden to output
  + 3 biases (one per neuron)


How hidden layer transforms XOR:

  Original space:           Hidden space (after layer 1):
  x2                        h2
  1 | 1   0                 1 | (0,0)     (1,0)
    |                         |        (0,1)
  0 | 0   1                 0 |     (1,1)
    +------                   +-----------
      0   1  x1                0         1  h1

  NOT separable!            NOW separable with one line!


Backpropagation flow:

  FORWARD:   x --> hidden --> output --> loss
  BACKWARD:  x <-- hidden <-- output <-- dL/d(output)
                (chain rule propagates gradients back)
```

---

## Hands-on Exploration

1. **Build a 2-layer MLP from scratch** with numpy. Use 2 hidden neurons with sigmoid activation. Initialize random weights and run a forward pass on all four XOR inputs to see random predictions.

2. **Implement backpropagation manually.** Compute the gradient of the loss with respect to every weight by applying the chain rule layer by layer. Update all weights with gradient descent.

3. **Train until XOR is solved.** Run the training loop for enough epochs and watch the loss decrease. Verify that all four XOR inputs produce the correct outputs. Print the learned hidden representations to see how the network "re-maps" the inputs.

---

## Live Code

```python
import numpy as np
np.random.seed(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    """Derivative of sigmoid given its output a."""
    return a * (1 - a)

# --- XOR dataset ---
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# --- Network: 2 inputs -> 2 hidden -> 1 output ---
W1 = np.random.randn(2, 2) * 0.5   # input-to-hidden weights
b1 = np.zeros((1, 2))               # hidden biases
W2 = np.random.randn(2, 1) * 0.5   # hidden-to-output weights
b2 = np.zeros((1, 1))               # output bias
lr = 2.0

print("=== Training MLP on XOR ===\n")
for epoch in range(5000):
    # Forward pass
    z1 = X @ W1 + b1
    h  = sigmoid(z1)          # hidden activations
    z2 = h @ W2 + b2
    out = sigmoid(z2)         # output predictions

    # Loss (MSE)
    loss = np.mean((out - y) ** 2)

    # Backward pass (manual backprop)
    d_out = (out - y) * sigmoid_deriv(out)       # (4,1)
    dW2 = h.T @ d_out / 4                        # (2,1)
    db2 = np.mean(d_out, axis=0, keepdims=True)

    d_hidden = d_out @ W2.T * sigmoid_deriv(h)   # (4,2)
    dW1 = X.T @ d_hidden / 4                     # (2,2)
    db1 = np.mean(d_hidden, axis=0, keepdims=True)

    # Update weights
    W2 -= lr * dW2;  b2 -= lr * db2
    W1 -= lr * dW1;  b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# Final results
print(f"\n=== Final predictions (after {epoch+1} epochs) ===\n")
for i in range(4):
    print(f"  Input: {X[i]}  Hidden: [{h[i,0]:.3f}, {h[i,1]:.3f}]  "
          f"Output: {out[i,0]:.4f}  Target: {y[i,0]:.0f}")

correct = np.sum((out > 0.5).astype(float) == y)
print(f"\nAccuracy: {int(correct)}/4")
print("\n--> The hidden layer re-maps inputs so XOR becomes separable!")
```

---

## Key Takeaways

- **Hidden layers transform the input space.** They learn intermediate representations where the original problem becomes linearly separable. This is the core mechanism of all deep learning.
- **Backpropagation = chain rule applied layer by layer.** Gradients flow backward from the loss through each layer, telling every weight how to adjust.
- **Two hidden neurons suffice for XOR.** The minimal MLP has 2 inputs, 2 hidden neurons, and 1 output -- a total of 9 learnable parameters (6 weights + 3 biases).
- **Depth enables compositional features.** Each layer builds on the previous one's representations, allowing networks to learn hierarchical structure in data.
- **Universal approximation is real but practical.** A single hidden layer can theoretically approximate any function, but training deeper networks is often more efficient in practice.
