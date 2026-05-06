# Visualizing Hidden Layers

> Phase 4 — Representation Learning | Kata 4.4

---

## Concept & Intuition

### What problem are we solving?

Neural networks are often called "black boxes" because the transformations happening inside them are opaque. We feed in inputs, get out predictions, and everything in between is a mystery of matrix multiplications and nonlinearities. But those hidden layers are doing something remarkable: they are learning to re-represent the data in progressively more useful coordinate systems. Understanding what happens at each layer is the key to demystifying neural networks.

In this kata, we build a small network, feed data through it, and extract the activations (outputs) at each hidden layer. By examining how the data points are arranged in each layer's activation space, we can see the network literally reshaping the input geometry -- stretching, rotating, folding, and separating the data until the classes become linearly separable at the final layer.

This is not just an academic exercise. Visualizing hidden layers is a practical debugging tool. If two classes that should be separated are still overlapping at a deep layer, you know the network needs more capacity or different architecture. If the representations collapse to a single point, you have a dying ReLU problem. Reading the internal representations tells you what the network has learned and where it is struggling.

### Why naive approaches fail

Without looking inside the network, you can only judge it by its final output -- the loss and accuracy. But these aggregate metrics hide crucial information. A network might achieve 70% accuracy in two very different ways: it might have learned excellent features but have a weak final classifier, or it might have terrible internal representations but be overfitting to spurious patterns. The loss curve alone cannot distinguish these cases.

Simply printing the weight matrices does not help either. A 64x32 weight matrix contains 2,048 numbers, and staring at them reveals almost nothing about what the layer is doing functionally. The insight comes from watching what happens to actual data points as they flow through the network -- how the geometry of the data cloud transforms layer by layer.

### Mental models

- **Assembly line.** Each hidden layer is a station on an assembly line. Raw materials (input features) enter, and each station reshapes them into something more useful. By inspecting the product at each station, you understand what each step contributes.
- **Coordinate transformation.** Each layer is a change of coordinates. The input might be in "pixel space," but hidden layer 1 re-expresses it in "edge space," layer 2 in "shape space," and so on. Visualizing activations is like reading the data in each coordinate system.
- **Folding paper.** A nonlinear hidden layer can fold the input space, bending regions that were far apart to become nearby (or vice versa). ReLU specifically "creases" the space along hyperplanes, collapsing the negative half to zero.
- **Progressive separation.** For classification, the network's job is to make classes linearly separable by the final layer. Each hidden layer contributes a step toward this goal, gradually untangling the data.

### Visual explanations

```
How a 2-layer network transforms XOR data:

Input Space (Layer 0)        Hidden Layer 1          Output Layer
   Class A: (0,0),(1,1)      After W1*x + b1         After W2*h + b2
   Class B: (0,1),(1,0)      + ReLU                  (linearly separable!)

   1 | B . . A               1 | . . . A             1 | . . . . A A
     |                         |                       |
   0 | A . . B               0 | A . .               0 | B B . . . .
     +--------->              0 +------->              0 +----------->
     0       1                  0     1                  0          1

   (NOT linearly             (data is                 (simple threshold
    separable!)               re-arranged)              separates classes)

Layer-by-layer activation flow:

  Input          Hidden 1        Hidden 2        Output
  [x1, x2]  --> [h1, h2, h3] --> [h4, h5] -->  [y]
     2D            3D               2D           1D

  Each arrow = (matrix multiply) + (bias) + (activation function)
  Each layer can change dimensionality!
```

---

## Hands-on Exploration

1. Create the XOR dataset (a classic nonlinearly separable problem) and build a small 2-hidden-layer network that solves it.
2. After training, do a forward pass and capture the activations at every layer -- input, hidden 1, hidden 2, and output.
3. Print the activations at each layer to see how the four XOR data points are repositioned in space, going from overlapping classes to clearly separated ones.

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- XOR dataset ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
labels = ["A", "B", "B", "A"]

# --- Network: 2 -> 4 -> 3 -> 1 ---
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 3) * 0.5
b2 = np.zeros((1, 3))
W3 = np.random.randn(3, 1) * 0.5
b3 = np.zeros((1, 1))

lr = 0.5
print("=== Training on XOR ===")
for epoch in range(2000):
    # Forward (save all activations)
    z1 = X @ W1 + b1;    a1 = relu(z1)
    z2 = a1 @ W2 + b2;   a2 = relu(z2)
    z3 = a2 @ W3 + b3;   a3 = sigmoid(z3)
    loss = -np.mean(y * np.log(a3 + 1e-8) + (1 - y) * np.log(1 - a3 + 1e-8))
    # Backward
    dz3 = a3 - y
    dW3 = a2.T @ dz3 / 4;  db3 = dz3.mean(axis=0, keepdims=True)
    da2 = dz3 @ W3.T;      dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2 / 4;  db2 = dz2.mean(axis=0, keepdims=True)
    da1 = dz2 @ W2.T;      dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1 / 4;   db1 = dz1.mean(axis=0, keepdims=True)
    W3 -= lr * dW3; b3 -= lr * db3
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1
    if epoch % 500 == 0:
        print(f"  Epoch {epoch:4d} | Loss: {loss:.4f}")

# --- Extract and display layer-by-layer activations ---
z1 = X @ W1 + b1;  a1 = relu(z1)
z2 = a1 @ W2 + b2; a2 = relu(z2)
z3 = a2 @ W3 + b3; a3 = sigmoid(z3)

layers = [("Input (2D)", X), ("Hidden 1 (4D)", a1),
          ("Hidden 2 (3D)", a2), ("Output (1D)", a3)]

print("\n=== Layer-by-Layer Activations ===")
for name, acts in layers:
    print(f"\n--- {name} ---")
    for i in range(4):
        vec = ", ".join(f"{v:+.3f}" for v in acts[i])
        print(f"  Point {labels[i]} ({X[i,0]:.0f},{X[i,1]:.0f}): [{vec}]")

# --- Measure class separation at each layer ---
print("\n=== Class Separation (distance between class centers) ===")
for name, acts in layers:
    class_a = acts[[0, 3]]  # XOR label 0
    class_b = acts[[1, 2]]  # XOR label 1
    center_a = class_a.mean(axis=0)
    center_b = class_b.mean(axis=0)
    dist = np.linalg.norm(center_a - center_b)
    var_a = np.mean(np.linalg.norm(class_a - center_a, axis=1))
    var_b = np.mean(np.linalg.norm(class_b - center_b, axis=1))
    ratio = dist / (var_a + var_b + 1e-8)
    bar = "#" * int(min(ratio * 5, 40))
    print(f"  {name:20s} | separation ratio: {ratio:.2f} {bar}")

print("\n=== Predictions ===")
for i in range(4):
    pred = a3[i, 0]
    correct = "OK" if (pred > 0.5) == y[i, 0] else "WRONG"
    print(f"  ({X[i,0]:.0f},{X[i,1]:.0f}) -> {pred:.4f}  (target {y[i,0]:.0f}) {correct}")
```

---

## Key Takeaways

- **Hidden layers re-represent data.** Each layer transforms the data into a new coordinate system where the relevant patterns become progressively easier to detect.
- **Nonlinearities enable folding.** Without activation functions, stacking layers would collapse to a single linear transformation. ReLU, sigmoid, and tanh let each layer bend and fold the space.
- **Class separation increases with depth.** In a well-trained network, the distance between class clusters grows at each successive layer, making final classification trivial.
- **Visualization is a debugging tool.** If activations collapse (all zeros), spread too much, or fail to separate classes, it signals architectural or training problems that loss curves alone cannot reveal.
- **Dimensionality changes at each layer.** The network can expand or compress the representation at each step, choosing whatever dimensionality best serves the task.
