# Activation Functions

> Phase 3 — Artificial Neural Networks | Kata 3.3

---

## Concept & Intuition

### What problem are we solving?

Without an activation function, a neural network is just a series of linear transformations: matrix multiply, add bias, repeat. But composing linear functions always yields another linear function -- so no matter how many layers you stack, the network can only learn linear relationships. **Activation functions inject nonlinearity**, allowing networks to approximate complex, curved decision boundaries.

The choice of activation function has a profound effect on how well a network trains. Early networks used **sigmoid** (squashes to 0-1) and **tanh** (squashes to -1 to +1), which are smooth and biologically inspired. But both suffer from a critical problem: their derivatives become very small for large inputs (**saturation**), causing gradients to shrink as they propagate backward through layers. This is the vanishing gradient problem (explored in Kata 3.4).

**ReLU** (Rectified Linear Unit) changed the game. Defined simply as `max(0, x)`, ReLU has a constant gradient of 1 for positive inputs -- gradients flow freely without shrinking. This single innovation enabled training of much deeper networks and is the default activation in most modern architectures. Variants like **Leaky ReLU** address ReLU's one weakness (dead neurons for negative inputs) by allowing a small gradient when the input is negative.

### Why naive approaches fail

Using no activation at all (the identity function) makes every layer redundant -- a 100-layer network with identity activations is mathematically equivalent to a single linear transformation. You get zero benefit from depth.

Using sigmoid everywhere seems reasonable because it outputs nice probabilities between 0 and 1. But sigmoid saturates for inputs outside roughly [-3, 3]: the derivative approaches zero, and gradients vanish exponentially through layers. A 10-layer network with sigmoid might see gradients shrink by a factor of 10,000x at the first layer, making learning impossibly slow.

### Mental models

| Activation | Analogy |
|-----------|---------|
| **Sigmoid** | A dimmer switch that smoothly goes from "fully off" (0) to "fully on" (1). Problem: once fully on/off, it barely responds to further changes. |
| **Tanh** | Same dimmer, but centered: goes from -1 to +1. Better than sigmoid because outputs are zero-centered, but still saturates. |
| **ReLU** | A one-way valve: blocks negative flow entirely (outputs 0), passes positive flow unchanged. Simple, fast, and gradients are either 0 or 1. |
| **Leaky ReLU** | A slightly leaky valve: blocks most negative flow but lets a trickle through (slope 0.01). This prevents neurons from "dying". |

### Visual explanations

```
Activation Functions (text-based plots, x from -4 to +4):

Sigmoid: f(x) = 1/(1+e^-x)              Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x)
  1.0 |            .-------               1.0 |            .-------
      |          ./                            |          ./
      |        ./                          0.0 |--------./--------
      |      ./                                |      ./
  0.0 |-----'                             -1.0 |-----'
      +-----|-----|-----                       +-----|-----|-----
           -2    0    +2                            -2    0    +2


ReLU: f(x) = max(0, x)                  Leaky ReLU: f(x) = max(0.01x, x)
      |         /                              |         /
      |        /                               |        /
      |       /                                |       /
      |      /                                 |      /
  0.0 |-----/                              0.0 |----/
      |                                        |  / (small slope=0.01)
      +-----|-----|-----                       +-----|-----|-----
           -2    0    +2                            -2    0    +2


Derivatives (critical for backpropagation):

Sigmoid':  max 0.25 at x=0, ~0 at extremes    ReLU':  1 for x>0, 0 for x<0
Tanh':     max 1.0  at x=0, ~0 at extremes    Leaky': 1 for x>0, 0.01 for x<0
```

---

## Hands-on Exploration

1. **Compute each activation and its derivative** for a range of inputs. Compare their shapes: which ones saturate? Which maintain strong gradients?

2. **Chain derivatives** to simulate gradient flow. Multiply 10 derivatives together (as if passing through 10 layers). Compare the result for sigmoid vs ReLU to see the vanishing gradient effect.

3. **Check zero-centering.** Compute the mean output of each activation for random inputs. Sigmoid outputs are always positive (mean around 0.5), which causes zig-zag gradient updates. Tanh and ReLU variants are better centered.

---

## Live Code

```python
import numpy as np
np.random.seed(42)

# --- Activation functions and derivatives ---
def sigmoid(x):     return 1 / (1 + np.exp(-x))
def sigmoid_d(x):   s = sigmoid(x); return s * (1 - s)

def tanh(x):        return np.tanh(x)
def tanh_d(x):      return 1 - np.tanh(x)**2

def relu(x):        return np.maximum(0, x)
def relu_d(x):      return (x > 0).astype(float)

def leaky_relu(x, a=0.01):   return np.where(x > 0, x, a * x)
def leaky_relu_d(x, a=0.01): return np.where(x > 0, 1.0, a)

# --- Compare activations across input range ---
x = np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4], dtype=float)
print("=== Activation outputs ===\n")
print(f"{'x':>6} | {'sigmoid':>8} {'tanh':>8} {'relu':>8} {'leaky':>8}")
print("-" * 50)
for xi in x:
    print(f"{xi:6.1f} | {sigmoid(xi):8.4f} {tanh(xi):8.4f} "
          f"{relu(xi):8.4f} {leaky_relu(xi):8.4f}")

print("\n=== Derivatives (critical for gradient flow) ===\n")
print(f"{'x':>6} | {'sig_d':>8} {'tanh_d':>8} {'relu_d':>8} {'leaky_d':>8}")
print("-" * 50)
for xi in x:
    print(f"{xi:6.1f} | {sigmoid_d(xi):8.4f} {tanh_d(xi):8.4f} "
          f"{relu_d(xi):8.4f} {leaky_relu_d(xi):8.4f}")

# --- Gradient flow through 10 layers ---
print("\n=== Gradient flow through 10 layers ===")
print("(Product of derivatives at a typical activation value)\n")
for name, deriv_fn, test_val in [
    ("Sigmoid", sigmoid_d, 0.5),
    ("Tanh",    tanh_d,    0.5),
    ("ReLU",    relu_d,    0.5),
    ("Leaky",   leaky_relu_d, 0.5),
]:
    d = deriv_fn(test_val)
    grad_10 = d ** 10
    print(f"  {name:8s}: deriv at x={test_val} is {d:.4f}, "
          f"after 10 layers: {grad_10:.8f}")

print("\n--> Sigmoid/Tanh gradients shrink exponentially through layers.")
print("--> ReLU maintains full gradient (1.0) for positive inputs.")
```

---

## Key Takeaways

- **Activations are the source of nonlinearity.** Without them, a deep network collapses to a single linear transformation, regardless of depth.
- **Sigmoid and tanh saturate.** For large positive or negative inputs, their derivatives approach zero. This causes gradients to vanish in deep networks.
- **ReLU is the modern default.** Its derivative is 1 for positive inputs, enabling gradient flow through arbitrarily deep networks. It is also computationally cheaper (just a max operation).
- **Zero-centering matters.** Sigmoid outputs are always positive, which biases gradient updates. Tanh is zero-centered and generally preferred over sigmoid for hidden layers.
- **Leaky ReLU fixes the "dead neuron" problem.** By allowing a small gradient (0.01) for negative inputs, neurons can recover even if they initially output zero for all training data.
