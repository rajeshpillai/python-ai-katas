# Vanishing Gradients

> Phase 3 — First Neural Networks | Kata 3.4

---

## Concept & Intuition

### What problem are we solving?

When you stack many layers in a neural network, gradients must flow backward through every layer via the chain rule. If each layer's activation function has a derivative less than 1 (as sigmoid does -- its maximum derivative is 0.25), the gradients get multiplied together and **shrink exponentially**. After 10 layers, a gradient might be multiplied by 0.25 ten times: 0.25^10 = 0.00000095. The gradient effectively reaches zero, and the early layers stop learning entirely.

This is the **vanishing gradient problem**, and it was the primary reason deep networks failed to train for decades. The backpropagation algorithm was known since the 1980s, but networks deeper than 2-3 layers simply would not learn. The gradients reaching the first layers were so tiny that parameter updates were negligible -- those layers stayed at their random initialization forever.

The fix turned out to be surprisingly simple: **use ReLU instead of sigmoid.** ReLU's derivative is either 0 or 1. When it's 1, the gradient passes through unchanged. Multiplying by 1 ten times gives 1 -- no shrinkage. This single change enabled deep networks (10, 50, 100+ layers) and launched the deep learning revolution.

### Why naive approaches fail

If you build a deep network with sigmoid activations and it fails to learn, the symptoms are confusing. The loss barely decreases. The outputs look random. You might blame the learning rate, the initialization, or the data. But the real problem is invisible without inspecting layer-by-layer gradients: the first layers have gradients that are essentially zero. No learning rate can fix a gradient of 0.000001 -- it would need to be absurdly large, destabilizing the later layers.

### Mental models

- **Telephone game**: a message passed through 10 people gets quieter at each step. By the end, it's inaudible. Sigmoid is like each person whispering (multiplying by 0.25). ReLU is like each person speaking at normal volume (multiplying by 1).
- **Water through pipes**: each sigmoid layer is a leaky pipe that loses 75% of the water (gradient). After 10 pipes, almost nothing flows through. ReLU pipes don't leak (when active).
- **Compound decay**: 0.25^n decays exponentially. At n=5, it's 0.001. At n=10, it's 0.000001. This is the same math as radioactive decay -- the gradient has a "half-life" of about 1 layer.

### Visual explanations

```
Gradient flow through 10 layers:

SIGMOID: gradient shrinks at each layer
Layer 10: |############################| 1.000
Layer  9: |#######                     | 0.250
Layer  8: |##                          | 0.062
Layer  7: |#                           | 0.016
Layer  6: |                            | 0.004
Layer  5: |                            | 0.001
Layer  4: |                            | 0.000
Layer  3: |                            | 0.000
Layer  1: |                            | 0.000  <-- DEAD (no learning)

ReLU: gradient passes through (when active)
Layer 10: |############################| 1.000
Layer  9: |############################| 1.000
Layer  8: |############################| 1.000
Layer  7: |############################| 1.000
Layer  6: |############################| 1.000
Layer  1: |############################| 1.000  <-- ALIVE (still learning!)
```

---

## Hands-on Exploration

1. Simulate forward and backward passes through 10 sigmoid layers
2. Watch the gradient magnitude shrink to near-zero at early layers
3. Repeat with ReLU and compare gradient flow

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Activation functions and their derivatives ---
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# --- Simulate gradient flow through N layers ---
def gradient_flow(n_layers, activation_deriv, input_val=0.5):
    """Track gradient magnitude through N layers using chain rule."""
    grad = 1.0  # start with gradient = 1 at the output
    grads = [grad]
    pre_activations = []

    # Simulate pre-activation values at each layer
    x = input_val
    for i in range(n_layers):
        pre_act = np.random.randn() * 0.5  # typical pre-activation
        pre_activations.append(pre_act)

    # Backward: multiply by derivative at each layer
    for i in range(n_layers - 1, -1, -1):
        local_grad = activation_deriv(np.array([pre_activations[i]]))[0]
        grad *= local_grad
        grads.append(abs(grad))

    return grads  # from output layer back to input

# --- Compare sigmoid vs ReLU ---
n_layers = 10
sig_grads = gradient_flow(n_layers, sigmoid_deriv)
relu_grads = gradient_flow(n_layers, relu_deriv)

print("=== GRADIENT MAGNITUDE THROUGH 10 LAYERS ===\n")
print(f"{'Layer':<8} {'Sigmoid Grad':>14} {'ReLU Grad':>14}")
print("=" * 40)

for i in range(n_layers + 1):
    layer_num = n_layers - i
    sg = sig_grads[i]
    rg = relu_grads[i]
    sig_bar = "#" * max(1, int(sg * 30)) if sg > 0.001 else "~"
    rel_bar = "#" * max(1, int(rg * 30)) if rg > 0.001 else "~"
    print(f"  {layer_num:<5} {sg:>14.8f}  {sig_bar}")
    print(f"  {'':5} {rg:>14.8f}  {rel_bar}")
    print()

# --- Theoretical analysis ---
print("=== THEORETICAL WORST CASE ===")
print("Sigmoid max derivative = 0.25 (at x=0)")
print(f"After N layers, gradient <= 0.25^N:\n")
for n in [1, 2, 5, 8, 10, 15, 20]:
    val = 0.25 ** n
    bar = "#" * max(1, int(val * 50)) if val > 0.0001 else "~"
    print(f"  {n:>2} layers: 0.25^{n:<2} = {val:.12f}  {bar}")

# --- Practical impact: learning rate perspective ---
print(f"\n=== WHAT THIS MEANS FOR LEARNING ===")
print(f"If learning rate = 0.01:")
for n in [1, 5, 10]:
    effective_lr = 0.01 * (0.25 ** n)
    print(f"  Layer {n:>2} from output: "
          f"effective LR = {effective_lr:.2e}"
          f"{'  (basically zero!)' if effective_lr < 1e-6 else ''}")

print(f"\nConclusion: sigmoid kills gradients in deep networks.")
print(f"ReLU (derivative = 0 or 1) does not compound this decay.")
print(f"This is why ReLU enabled the deep learning revolution.")
```

---

## Key Takeaways

- **Vanishing gradients happen when activation derivatives are less than 1.** The chain rule multiplies these small values across layers, causing exponential decay.
- **Sigmoid's maximum derivative is 0.25.** After 10 layers: 0.25^10 = ~0.000001. Early layers receive effectively zero gradient.
- **ReLU's derivative is 0 or 1.** When active, gradients pass through unchanged -- no exponential decay.
- **This was the historical bottleneck of deep learning.** Networks deeper than 2-3 layers couldn't train until ReLU (and related fixes) appeared.
- **Always inspect gradient magnitudes.** If early layers have near-zero gradients, the network is not learning -- it just looks like it's training.
