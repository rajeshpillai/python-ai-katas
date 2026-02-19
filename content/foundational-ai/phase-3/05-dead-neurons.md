# Dead Neurons

> Phase 3 — First Neural Networks | Kata 3.5

---

## Concept & Intuition

### What problem are we solving?

ReLU solved the vanishing gradient problem (Kata 3.4), but it introduced a new one: **dead neurons**. ReLU outputs zero for any negative input, and its derivative is also zero for negative inputs. If a neuron's weighted sum becomes negative for every example in the dataset, that neuron outputs zero everywhere, receives zero gradient, and can never update its weights. It is permanently dead -- consuming memory and computation while contributing nothing.

Dead neurons typically arise in two scenarios. First, **large negative biases**: if a neuron's bias drifts very negative during training, the weighted sum `w*x + b` is negative for all inputs, and the neuron dies. Second, **high learning rates**: a large gradient update can push weights so far that the neuron's output becomes negative for the entire dataset in a single step. Once dead, nothing can revive it because the gradient is exactly zero.

The fix is **Leaky ReLU**: instead of outputting zero for negative inputs, it outputs a small fraction of the input (typically 0.01*x). This means the gradient is never exactly zero -- it's 0.01 instead of 0. Dead neurons become merely "drowsy" -- they still receive a trickle of gradient and can potentially recover.

### Why naive approaches fail

Dead neurons are silent failures. The network's loss may plateau, and you might blame the learning rate, the data, or the architecture. But the real problem is that a fraction of your neurons have permanently stopped contributing. A 100-neuron layer with 30 dead neurons is effectively a 70-neuron layer. Without monitoring neuron activity, you won't know why your network underperforms -- it just seems like it stopped learning.

### Mental models

- **Circuit breaker tripped**: a ReLU neuron that goes negative is like a tripped circuit breaker. Zero current flows (zero output), and the breaker can't reset itself because it needs current (gradient) to flip back. Leaky ReLU is like a breaker that always lets a trickle through.
- **Employee who stopped showing up**: a dead neuron is an employee who checked out. They sit at their desk (use parameters) but produce nothing. With ReLU, they can't even receive feedback (zero gradient). With Leaky ReLU, they at least get performance reviews (small gradient) and might re-engage.
- **One-way door**: crossing zero with ReLU is a one-way door. Once the neuron's output is negative for all inputs, the zero gradient locks the door forever. Leaky ReLU keeps the door slightly ajar.

### Visual explanations

```
ReLU vs Leaky ReLU:

  output                      output
    |        /                  |        /
    |       /                   |       /
    |      /  ReLU              |      /  Leaky ReLU
    |     /                     |     /
    |    /                      |    /
    |___/_______ input          |   /________ input
    0                          /|  0
   (derivative = 0            / | (derivative = 0.01
    for all x < 0)           /  |  for x < 0: still alive!)
                           small slope

Dead neuron scenario:
  Input data range: [x_min ... x_max]
  Neuron computes:  w*x + b

  If b = -100 and w is small:
    w*x + b < 0 for ALL x in dataset
    -> ReLU output = 0 for ALL inputs
    -> gradient = 0 for ALL inputs
    -> weights never update
    -> PERMANENTLY DEAD
```

---

## Hands-on Exploration

1. Create a simple network and show how large negative biases kill neurons
2. Demonstrate that dead ReLU neurons have zero gradient and cannot recover
3. Show that Leaky ReLU neurons receive nonzero gradient and can recover

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Activation functions ---
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

# --- Simulate a layer with 8 neurons ---
n_neurons = 8
n_inputs = 4
n_samples = 20

X = np.random.randn(n_samples, n_inputs)  # random input data

# Weights: normal, but some biases are very negative (will die)
W = np.random.randn(n_inputs, n_neurons) * 0.5
b = np.array([0.5, -0.2, -8.0, 0.1, -10.0, 0.3, -15.0, -0.1])

print("=== NEURON ACTIVITY WITH RELU ===")
print(f"Biases: {b}\n")

pre_act = X @ W + b          # (n_samples, n_neurons)
output = relu(pre_act)
grads = relu_deriv(pre_act)

# Check which neurons are alive
alive_pct = (output > 0).mean(axis=0) * 100
print(f"{'Neuron':>8} {'Bias':>6} {'Active%':>9} {'Avg Grad':>10} {'Status':<10}")
print("=" * 50)
for i in range(n_neurons):
    avg_grad = grads[:, i].mean()
    status = "DEAD" if alive_pct[i] == 0 else "alive"
    bar = "#" * int(alive_pct[i] / 5)
    print(f"{i:>8} {b[i]:>+6.1f} {alive_pct[i]:>8.0f}% "
          f"{avg_grad:>10.4f} {status:<6} {bar}")

dead_count = (alive_pct == 0).sum()
print(f"\nDead neurons: {dead_count}/{n_neurons} "
      f"({dead_count/n_neurons*100:.0f}% of capacity wasted!)")

# --- Same neurons with Leaky ReLU ---
print(f"\n=== SAME NEURONS WITH LEAKY RELU (alpha=0.01) ===")
l_output = leaky_relu(pre_act)
l_grads = leaky_relu_deriv(pre_act)

print(f"{'Neuron':>8} {'Bias':>6} {'Avg Output':>12} {'Avg Grad':>10} {'Status':<10}")
print("=" * 50)
for i in range(n_neurons):
    avg_out = l_output[:, i].mean()
    avg_grad = l_grads[:, i].mean()
    status = "recovering" if alive_pct[i] == 0 else "alive"
    print(f"{i:>8} {b[i]:>+6.1f} {avg_out:>12.4f} "
          f"{avg_grad:>10.4f} {status:<10}")

print(f"\nNotice: previously dead neurons now have nonzero gradients!")

# --- High learning rate causing neuron death ---
print(f"\n=== HIGH LEARNING RATE KILLS NEURONS ===")
W2 = np.random.randn(n_inputs, n_neurons) * 0.5
b2 = np.zeros(n_neurons)

print(f"Starting: all biases = 0 (all neurons alive)")

for lr_name, lr in [("small (0.01)", 0.01), ("large (5.0)", 5.0)]:
    b_test = b2.copy()
    # Simulate one big gradient update
    grad_b = np.random.randn(n_neurons) * 2  # random gradient signal
    b_test -= lr * grad_b

    pre = X @ W2 + b_test
    out = relu(pre)
    alive = (out > 0).any(axis=0).sum()
    print(f"  LR {lr_name}: biases after update = "
          f"[{', '.join(f'{v:+.1f}' for v in b_test)}]")
    print(f"  {'':20} alive neurons: {alive}/{n_neurons}\n")

print("Large LR -> extreme bias updates -> neurons pushed negative -> DEAD")
print("Leaky ReLU or careful LR scheduling prevents permanent death.")
```

---

## Key Takeaways

- **Dead neurons output zero for all inputs.** Their gradient is exactly zero, so they can never recover. They are permanently wasted capacity.
- **Large negative biases cause death.** If `w*x + b < 0` for every input in the dataset, the ReLU neuron is dead.
- **High learning rates cause death.** A single large update can push biases so negative that neurons die in one step.
- **Leaky ReLU prevents permanent death.** By allowing a small gradient (0.01) for negative inputs, neurons can always receive a signal and potentially recover.
- **Monitor neuron activity during training.** If a large percentage of neurons are outputting zero on all inputs, your effective network is smaller than you think.
