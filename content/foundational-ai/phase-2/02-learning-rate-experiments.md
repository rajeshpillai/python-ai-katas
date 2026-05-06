# Learning Rate Experiments

> Phase 2 — Optimization | Kata 2.2

---

## Concept & Intuition

### What problem are we solving?

Gradient descent tells us the *direction* to move, but not *how far* to step. The **learning rate** controls step size. Set it too small and training crawls — you waste thousands of iterations barely moving. Set it too large and you overshoot the minimum, bouncing wildly or even making the loss explode. Finding the right learning rate is one of the most important practical skills in machine learning.

This kata gives you hands-on intuition for the learning rate by running the same gradient descent with three different values and comparing the results side by side. You will see smooth convergence, sluggish crawling, and catastrophic divergence — all from changing a single number.

### Why naive approaches fail

Beginners often pick a learning rate once and never revisit it. If the loss is not decreasing, they assume the model is wrong or the data is bad. In reality, the learning rate is the most common culprit. A factor-of-10 change in learning rate can be the difference between a model that learns in 50 epochs and one that never learns at all.

### Mental models

| Learning Rate | Analogy | Behavior |
|---|---|---|
| Too small (0.0001) | Taking baby steps down a mountain | Gets there eventually, but painfully slow |
| Just right (0.01) | Confident strides downhill | Reaches the bottom efficiently |
| Too large (1.0) | Leaping wildly down the slope | Overshoots, bounces, may fly off the mountain |

### Visual explanations

```
Loss over epochs for different learning rates:

  Loss
   |
 8 | S S S                     S = too small (barely moves)
   | S                         R = just right
 6 | S   L                     L = too large (bouncing)
   | S R
 4 | S R   L
   |   R         L
 2 | S   R             L       L       L
   |       R R R R R R R R     <-- converged
 0 +---|---|---|---|---|---|---> Epochs
   0   10  20  30  40  50  60
```

---

## Hands-on Exploration

1. Run gradient descent with lr=0.0001, lr=0.01, lr=1.0 — compare how many epochs each needs
2. Find the largest learning rate that still converges — this is the practical sweet spot
3. Observe what happens to the parameter values (m and b) when the learning rate is too high

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Generate data: y = 2x + 1 + noise ---
X = np.linspace(0, 5, 20)
y_true = 2 * X + 1 + np.random.randn(20) * 0.5
n = len(X)

# @param custom_lr float 0.0001 1.0 0.01 0.001
custom_lr = 0.01

def train(lr, epochs=80):
    """Run gradient descent and return loss history."""
    m, b = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        error = (m * X + b) - y_true
        mse = np.mean(error ** 2)
        losses.append(min(mse, 100))  # cap for display
        m -= lr * (2 / n) * np.sum(error * X)
        b -= lr * (2 / n) * np.sum(error)
    return losses, m, b

# --- Compare three learning rates + your custom rate ---
rates = [0.0001, 0.01, 0.5, custom_lr]
labels = ["Too small", "Just right", "Too large", f"Your choice ({custom_lr})"]

for lr, label in zip(rates, labels):
    losses, m, b = train(lr)
    print(f"\nlr={lr:<8} ({label})")
    print(f"  Final m={m:.3f}, b={b:.3f}, loss={losses[-1]:.4f}")

    # Text-based loss curve (sample every 10 epochs)
    print("  Loss: ", end="")
    for i in range(0, len(losses), 10):
        bar_len = int(losses[i] / 2)
        print(f"\n    Epoch {i:>2}: {'#' * min(bar_len, 40):40s} {losses[i]:.2f}", end="")
    print()

# --- Find the tipping point ---
print("\n--- Learning rate sweep ---")
print(f"{'lr':>10}  {'final_loss':>10}  status")
for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
    losses, m, b = train(lr, epochs=100)
    status = "converged" if losses[-1] < 1.0 else ("slow" if losses[-1] < 5.0 else "DIVERGED")
    print(f"{lr:>10.3f}  {losses[-1]:>10.4f}  {status}")
```

---

## Key Takeaways

- **The learning rate is the single most important hyperparameter.** It controls whether training succeeds or fails.
- **Too small = slow convergence.** The model learns but wastes computation crawling toward the minimum.
- **Too large = divergence.** The model overshoots the minimum and the loss explodes.
- **The sweet spot depends on the problem.** There is no universal best learning rate — you must experiment.
- **Always plot the loss curve.** A quick glance at loss-over-epochs instantly reveals learning rate problems.
