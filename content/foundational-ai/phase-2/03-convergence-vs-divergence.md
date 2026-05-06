# Convergence vs Divergence

> Phase 2 — Optimization | Kata 2.3

---

## Concept & Intuition

### What problem are we solving?

Gradient descent does not always work. Sometimes the loss drops steadily toward zero — this is **convergence**. Other times the loss climbs, oscillates, or explodes to infinity — this is **divergence**. Understanding *why* each happens is essential because in real projects you will encounter both, and you need to diagnose the problem quickly.

Convergence depends on the relationship between the learning rate and the curvature of the loss surface. For a quadratic loss (like MSE with linear regression), there is a precise mathematical boundary: if the learning rate exceeds `2 / L` where `L` is the largest eigenvalue of the Hessian, gradient descent diverges. In practice, this means the learning rate must be small enough relative to the steepness of the loss landscape.

### Why naive approaches fail

Without understanding convergence theory, debugging is guesswork. You might blame the data, the model architecture, or the number of epochs — when the real issue is that the learning rate crossed the stability threshold. Divergence can also be subtle: the loss might decrease for a while, then suddenly spike. Knowing the warning signs lets you intervene before wasting hours of training.

### Mental models

- **Convergence = damped oscillation**: Like a pendulum with friction — swings get smaller until it rests at the bottom.
- **Divergence = amplified oscillation**: Like pushing a swing harder each time — the arcs grow until it flips over.
- **The stability boundary**: Think of balancing on a tightrope. Tiny corrections keep you stable. Big corrections make you overcorrect, and each correction is worse than the last.

### Visual explanations

```
Convergence (lr=0.01):          Divergence (lr=0.5):

  Loss                            Loss
   |                               |                    *
 8 | *                           8 | *
   |  *                            |  *              *
 6 |   *                         6 |
   |    *                          |   *          *
 4 |     *                       4 |
   |       *                       |    *      *
 2 |         * *                 2 |
   |             * * * * *         |      *  *
 0 +-------------------> Epoch   0 +-------------------> Epoch

   Loss steadily decreases          Loss bounces & grows
```

---

## Hands-on Exploration

1. Run gradient descent and record the loss at every epoch — does it monotonically decrease, oscillate then settle, or explode?
2. Find the exact learning rate where behavior transitions from convergence to divergence
3. Track the parameter values (m and b) during divergence — watch them grow without bound

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Generate data: y = 3x + 2 + noise ---
X = np.linspace(0, 4, 15)
y_true = 3 * X + 2 + np.random.randn(15) * 0.5
n = len(X)

def gradient_descent(lr, epochs=50):
    """Returns loss trajectory and final params."""
    m, b = 0.0, 0.0
    trajectory = []
    for epoch in range(epochs):
        error = (m * X + b) - y_true
        mse = np.mean(error ** 2)
        trajectory.append(mse)
        if mse > 1e10:  # early stop on divergence
            break
        m -= lr * (2 / n) * np.sum(error * X)
        b -= lr * (2 / n) * np.sum(error)
    return trajectory, m, b

# --- Show convergence, oscillation, and divergence ---
test_rates = [0.01, 0.05, 0.08, 0.12]

for lr in test_rates:
    traj, m, b = gradient_descent(lr, epochs=40)

    # Classify behavior
    if traj[-1] < 1.0:
        status = "CONVERGED"
    elif traj[-1] < traj[0]:
        status = "OSCILLATING (slow)"
    else:
        status = "DIVERGED"

    print(f"\nlr={lr:.2f} -> {status}")
    print(f"  Start loss: {traj[0]:.2f}  Final loss: {traj[-1]:.2f}")

    # Text plot: loss trajectory
    print("  Trajectory: ", end="")
    step = max(1, len(traj) // 10)
    for i in range(0, len(traj), step):
        bar = "#" * min(int(traj[i] / 2), 35)
        print(f"\n    Ep {i:>2}: {bar} {traj[i]:.2f}", end="")
    print()

# --- Find the critical learning rate ---
print("\n--- Stability boundary search ---")
print(f"{'lr':>8}  {'final_loss':>12}  {'epochs':>6}  verdict")
for lr in [0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.11, 0.12]:
    traj, m, b = gradient_descent(lr, epochs=100)
    verdict = "stable" if traj[-1] < 2.0 else "UNSTABLE"
    print(f"{lr:>8.2f}  {traj[-1]:>12.2f}  {len(traj):>6}  {verdict}")
```

---

## Key Takeaways

- **Convergence means the loss steadily decreases to a minimum.** This is the goal of training.
- **Divergence means the loss grows without bound.** The parameters explode because each update overcorrects.
- **There is a sharp boundary.** A small increase in learning rate can flip behavior from convergence to divergence.
- **Oscillation is a warning sign.** If the loss bounces up and down, you are near the stability edge — reduce the learning rate.
- **Always monitor loss trajectory, not just the final value.** A final loss that looks okay might hide dangerous oscillations that happened during training.
