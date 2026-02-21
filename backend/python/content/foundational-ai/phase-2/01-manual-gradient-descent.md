# Manual Gradient Descent

> Phase 2 — Optimization | Kata 2.1

---

## Concept & Intuition

### What problem are we solving?

In Phase 1 we learned to measure error with a loss function like MSE. Now the question is: **how do we systematically reduce that error?** We have a linear model `y = mx + b` with two knobs — slope `m` and intercept `b`. We need a method that tells us which direction to turn each knob, and by how much, to make our predictions better.

Gradient descent is that method. The **gradient** of the loss function tells us the direction of steepest increase. By moving in the *opposite* direction, we walk downhill toward the minimum loss. For MSE with a linear model, the math is clean: we compute partial derivatives of the loss with respect to `m` and `b`, then subtract a small fraction of each gradient from the current parameter values.

Each update step nudges `m` and `b` toward values that fit the data better. Repeat this hundreds of times, and the model converges to parameters that minimize the error — no guessing, no brute-force search.

### Why naive approaches fail

You could try random search — pick random values of `m` and `b` and keep the best. But with two parameters this is wasteful, and with thousands of parameters (as in neural networks) it is hopeless. You could try a grid search, but the number of grid points grows exponentially with the number of parameters. Gradient descent scales because it uses the **slope of the loss surface** to make informed steps, not blind ones.

### Mental models

- **Rolling a ball downhill**: The loss surface is a bowl. The gradient points uphill. Negate it and you roll toward the bottom.
- **Adjusting a recipe**: Taste too salty? Reduce salt a little. Too bland? Add a little. The gradient tells you *which ingredient* to adjust and *in which direction*.
- **Compass on a mountain**: You are in fog and want to reach the valley. The gradient is a compass that always points downhill.

### Visual explanations

```
Loss surface (MSE as a function of m):

  Loss
   |
 8 |  *                          *
   |   *                       *
 6 |    *                    *
   |      *               *
 4 |        *           *
   |          *       *
 2 |            *   *
   |              *  <-- minimum (goal)
 0 +-------|-------|-------|-------> m
           1       2       3

  Gradient descent steps:
  Step 0: m=0.5, loss=7.2  -->  gradient is negative, so m increases
  Step 1: m=1.2, loss=4.1  -->  gradient still negative, keep going
  Step 2: m=1.8, loss=1.5  -->  getting closer
  Step 3: m=2.1, loss=0.3  -->  nearly there
```

---

## Hands-on Exploration

1. Write out the MSE formula by hand for 3 data points — then take the derivative with respect to `m`
2. Start with a bad guess for `m` and `b`, compute the gradient, and do one update step on paper
3. Watch how the loss decreases with each step — does it slow down near the minimum?

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Generate data: y = 2x + 1 + noise ---
X = np.linspace(0, 5, 20)
y_true = 2 * X + 1 + np.random.randn(20) * 0.5

# --- Initialize parameters (bad guess) ---
m = 0.0
b = 0.0
# @param lr float 0.001 0.1 0.01 0.001
lr = 0.01   # learning rate
# @param epochs int 10 200 60 10
epochs = 60
n = len(X)

print(f"{'Epoch':>5}  {'m':>6}  {'b':>6}  {'MSE':>8}")
print("-" * 32)

for epoch in range(epochs):
    # Forward pass: predictions
    y_pred = m * X + b

    # Compute MSE loss
    error = y_pred - y_true
    mse = np.mean(error ** 2)

    # Compute gradients (partial derivatives of MSE)
    dm = (2 / n) * np.sum(error * X)   # dMSE/dm
    db = (2 / n) * np.sum(error)       # dMSE/db

    # Update parameters (step opposite to gradient)
    m = m - lr * dm
    b = b - lr * db

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"{epoch:>5}  {m:>6.3f}  {b:>6.3f}  {mse:>8.4f}")

print(f"\nLearned: y = {m:.3f}x + {b:.3f}")
print(f"True:    y = 2.000x + 1.000")
```

---

## Key Takeaways

- **The gradient points uphill; we step downhill.** Subtracting the gradient from parameters reduces the loss.
- **MSE gradients have closed-form formulas.** For `y = mx + b`, dMSE/dm and dMSE/db are simple averages over the data.
- **Each epoch refines the parameters.** Loss drops quickly at first, then slows as we approach the minimum.
- **This is the foundation of all neural network training.** Every deep learning optimizer is a variation of this basic loop: predict, compute loss, compute gradients, update.
- **Start with a bad guess — gradient descent finds the way.** The algorithm does not need a good initialization to eventually converge (for convex problems like linear regression).
