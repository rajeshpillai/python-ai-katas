# Visualizing Loss Curves

> Phase 2 — Optimization | Kata 2.4

---

## Concept & Intuition

### What problem are we solving?

A single loss number at the end of training tells you almost nothing. Was the model still improving when you stopped? Did it plateau 50 epochs ago? Did it spike in the middle and recover? The **loss curve** — loss plotted over epochs — is the most important diagnostic tool in machine learning. It tells you whether training is healthy at a glance.

This kata teaches you to build text-based loss visualizations from scratch using only numpy and print statements. In real projects you would use plotting libraries, but understanding how to read and interpret loss curves matters far more than the tool you use to draw them. We will compare curves from different hyperparameter settings and learn to read the story each curve tells.

### Why naive approaches fail

Many beginners train a model, look at the final accuracy, and call it done. They miss critical information: the model might have overfit (training loss drops but validation loss rises), the learning rate might be too high (loss oscillates), or training might have stalled (loss flattens early). Without visualizing the loss curve, these problems are invisible.

### Mental models

- **Loss curve as a patient's heart monitor**: A healthy curve shows steady decline. Spikes mean trouble. A flatline means nothing is happening.
- **Reading the shape**: Steep drop then plateau = fast learning then convergence. Gradual decline = slow but steady. Sawtooth pattern = learning rate too high. Flat line = learning rate too low or model cannot learn.

### Visual explanations

```
What different loss curves tell you:

Healthy:           Too slow:          Too fast:          Overfit:
  |*                 |*                 |*   *            |* * *
  | *                |*                 | * * *           | *   *
  |  *               | *               |  *   *          |  * t *  (t=test)
  |   **             |  *              |   *   *         |   **
  |     ***          |   **            |    *   *        |    *  *
  |        ****      |     ***         |     *           |        ***
  +--------->        +--------->       +--------->       +--------->
   epochs             epochs            epochs            epochs
```

---

## Hands-on Exploration

1. Train the same model with 3 different learning rates — plot all three loss curves side by side
2. Identify which curve shows underfitting (still decreasing at the end) and which shows convergence (flattened)
3. Experiment with different epoch counts — when does more training stop helping?

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Generate data: y = 2.5x + 3 + noise ---
X = np.linspace(0, 5, 25)
y_true = 2.5 * X + 3 + np.random.randn(25) * 0.8
n = len(X)

def train_and_record(lr, epochs):
    """Train linear regression, return loss at every epoch."""
    m, b = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        error = (m * X + b) - y_true
        mse = np.mean(error ** 2)
        losses.append(mse)
        m -= lr * (2 / n) * np.sum(error * X)
        b -= lr * (2 / n) * np.sum(error)
    return np.array(losses)

def text_plot(losses, title, width=50, height=12):
    """Print a text-based loss curve."""
    print(f"\n  {title}")
    lo, hi = min(losses), max(losses)
    if hi - lo < 0.001:
        hi = lo + 1  # avoid division by zero
    # Sample epochs to fit width
    indices = np.linspace(0, len(losses) - 1, min(width, len(losses))).astype(int)
    sampled = losses[indices]
    for row in range(height, -1, -1):
        threshold = lo + (hi - lo) * row / height
        line = "  "
        if row == height:
            line += f"{hi:>7.1f} |"
        elif row == 0:
            line += f"{lo:>7.1f} |"
        else:
            line += "        |"
        for val in sampled:
            if abs(val - threshold) < (hi - lo) / height / 2:
                line += "*"
            elif row == 0:
                line += "-"
            else:
                line += " "
        print(line)
    print(f"          +{''.join(['-'] * len(sampled))}")
    print(f"           Epoch 0{' ' * (len(sampled) - 10)}Epoch {len(losses)-1}")

# --- Compare three scenarios ---
configs = [
    (0.001, 100, "Slow (lr=0.001, 100 epochs)"),
    (0.015, 100, "Good (lr=0.015, 100 epochs)"),
    (0.015,  30, "Short (lr=0.015, 30 epochs)"),
]

for lr, epochs, title in configs:
    losses = train_and_record(lr, epochs)
    text_plot(losses, title)
    print(f"    Start: {losses[0]:.2f}  End: {losses[-1]:.2f}  "
          f"Drop: {((losses[0] - losses[-1]) / losses[0] * 100):.0f}%")

# --- Epoch milestone analysis ---
print("\n--- When does training stop helping? ---")
losses = train_and_record(0.015, 200)
milestones = [1, 5, 10, 25, 50, 100, 150, 200]
print(f"{'Epoch':>6}  {'Loss':>8}  {'Improvement':>12}")
prev = losses[0]
for ep in milestones:
    improvement = prev - losses[ep - 1]
    print(f"{ep:>6}  {losses[ep-1]:>8.4f}  {improvement:>+12.4f}")
    prev = losses[ep - 1]
```

---

## Key Takeaways

- **The loss curve is your most important diagnostic tool.** Always visualize loss over epochs before drawing conclusions.
- **A healthy curve drops steeply then flattens.** The steep part is rapid learning; the flat part means convergence.
- **Compare curves to choose hyperparameters.** Side-by-side curves reveal which learning rate and epoch count work best.
- **Diminishing returns are normal.** Most improvement happens in early epochs. Late epochs give tiny gains.
- **You do not need a plotting library to read curves.** Even printed numbers at milestones reveal the training story.
