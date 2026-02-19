# Constant Predictor

> Phase 1 — What Does It Mean to Learn? | Kata 1.1

---

## Concept & Intuition

### What problem are we solving?

Before building any sophisticated model, we need a **baseline** -- the dumbest possible predictor. A constant predictor ignores all input features and predicts the same value for every data point. If someone asks "what will this house sell for?" it always answers the same number, regardless of square footage, location, or anything else.

Why would anyone use such a useless model? Because it sets the **floor**. If your fancy neural network can't beat a constant predictor, your model is worthless. Every prediction task should start here: pick a constant, measure its error, and use that as the bar to clear.

The constant predictor also introduces the fundamental question of ML: **how do we measure how wrong we are?** We need a number that quantifies prediction quality. This is where error metrics begin.

### Why naive approaches fail

You might think "just pick a reasonable number." But which one? If house prices range from 150k to 500k, should you predict 150k? 500k? 325k? Each choice produces wildly different errors. Without a principled way to choose the constant and measure error, you're guessing about your guessing.

### Mental models

- **Weather forecaster who only says one temperature**: "It's 72 degrees" every single day. Some days they're close, most days they're off. The total error across all days tells you how bad this strategy is.
- **Dart player throwing at one spot**: they always aim at the same point. The average distance from the bullseye is their error.

### Visual explanations

```
Actual prices:      250  310  340  190  420  230  350  480
                     |    |    |    |    |    |    |    |
Predict 300:        -50  +10  +40 -110 +120  -70  +50 +180
                     ~~   ~    ~~   ~~~  ~~~  ~~   ~~  ~~~~
                     (every prediction is the same line)

Errors:  some small, some huge -- but it's our BASELINE
```

---

## Hands-on Exploration

1. Pick any constant value and compute the total squared error against the dataset
2. Try several constants (min, max, midpoint, median) and compare their errors
3. Notice which constant gives the lowest error -- this foreshadows Kata 1.2

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- House price dataset (thousands) ---
sqft = np.array([1400, 1600, 1700, 1100, 2100, 1300, 1800, 2400, 1500, 1900])
prices = np.array([250, 310, 340, 190, 420, 230, 350, 480, 280, 370])

print("House prices:", prices)
print(f"Range: {prices.min()}k to {prices.max()}k\n")

# --- Try different constant predictions ---
def compute_errors(y_true, constant):
    errors = y_true - constant
    mse = (errors ** 2).mean()
    mae = np.abs(errors).mean()
    return mse, mae

constants = {
    "Minimum (190)": prices.min(),
    "Maximum (480)": prices.max(),
    "Midpoint (335)": (prices.min() + prices.max()) / 2,
    "Mean (322)": prices.mean(),
    "Median (315)": float(np.median(prices)),
    "Random (400)": 400,
}

print(f"{'Constant':<20} {'MSE':>10} {'MAE':>8}")
print("=" * 42)
for name, c in constants.items():
    mse, mae = compute_errors(prices, c)
    bar = "#" * int(mse / 300)
    print(f"{name:<20} {mse:>10.1f} {mae:>8.1f}  {bar}")

# --- The constant predictor in action ---
best_constant = prices.mean()
print(f"\nBest constant (mean): {best_constant}")
print(f"\nPrediction vs Actual:")
print(f"{'House':>6} {'Actual':>8} {'Predict':>8} {'Error':>8}")
print("-" * 34)
for i in range(len(prices)):
    err = prices[i] - best_constant
    print(f"{i+1:>6} {prices[i]:>8.0f} {best_constant:>8.0f} {err:>+8.0f}")

rmse = np.sqrt(((prices - best_constant) ** 2).mean())
print(f"\nBaseline RMSE: {rmse:.1f}k  <-- any real model must beat this")
```

---

## Key Takeaways

- **The constant predictor is your baseline.** Any useful model must produce lower error than predicting a single value for everything.
- **Choosing the constant matters.** Not all constants are equal -- the mean turns out to be optimal for squared error (proven in Kata 1.2).
- **Error metrics quantify wrongness.** MSE and MAE give different perspectives on prediction quality.
- **Always start with a baseline.** Before building complex models, know what "doing nothing smart" looks like.
- **The gap between baseline and perfect is your opportunity.** The larger the gap, the more a model can potentially help.
