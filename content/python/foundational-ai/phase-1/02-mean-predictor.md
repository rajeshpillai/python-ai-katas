# Mean Predictor

> Phase 1 — What Does It Mean to Learn? | Kata 1.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 1.1, we tried different constants and noticed the mean performed well. Now we prove **why**: the mean minimizes the sum of squared errors. This is not a coincidence -- it is a mathematical fact that underpins most of machine learning.

If you take the derivative of the squared error with respect to the constant prediction c, set it to zero, and solve, you get c = mean(y). The mean is the optimal constant predictor under squared loss. This is the first example of **optimization** in this course -- finding the value that minimizes a loss function.

Understanding why the mean is special also reveals a deeper truth: **the loss function you choose determines the optimal prediction.** Squared error gives you the mean. Absolute error gives you the median. This connection between loss functions and optimal predictors runs through all of ML.

### Why naive approaches fail

Guessing a constant by intuition (e.g., "prices are around 300k") might be close, but it is never optimal. Even being off by a small amount compounds across hundreds of data points. The mean is the mathematically provable best you can do with a single number under squared error -- and proving this yourself builds the foundation for understanding gradient descent later.

### Mental models

- **Balance point on a ruler**: place weights at each data point on a ruler. The mean is the exact balance point -- the fulcrum where the ruler doesn't tip. Moving it in any direction increases the total "torque" (error).
- **Least regret number**: if you had to whisper one number to predict every house price, the mean minimizes your total embarrassment (squared error).

### Visual explanations

```
Prices:  190  230  250  280  310  340  350  370  420  480
          |    |    |    |    |    |    |    |    |    |
          +----+----+----+--|-+----+----+----+----+----+
                            322 (mean)

Errors from mean:    -132  -92  -72  -42  -12  +18  +28  +48  +98  +158
                      sum of errors = 0  (always true for the mean!)

Try c=250:           -60  -20    0  +30  +60  +90 +100 +120 +170 +230
                      sum of errors = +720  (positive = too low)
```

---

## Hands-on Exploration

1. Compute MSE for many candidate constants -- verify the mean gives the minimum
2. Show that errors from the mean always sum to zero (the balance property)
3. Compare mean vs median as predictors under both MSE and MAE

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- House price dataset ---
prices = np.array([250, 310, 340, 190, 420, 230, 350, 480, 280, 370])
mean_price = prices.mean()
median_price = np.median(prices)

print(f"Prices: {prices}")
print(f"Mean:   {mean_price:.0f}k")
print(f"Median: {median_price:.0f}k\n")

# --- Proof: mean minimizes MSE ---
# Sweep constants from 150 to 500, find which minimizes MSE
candidates = np.arange(150, 501, 1)
mse_values = np.array([(( prices - c) ** 2).mean() for c in candidates])
best_c = candidates[np.argmin(mse_values)]
print(f"Optimal constant by exhaustive search: {best_c}k")
print(f"Mean of data:                          {mean_price:.0f}k")
print(f"They match!\n")

# --- The balance property ---
errors_from_mean = prices - mean_price
print(f"Errors from mean: {errors_from_mean}")
print(f"Sum of errors:    {errors_from_mean.sum():.10f}  (zero!)\n")

# --- Mean vs Median under MSE and MAE ---
mse_mean = ((prices - mean_price) ** 2).mean()
mse_median = ((prices - median_price) ** 2).mean()
mae_mean = np.abs(prices - mean_price).mean()
mae_median = np.abs(prices - median_price).mean()

print(f"{'Predictor':<10} {'MSE':>10} {'MAE':>10}")
print("=" * 32)
print(f"{'Mean':<10} {mse_mean:>10.1f} {mae_mean:>10.1f}")
print(f"{'Median':<10} {mse_median:>10.1f} {mae_median:>10.1f}")
print(f"\nMean wins on MSE, Median wins on MAE\n")

# --- Visual: MSE landscape around the mean ---
print("MSE as we move the constant prediction:")
for c in range(200, 460, 25):
    mse = ((prices - c) ** 2).mean()
    bar = "#" * int(mse / 400)
    marker = " <-- mean" if abs(c - mean_price) < 13 else ""
    print(f"  c={c:>3}: MSE={mse:>8.0f}  {bar}{marker}")
```

---

## Key Takeaways

- **The mean minimizes squared error.** This is provable with calculus -- take the derivative, set to zero, solve.
- **Errors from the mean sum to zero.** The mean is the balance point of the dataset.
- **The median minimizes absolute error.** Different loss functions have different optimal constants.
- **This is your first optimization.** Finding the value that minimizes a loss function is what all of ML does.
- **The mean predictor is the baseline to beat.** Every regression model's R-squared is measured against this.
