# Linear Regression

> Phase 1 — What Does It Mean to Learn? | Kata 1.4

---

## Concept & Intuition

### What problem are we solving?

We want to predict a continuous value (house price) from input features (square footage) using the simplest possible formula: **y = mx + b**. The slope m captures how much price increases per additional square foot, and the intercept b is the base price. Unlike k-NN, which stores all data, linear regression distills the entire dataset into just two numbers.

The beauty of linear regression is that the optimal m and b can be computed exactly using the **normal equation** -- no iteration, no guessing, no gradient descent. You plug in the data and out come the perfect parameters. This closed-form solution exists because the squared error loss for a linear model is a smooth, bowl-shaped function with exactly one minimum.

Linear regression is also where we meet **R-squared**, the standard measure of model quality. R-squared tells you what fraction of the variance in prices your model explains. R-squared = 0 means you're no better than the mean predictor. R-squared = 1 means perfect prediction.

### Why naive approaches fail

You could eyeball a line through a scatter plot, but "looks about right" is not rigorous. Even a small tilt error in the slope compounds across the data range. The normal equation guarantees the mathematically optimal line -- no human judgment needed. This is the difference between intuition and optimization.

### Mental models

- **Rubber band through nails**: imagine each data point is a nail on a board. Stretch a rubber band to pass as close to all nails as possible. The band's resting position is the least-squares line.
- **Compression**: 100 data points compressed into 2 numbers (m, b). You lose detail but capture the trend.

### Visual explanations

```
Price (k)
  480 |                              *
  420 |                      *
  350 |                *  *
  310 |           * /
  250 |        */
  230 |      /
  190 |   */
      +---/--------------------------- sqft
       1000    1500    2000    2500

  Line: price = 0.19 * sqft - 21  (approximate)
  R-squared: how much scatter is explained by the line
```

---

## Hands-on Exploration

1. Compute the slope and intercept using the normal equation (closed-form)
2. Compare predictions vs actual values for each house
3. Calculate R-squared and interpret what it means

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Dataset ---
sqft = np.array([1100, 1300, 1400, 1600, 1700, 1800, 2100, 2400, 1500, 1900])
prices = np.array([190, 230, 250, 310, 340, 350, 420, 480, 280, 370])

# --- Normal equation: solve for m and b in y = mx + b ---
# Using: X @ w = y, where X has columns [sqft, 1], w = [m, b]
X = np.column_stack([sqft, np.ones(len(sqft))])
# w = (X^T X)^-1 X^T y
w = np.linalg.inv(X.T @ X) @ X.T @ prices
m, b = w[0], w[1]

print(f"Fitted line: price = {m:.4f} * sqft + {b:.2f}")
print(f"Slope: each extra sqft adds ${m*1000:.0f} to price\n")

# --- Predictions vs Actual ---
predictions = m * sqft + b
print(f"{'House':>6} {'Sqft':>6} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print("=" * 42)
for i in range(len(sqft)):
    err = prices[i] - predictions[i]
    print(f"{i+1:>6} {sqft[i]:>6} {prices[i]:>8.0f} {predictions[i]:>10.1f} {err:>+8.1f}")

# --- R-squared ---
ss_res = ((prices - predictions) ** 2).sum()      # residual sum of squares
ss_tot = ((prices - prices.mean()) ** 2).sum()     # total sum of squares
r_squared = 1 - ss_res / ss_tot

print(f"\nResidual SS:  {ss_res:.1f}")
print(f"Total SS:     {ss_tot:.1f}")
print(f"R-squared:    {r_squared:.4f}")
print(f"Interpretation: model explains {r_squared*100:.1f}% of price variance\n")

# --- Compare to mean predictor baseline ---
baseline_mse = ((prices - prices.mean()) ** 2).mean()
model_mse = ((prices - predictions) ** 2).mean()
print(f"Mean predictor MSE:     {baseline_mse:.1f}")
print(f"Linear regression MSE:  {model_mse:.1f}")
print(f"Improvement:            {(1 - model_mse/baseline_mse)*100:.1f}%")

# --- Predict new houses ---
new_sqft = np.array([1250, 2000, 2600])
new_preds = m * new_sqft + b
print(f"\nNew predictions:")
for s, p in zip(new_sqft, new_preds):
    print(f"  {s} sqft --> ${p:.0f}k")
```

---

## Key Takeaways

- **Linear regression fits y = mx + b to data.** The slope and intercept capture the trend in two numbers.
- **The normal equation gives the exact solution.** No iteration needed -- pure linear algebra.
- **R-squared measures explained variance.** 0 = no better than the mean, 1 = perfect fit.
- **Linear regression is the first real model.** It uses features to make different predictions for different inputs.
- **The improvement over the mean predictor is the whole point.** R-squared directly measures this improvement.
