# Exponential Smoothing

> Phase 8 — Time Series & Sequential Data | Kata 8.4

---

## Concept & Intuition

### What problem are we solving?

Exponential smoothing is a family of forecasting methods based on a simple, powerful idea: recent observations should get more weight than older ones, and the weights should decay exponentially. Unlike ARIMA, which models temporal dependencies through autoregressive and moving average terms, exponential smoothing directly models the *level*, *trend*, and *seasonality* of a time series as evolving states that are updated with each new observation.

There are three main variants. **Simple Exponential Smoothing (SES)** handles series with no trend and no seasonality -- it estimates only the level (smoothed average). **Holt's method** (double exponential smoothing) adds a trend component, so it can forecast upward or downward trends. **Holt-Winters** (triple exponential smoothing) adds both trend and seasonality, making it suitable for series with regular periodic patterns like monthly sales or hourly electricity demand.

The smoothing parameters (alpha for level, beta for trend, gamma for seasonality) control how quickly the model adapts to new data. High alpha (close to 1) means the model reacts strongly to recent changes. Low alpha (close to 0) means the model is sluggish, relying more on historical patterns. These parameters are typically optimized by minimizing the in-sample forecast error.

### Why naive approaches fail

A simple moving average gives equal weight to all observations in the window, which means a data point from 30 days ago has the same influence as yesterday's point. This is clearly suboptimal -- recent data is more relevant for forecasting. You could use a weighted moving average with custom weights, but then you have to choose the weights manually. Exponential smoothing provides an optimal, principled weighting scheme where the weights decay geometrically and the decay rate is learned from data.

### Mental models

- **Exponential memory**: the model remembers all past observations, but with exponentially fading memory. Yesterday's observation is remembered vividly; last year's observation is a faint echo.
- **Three evolving states**: think of level, trend, and seasonality as three dials that the model continuously adjusts as new data arrives. The smoothing parameters control how quickly each dial turns.
- **Adaptive baseline**: SES is like a moving target that tracks the current level of the series. Holt adds a velocity (trend). Holt-Winters adds a seasonal oscillation on top.

### Visual explanations

```
Exponential Weights (alpha = 0.3):
  y_t:    weight = 0.30    ||||||||||||||||
  y_{t-1}: weight = 0.21   |||||||||||
  y_{t-2}: weight = 0.15   ||||||||
  y_{t-3}: weight = 0.10   |||||
  y_{t-4}: weight = 0.07   ||||
  y_{t-5}: weight = 0.05   |||
  ...                       (exponential decay)

Three Variants:
  SES:           Level only        --> flat forecast line
  Holt:          Level + Trend     --> straight forecast line (up or down)
  Holt-Winters:  Level + Trend + Season --> wavy forecast with trend

Holt-Winters Update Equations:
  Level:    l_t = alpha * (y_t - s_{t-m}) + (1-alpha) * (l_{t-1} + b_{t-1})
  Trend:    b_t = beta * (l_t - l_{t-1}) + (1-beta) * b_{t-1}
  Season:   s_t = gamma * (y_t - l_t) + (1-gamma) * s_{t-m}
  Forecast: y_{t+h} = l_t + h*b_t + s_{t+h-m}
```

---

## Hands-on Exploration

1. Generate a series with no trend and no seasonality. Apply SES with different alpha values and observe how alpha controls responsiveness vs smoothness.
2. Generate a trended series. Show that SES fails to track the trend, while Holt's method captures it by maintaining a separate trend component.
3. Generate a trended-seasonal series. Apply Holt-Winters and compare its forecast to Holt's method (which misses the seasonality).
4. Optimize the smoothing parameters automatically and compare the optimized model to manual choices.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)

# --- Generate synthetic data with trend + seasonality ---
n = 365 * 2
t = np.arange(n)
dates = pd.date_range("2022-01-01", periods=n, freq="D")

# Components
trend = 0.05 * t + 100
season = 20 * np.sin(2 * np.pi * t / 365.25)
noise = np.random.normal(0, 5, n)
y = trend + season + noise

ts = pd.Series(y, index=dates)
train = ts[:500]
test = ts[500:]
h = len(test)

print(f"=== Dataset: {n} days, train={len(train)}, test={len(test)} ===\n")

# ============================================================
# 1. Simple Exponential Smoothing (SES)
# ============================================================
print("=== 1. Simple Exponential Smoothing ===\n")

alphas = [0.1, 0.3, 0.6, 0.9]
ses_results = {}

for alpha in alphas:
    ses = SimpleExpSmoothing(train).fit(
        smoothing_level=alpha, optimized=False
    )
    forecast = ses.forecast(h)
    mae = mean_absolute_error(test, forecast)
    ses_results[alpha] = {"forecast": forecast, "mae": mae}
    print(f"  alpha={alpha}: MAE={mae:.2f}, "
          f"forecast={forecast.values[0]:.2f} (constant line)")

# Optimized SES
ses_opt = SimpleExpSmoothing(train).fit()
forecast_opt = ses_opt.forecast(h)
mae_opt = mean_absolute_error(test, forecast_opt)
print(f"  Optimized alpha={ses_opt.params['smoothing_level']:.4f}: "
      f"MAE={mae_opt:.2f}")
print(f"  (SES cannot capture trend or seasonality)\n")

# ============================================================
# 2. Holt's Method (trend, no seasonality)
# ============================================================
print("=== 2. Holt's Method (adds trend) ===\n")

holt = ExponentialSmoothing(
    train, trend="add", seasonal=None
).fit()
forecast_holt = holt.forecast(h)
mae_holt = mean_absolute_error(test, forecast_holt)

print(f"  alpha (level):  {holt.params['smoothing_level']:.4f}")
print(f"  beta (trend):   {holt.params['smoothing_trend']:.4f}")
print(f"  MAE: {mae_holt:.2f}")
print(f"  (Captures trend but misses seasonality)\n")

# ============================================================
# 3. Holt-Winters (trend + seasonality)
# ============================================================
print("=== 3. Holt-Winters (trend + seasonality) ===\n")

# Additive seasonality
hw_add = ExponentialSmoothing(
    train, trend="add", seasonal="add", seasonal_periods=365
).fit()
forecast_hw_add = hw_add.forecast(h)
mae_hw_add = mean_absolute_error(test, forecast_hw_add)

print(f"  Additive Holt-Winters:")
print(f"    alpha (level):    {hw_add.params['smoothing_level']:.4f}")
print(f"    beta (trend):     {hw_add.params['smoothing_trend']:.4f}")
print(f"    gamma (seasonal): {hw_add.params['smoothing_seasonal']:.4f}")
print(f"    MAE: {mae_hw_add:.2f}\n")

# Multiplicative seasonality
hw_mul = ExponentialSmoothing(
    train, trend="add", seasonal="mul", seasonal_periods=365
).fit()
forecast_hw_mul = hw_mul.forecast(h)
mae_hw_mul = mean_absolute_error(test, forecast_hw_mul)

print(f"  Multiplicative Holt-Winters:")
print(f"    MAE: {mae_hw_mul:.2f}\n")

# ============================================================
# Summary comparison
# ============================================================
print("=== Summary ===\n")
methods = {
    "SES (optimized)": mae_opt,
    "Holt (trend)": mae_holt,
    "Holt-Winters (add)": mae_hw_add,
    "Holt-Winters (mul)": mae_hw_mul,
}

for name, mae in sorted(methods.items(), key=lambda x: x[1]):
    bar = "|" * int(50 / mae)
    print(f"  {name:<25}: MAE = {mae:.2f}  {bar}")

# ============================================================
# Smoothing parameter effect
# ============================================================
print("\n=== Exponential Weight Decay ===\n")
for alpha in [0.1, 0.3, 0.7]:
    weights = [(1 - alpha) ** i * alpha for i in range(10)]
    total = sum(weights)
    print(f"  alpha={alpha}:")
    for i, w in enumerate(weights):
        bar = "|" * int(w * 100)
        print(f"    t-{i}: weight={w:.4f}  {bar}")
    print(f"    Sum of first 10 weights: {total:.4f}\n")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SES with different alphas
axes[0, 0].plot(train.index[-100:], train[-100:], 'b-', linewidth=0.8, label="Train")
axes[0, 0].plot(test.index[:60], test[:60], 'g-', linewidth=0.8, label="Actual")
for alpha in [0.1, 0.5, 0.9]:
    ses_fitted = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
    fc = ses_fitted.forecast(60)
    axes[0, 0].plot(fc.index, fc, '--', linewidth=1.5, label=f"SES alpha={alpha}")
axes[0, 0].set_title("SES: Effect of Alpha")
axes[0, 0].legend(fontsize=7)

# SES vs Holt vs Holt-Winters
axes[0, 1].plot(test.index, test, 'g-', linewidth=0.8, label="Actual")
axes[0, 1].plot(test.index, forecast_opt.values[:h], 'r--', label="SES")
axes[0, 1].plot(test.index, forecast_holt.values[:h], 'm--', label="Holt")
axes[0, 1].plot(test.index, forecast_hw_add.values[:h], 'b--', label="Holt-Winters")
axes[0, 1].set_title("Method Comparison")
axes[0, 1].legend(fontsize=8)

# Holt-Winters fit (in-sample)
axes[1, 0].plot(train.index, train, 'b-', linewidth=0.5, alpha=0.5, label="Actual")
axes[1, 0].plot(train.index, hw_add.fittedvalues, 'r-', linewidth=1, label="Fitted")
axes[1, 0].set_title("Holt-Winters In-Sample Fit")
axes[1, 0].legend(fontsize=8)

# MAE comparison
names = list(methods.keys())
maes = [methods[n] for n in names]
colors = ["coral" if m == min(maes) else "steelblue" for m in maes]
axes[1, 1].barh(names, maes, color=colors)
axes[1, 1].set_xlabel("MAE (lower is better)")
axes[1, 1].set_title("Forecast Accuracy Comparison")
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig("exponential_smoothing.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to exponential_smoothing.png")
```

---

## Key Takeaways

- **Exponential smoothing weights recent data more heavily.** The weights decay geometrically, providing a principled alternative to moving averages.
- **Three variants handle three situations.** SES for level-only series, Holt for trended series, Holt-Winters for trended-seasonal series. Use the simplest variant that fits your data.
- **Smoothing parameters control adaptiveness.** Alpha near 1 = reactive (follows data closely). Alpha near 0 = smooth (relies on history). Parameters are optimized automatically.
- **Additive vs multiplicative seasonality matters.** If seasonal amplitude grows with the level, use multiplicative. If it stays constant, use additive.
- **Exponential smoothing often matches ARIMA for practical forecasting.** It is simpler to understand and tune, and for many business time series, it performs equally well.
