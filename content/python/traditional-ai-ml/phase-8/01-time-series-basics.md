# Time Series Basics

> Phase 8 — Time Series & Sequential Data | Kata 8.1

---

## Concept & Intuition

### What problem are we solving?

A time series is a sequence of data points ordered by time: daily stock prices, hourly temperature readings, monthly sales figures. Unlike standard tabular data where rows are independent, time series data has *temporal structure* -- the order matters, and nearby observations are related. The three fundamental components of any time series are **trend** (long-term direction: up, down, or flat), **seasonality** (regular repeating patterns: daily, weekly, yearly), and **residuals** (what remains after removing trend and seasonality).

Understanding these components is the first step in time series analysis. Before building any model, you must decompose the series and assess whether it is **stationary** -- meaning its statistical properties (mean, variance) do not change over time. Most time series models assume stationarity, so if your data has a trend or changing variance, you must transform it first (typically through differencing or log transforms).

Stationarity matters because a stationary process is predictable in a statistical sense: the patterns you learn from the past will repeat in the future. A non-stationary process is unpredictable because its behavior is fundamentally changing over time. The Augmented Dickey-Fuller (ADF) test is the standard statistical test for stationarity.

### Why naive approaches fail

Treating time series as regular tabular data and using standard cross-validation is one of the most common mistakes in data science. Standard k-fold CV randomly shuffles data, which means future observations can leak into the training set. For time series, you must always train on the past and test on the future. Similarly, computing simple statistics (mean, std) on non-stationary data is meaningless -- the mean of a trending series does not represent any actual value the series takes.

### Mental models

- **Signal decomposition**: every time series is the sum of three signals -- trend (the slowly moving baseline), seasonality (the regular oscillation), and noise (random fluctuations). Analysis means separating these components.
- **Stationarity as statistical equilibrium**: a stationary series fluctuates around a fixed mean with constant variance. Like a ball bouncing at a stable height vs one bouncing progressively higher.
- **Differencing as detrending**: if today's value minus yesterday's value removes the trend, the differenced series is stationary even if the original was not.

### Visual explanations

```
Time Series Decomposition:

  Original:    ~~~~~/\/\/\~~/\/\/\~~~/\/\/\~~~~
                                                  ↗ upward trend
  Trend:       ___________/___________________
                                                  ~ repeating pattern
  Seasonal:    /\/\/\  /\/\/\  /\/\/\  /\/\/\

  Residual:    ~~  ~ ~~  ~~ ~ ~~  ~ ~~  ~~

  Original = Trend + Seasonal + Residual (additive)
  Original = Trend * Seasonal * Residual (multiplicative)

Stationary vs Non-Stationary:
  Stationary:       ~~~/\~~~/\~~~~/\~~~~/\~~~
                    (constant mean, constant variance)

  Non-stationary:   ___/\_____/\/\______/\/\/\/\
                    (changing mean or changing variance)

Differencing:
  Original:     10  12  15  14  17  20  19  22
  Differenced:     +2  +3  -1  +3  +3  -1  +3
                (trend removed, now fluctuates around +1.7)
```

---

## Hands-on Exploration

1. Generate a synthetic time series with known trend, seasonality, and noise components. Plot the original series and its individual components.
2. Use seasonal decomposition to automatically separate trend, seasonality, and residuals. Compare the extracted components to the known true components.
3. Test for stationarity using the Augmented Dickey-Fuller test on the original series (should be non-stationary) and on the differenced series (should be stationary).
4. Apply first-order differencing and observe how it removes the trend. Apply seasonal differencing (lag-12 for monthly data) and observe how it removes seasonality.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

# --- Generate synthetic time series with known components ---
n = 365 * 3  # 3 years of daily data
t = np.arange(n)
dates = pd.date_range("2021-01-01", periods=n, freq="D")

# Known components
trend = 0.02 * t + 50                                    # linear upward trend
seasonality = 15 * np.sin(2 * np.pi * t / 365.25)        # yearly cycle
weekly = 5 * np.sin(2 * np.pi * t / 7)                   # weekly cycle
noise = np.random.normal(0, 3, n)                         # random noise

# Combine
y = trend + seasonality + weekly + noise

ts = pd.Series(y, index=dates, name="value")
print("=== Synthetic Time Series ===")
print(f"  Length: {len(ts)} days ({len(ts)/365:.1f} years)")
print(f"  Range: [{ts.min():.1f}, {ts.max():.1f}]")
print(f"  Mean: {ts.mean():.1f}, Std: {ts.std():.1f}\n")

# --- Seasonal Decomposition ---
print("=== Seasonal Decomposition (period=365) ===\n")
decomposition = seasonal_decompose(ts, model="additive", period=365)

print(f"  Trend range:    [{decomposition.trend.dropna().min():.1f}, "
      f"{decomposition.trend.dropna().max():.1f}]")
print(f"  Seasonal range: [{decomposition.seasonal.min():.1f}, "
      f"{decomposition.seasonal.max():.1f}]")
print(f"  Residual std:   {decomposition.resid.dropna().std():.2f}\n")

# --- Stationarity Testing ---
print("=== Augmented Dickey-Fuller Test ===\n")


def adf_test(series, name):
    result = adfuller(series.dropna(), autolag="AIC")
    stationary = result[1] < 0.05
    print(f"  {name}:")
    print(f"    ADF statistic: {result[0]:.4f}")
    print(f"    p-value:       {result[1]:.6f}")
    print(f"    Stationary:    {'YES' if stationary else 'NO'}")
    print(f"    Critical values: 1%={result[4]['1%']:.3f}, "
          f"5%={result[4]['5%']:.3f}, 10%={result[4]['10%']:.3f}\n")
    return stationary


adf_test(ts, "Original series")

# --- Differencing ---
ts_diff1 = ts.diff().dropna()
adf_test(ts_diff1, "First difference")

# Seasonal differencing (lag-365 for annual seasonality)
ts_seasonal_diff = ts.diff(365).dropna()
adf_test(ts_seasonal_diff, "Seasonal difference (lag=365)")

# Both: first difference of seasonal difference
ts_both = ts.diff(365).diff().dropna()
adf_test(ts_both, "Seasonal + first difference")

# --- Statistics on rolling windows to show non-stationarity ---
print("=== Rolling Statistics (window=90 days) ===\n")
rolling_mean = ts.rolling(90).mean()
rolling_std = ts.rolling(90).std()

print(f"  Rolling mean range: [{rolling_mean.dropna().min():.1f}, "
      f"{rolling_mean.dropna().max():.1f}]")
print(f"  (A stationary series would have nearly constant rolling mean)\n")

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

# Original with rolling stats
axes[0].plot(ts.index, ts, linewidth=0.5, alpha=0.6, label="Original")
axes[0].plot(rolling_mean.index, rolling_mean, 'r-', linewidth=2,
             label="90-day rolling mean")
axes[0].fill_between(rolling_std.index,
                      rolling_mean - 2 * rolling_std,
                      rolling_mean + 2 * rolling_std,
                      alpha=0.1, color='r')
axes[0].set_title("Original Time Series with Rolling Statistics")
axes[0].legend()

# Trend
axes[1].plot(decomposition.trend.index, decomposition.trend, 'g-', linewidth=2)
axes[1].plot(dates, trend, 'k--', linewidth=1, label="True trend")
axes[1].set_title("Extracted Trend vs True Trend")
axes[1].legend()

# Seasonal
axes[2].plot(decomposition.seasonal.index[:365],
             decomposition.seasonal.values[:365], 'b-', linewidth=1)
axes[2].plot(dates[:365], seasonality[:365], 'k--', linewidth=1,
             label="True seasonality")
axes[2].set_title("Extracted Seasonality (1 year) vs True")
axes[2].legend()

# Differenced series
axes[3].plot(ts_diff1.index, ts_diff1, linewidth=0.5, alpha=0.7)
axes[3].axhline(0, color='r', linewidth=1, linestyle='--')
axes[3].set_title("First-Differenced Series (stationary)")

plt.tight_layout()
plt.savefig("time_series_basics.png", dpi=100, bbox_inches="tight")
plt.show()
print("Plot saved to time_series_basics.png")
```

---

## Key Takeaways

- **Every time series has three components: trend, seasonality, and residuals.** Understanding which components are present is the first step in analysis.
- **Stationarity means constant statistical properties over time.** Most models require it, so you must test and transform your data accordingly.
- **The ADF test detects non-stationarity.** A p-value below 0.05 means the series is stationary. If it is above 0.05, apply differencing.
- **Differencing removes trend; seasonal differencing removes seasonality.** First difference (y_t - y_{t-1}) kills linear trends. Seasonal difference (y_t - y_{t-s}) kills periodic patterns.
- **Never use standard cross-validation on time series.** Always train on the past and test on the future. Time-ordered splits are mandatory.
