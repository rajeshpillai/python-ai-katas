# Datetime Features

> Phase 7 — Feature Engineering & Pipelines | Kata 7.4

---

## Concept & Intuition

### What problem are we solving?

Datetime columns are among the most information-rich features in any dataset, but models cannot use a raw timestamp directly. The string "2024-03-15 14:30:00" must be transformed into numerical features that capture the temporal patterns within it. A single timestamp can yield dozens of useful features: hour of day (captures daily patterns), day of week (weekday vs weekend effects), month (seasonal patterns), and more.

Beyond simple extraction, there are three advanced techniques. **Cyclical encoding** handles the fact that hour 23 is close to hour 0, but a raw integer encoding puts them far apart. By encoding cyclical features as sine/cosine pairs, we preserve this circular relationship. **Lag features** use past values of a target variable to predict the current value (yesterday's sales predict today's sales). **Rolling statistics** compute moving averages, moving standard deviations, and other aggregations over a sliding window, capturing trends and volatility.

These techniques are essential for any time-stamped dataset: sales forecasting, web traffic prediction, sensor monitoring, financial modeling, and more. The difference between a model with raw timestamps and one with properly engineered datetime features is often dramatic.

### Why naive approaches fail

Feeding a raw timestamp as a single integer (Unix epoch) tells the model nothing about daily or weekly patterns -- the number 1710510600 carries no hint that it is a Friday afternoon. Even extracting "hour = 14" as an integer creates a problem: the model thinks hour 23 and hour 0 are far apart (difference of 23), when they are actually adjacent. And without lag features, the model has no memory of recent history -- it cannot know that "sales have been trending up for the past 3 days."

### Mental models

- **Clock hands as coordinates**: encode cyclical features using the position on a circle. Hour 0 and hour 23 are both near the top of the clock, which sine/cosine encoding captures naturally.
- **Lag features as short-term memory**: the model sees the current features plus what happened 1, 2, 7 days ago. This gives it context about recent trends.
- **Rolling windows as smoothing**: a 7-day moving average smooths out daily noise, revealing the underlying weekly trend.

### Visual explanations

```
Timestamp: 2024-03-15 14:30:00 (Friday)

Extracted Features:
  hour = 14          day_of_week = 4 (Friday)
  month = 3          day_of_month = 15
  quarter = 1        is_weekend = 0
  year = 2024        day_of_year = 75

Cyclical Encoding (hour):
  hour_sin = sin(2*pi*14/24) = -0.866
  hour_cos = cos(2*pi*14/24) = -0.500

  Why? Because:            23
                       22 /    \ 0/24
                      21 |      | 1
                      20 |      | 2
                       19 \    / 3
                          18--6
  Hours 23 and 0 are neighbors on the circle!

Lag Features (for sales):
  today       = 150
  lag_1       = 142    (yesterday)
  lag_7       = 155    (same day last week)
  rolling_7   = 148.3  (7-day average)
  rolling_std = 8.2    (7-day std dev)
```

---

## Hands-on Exploration

1. Generate a synthetic daily sales dataset with known weekly and seasonal patterns. Extract basic datetime features (day_of_week, month, hour) and show they improve prediction.
2. Implement cyclical encoding for hour and day_of_week. Compare model performance with integer encoding vs sine/cosine encoding.
3. Create lag features (1-day, 7-day) and rolling statistics (7-day mean, 7-day std). Show how they capture trends and seasonality.
4. Combine all datetime features and compare the full-featured model against a baseline with just the raw date.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# --- Generate synthetic daily sales with patterns ---
n_days = 365 * 2  # 2 years
dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

# Weekly pattern (weekends lower) + seasonal pattern + trend
day_of_week_effect = np.array([0, 5, 10, 15, 20, -10, -15])
weekly = np.array([day_of_week_effect[d.weekday()] for d in dates])
seasonal = 30 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # yearly cycle
trend = 0.05 * np.arange(n_days)  # slight upward trend
noise = np.random.normal(0, 8, n_days)

sales = 200 + weekly + seasonal + trend + noise
df = pd.DataFrame({"date": dates, "sales": sales})

print("=== Dataset ===")
print(f"  {len(df)} days of sales data")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Sales range: {df['sales'].min():.0f} to {df['sales'].max():.0f}\n")

# --- Feature Engineering ---

# Basic datetime features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["quarter"] = df["date"].dt.quarter

# Cyclical encoding
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# Lag features
for lag in [1, 2, 3, 7, 14, 28]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)

# Rolling statistics
for window in [7, 14, 30]:
    df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
    df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

# Drop rows with NaN from lag/rolling features
df_clean = df.dropna().reset_index(drop=True)
print(f"  After feature engineering: {len(df_clean)} rows, {len(df_clean.columns)} columns\n")

# --- Define feature sets ---
basic_features = ["day_of_week", "month", "day_of_year", "is_weekend", "quarter"]
cyclical_features = ["dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "is_weekend"]
lag_features = [f"lag_{l}" for l in [1, 2, 3, 7, 14, 28]]
rolling_features = [f"rolling_mean_{w}" for w in [7, 14, 30]] + \
                   [f"rolling_std_{w}" for w in [7, 14, 30]]
all_features = cyclical_features + lag_features + rolling_features

# --- Time Series Cross-Validation ---
tscv = TimeSeriesSplit(n_splits=5)
target = df_clean["sales"].values

print("=== Model Comparison (TimeSeriesSplit CV) ===\n")
feature_sets = {
    "Basic (integer)": basic_features,
    "Cyclical encoding": cyclical_features,
    "Lag features only": lag_features,
    "Rolling stats only": rolling_features,
    "All features": all_features,
}

results = {}
for name, features in feature_sets.items():
    X = df_clean[features].values
    maes = []
    for train_idx, test_idx in tscv.split(X):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], target[train_idx])
        pred = model.predict(X[test_idx])
        maes.append(mean_absolute_error(target[test_idx], pred))
    results[name] = np.mean(maes)
    print(f"  {name:<25}: MAE = {np.mean(maes):.2f}")

# --- Show cyclical encoding advantage ---
print("\n=== Cyclical vs Integer Encoding (day_of_week) ===\n")
print("  Integer: day 6 (Sat) to day 0 (Mon) distance = 6 (WRONG)")
print("  Cyclical: sin/cos naturally wraps around")
print()
for d in range(7):
    day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d]
    s = np.sin(2 * np.pi * d / 7)
    c = np.cos(2 * np.pi * d / 7)
    print(f"    {day_name}: sin={s:+.3f}, cos={c:+.3f}")

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sales time series
axes[0, 0].plot(df["date"], df["sales"], linewidth=0.5, alpha=0.7)
axes[0, 0].plot(df["date"], 200 + seasonal + trend, 'r-', linewidth=1.5,
                label="Trend + Season")
axes[0, 0].set_title("Daily Sales with Trend & Seasonality")
axes[0, 0].legend()

# Day-of-week pattern
dow_means = df.groupby("day_of_week")["sales"].mean()
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[0, 1].bar(day_labels, dow_means.values, color="steelblue")
axes[0, 1].set_title("Average Sales by Day of Week")
axes[0, 1].set_ylabel("Mean Sales")

# Cyclical encoding visualization
theta = np.linspace(0, 2 * np.pi, 100)
axes[1, 0].plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.2)
for d in range(7):
    s = np.sin(2 * np.pi * d / 7)
    c = np.cos(2 * np.pi * d / 7)
    axes[1, 0].plot(c, s, 'o', markersize=10)
    axes[1, 0].annotate(day_labels[d], (c, s), textcoords="offset points",
                        xytext=(10, 5), fontsize=9)
axes[1, 0].set_title("Cyclical Encoding (Day of Week)")
axes[1, 0].set_aspect("equal")
axes[1, 0].grid(True, alpha=0.3)

# Method comparison
names = list(results.keys())
maes = [results[n] for n in names]
colors = ["coral" if n == "All features" else "steelblue" for n in names]
axes[1, 1].barh(names, maes, color=colors)
axes[1, 1].set_xlabel("MAE (lower is better)")
axes[1, 1].set_title("Feature Set Comparison")
axes[1, 1].invert_yaxis()
axes[1, 1].invert_xaxis()

plt.tight_layout()
plt.savefig("datetime_features.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to datetime_features.png")
```

---

## Key Takeaways

- **A single timestamp can yield dozens of useful features.** Hour, day of week, month, quarter, is_weekend, day_of_year -- each captures a different temporal pattern.
- **Cyclical encoding (sine/cosine) preserves circular relationships.** Hour 23 and hour 0 are neighbors, and so are December and January. Integer encoding misses this.
- **Lag features give the model memory.** Yesterday's value, last week's value, and last month's value provide context about recent history.
- **Rolling statistics capture trends and volatility.** A 7-day moving average smooths daily noise; a rolling standard deviation signals periods of high or low variability.
- **Use TimeSeriesSplit for temporal data.** Standard k-fold cross-validation leaks future information into the training set. Time series CV always trains on the past and tests on the future.
