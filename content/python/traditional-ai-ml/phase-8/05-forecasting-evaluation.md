# Forecasting Evaluation

> Phase 8 — Time Series & Sequential Data | Kata 8.5

---

## Concept & Intuition

### What problem are we solving?

Evaluating a time series model is fundamentally different from evaluating a standard ML model. You cannot randomly shuffle data into train/test splits because temporal order matters -- future data must never leak into training. **Walk-forward validation** (also called rolling-origin evaluation) is the correct approach: train on data up to time t, forecast the next h steps, then slide the training window forward and repeat. This simulates how the model would actually be used in production.

Beyond the validation strategy, the choice of error metric matters enormously. **RMSE** (Root Mean Squared Error) penalizes large errors heavily because of the squaring. **MAE** (Mean Absolute Error) treats all errors equally regardless of size. **MAPE** (Mean Absolute Percentage Error) expresses error as a percentage, making it comparable across series with different scales. Each metric tells a different story, and the right choice depends on your business context: is a $100 error on a $10,000 forecast the same as a $100 error on a $200 forecast?

Understanding these metrics and validation strategies is critical because overly optimistic evaluations (from data leakage or inappropriate metrics) lead to models that fail in production. A model that looks great in backtesting but uses future information or reports only its best metric will disappoint when deployed.

### Why naive approaches fail

Standard k-fold cross-validation randomly shuffles data, which means a model might train on January and March data to predict February -- using the future to predict the past. This produces overly optimistic error estimates. Similarly, evaluating only on the last test period ignores how the model would have performed in different market conditions or seasons. Walk-forward validation tests the model across multiple time periods, giving a more robust estimate of future performance.

### Mental models

- **Walk-forward as simulated deployment**: each evaluation window simulates one month in production. You are literally measuring "if I had deployed this model on date X, how would it have performed?"
- **RMSE as worst-case focus**: because large errors are squared, RMSE is dominated by the biggest mistakes. Use it when large errors are costly.
- **MAPE as relative perspective**: a 10% error on a small value is the same as a 10% error on a large value. Use it when you care about proportional accuracy.
- **Multiple metrics as multiple lenses**: no single metric tells the whole story. Always report at least two (e.g., MAE + MAPE, or RMSE + MAE).

### Visual explanations

```
Walk-Forward Validation:

  Data: [===================================================================]

  Split 1: [TRAIN==================] [TEST===]
  Split 2: [TRAIN========================] [TEST===]
  Split 3: [TRAIN==============================] [TEST===]
  Split 4: [TRAIN====================================] [TEST===]

  Each split: train on all past data, test on next h steps
  Final score = average of all test scores

Standard CV (WRONG for time series):
  Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN]    <-- future in training!
  Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN]    <-- future in training!

Error Metrics:
  MAE  = mean(|actual - predicted|)              Scale-dependent
  RMSE = sqrt(mean((actual - predicted)^2))      Penalizes large errors
  MAPE = mean(|actual - predicted| / |actual|)   Scale-independent (%)

  Example:   actual = [100, 200, 150]
             pred   = [110, 220, 145]
             errors = [ 10,  20,   5]
  MAE  = (10+20+5)/3 = 11.67
  RMSE = sqrt((100+400+25)/3) = 13.23  (higher because 20 is squared)
  MAPE = (10%+10%+3.3%)/3 = 7.8%
```

---

## Hands-on Exploration

1. Generate a time series and split it using both standard k-fold CV and walk-forward validation. Compare the error estimates -- the k-fold estimate should be optimistically biased.
2. Compute MAE, RMSE, and MAPE for the same model. Discuss when each metric gives different guidance about model quality.
3. Implement walk-forward validation with expanding and sliding windows. Compare the two approaches.
4. Evaluate multiple models (naive baseline, SES, ARIMA) using walk-forward validation and declare a winner based on multiple metrics.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# --- Generate realistic sales data ---
n = 365 * 2
t = np.arange(n)
dates = pd.date_range("2022-01-01", periods=n, freq="D")

trend = 0.03 * t + 200
season = 30 * np.sin(2 * np.pi * t / 365.25)
weekly = 10 * np.sin(2 * np.pi * t / 7)
noise = np.random.normal(0, 8, n)
y = trend + season + weekly + noise
y = np.maximum(y, 10)  # sales can't be negative

ts = pd.Series(y, index=dates, name="sales")

print(f"=== Dataset: {len(ts)} daily sales observations ===\n")

# ============================================================
# Error Metrics
# ============================================================
def compute_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    errors = actual - predicted
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / np.abs(actual)) * 100  # percentage

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ============================================================
# Walk-Forward Validation
# ============================================================
print("=== Walk-Forward Validation ===\n")

# Parameters
n_splits = 5
test_size = 60  # forecast horizon: 60 days
min_train = 200

tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

# --- Define forecasting models ---
def naive_forecast(train, h):
    """Last value repeated."""
    return np.full(h, train.iloc[-1])

def seasonal_naive(train, h, period=7):
    """Last season repeated."""
    last_season = train.iloc[-period:].values
    return np.tile(last_season, h // period + 1)[:h]

def ses_forecast(train, h):
    """Simple Exponential Smoothing."""
    model = SimpleExpSmoothing(train).fit(optimized=True)
    return model.forecast(h).values

def hw_forecast(train, h):
    """Holt-Winters with weekly seasonality."""
    try:
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=7
        ).fit(optimized=True)
        return model.forecast(h).values
    except Exception:
        return ses_forecast(train, h)

models = {
    "Naive (last value)": naive_forecast,
    "Seasonal Naive (7d)": seasonal_naive,
    "SES": ses_forecast,
    "Holt-Winters": hw_forecast,
}

# --- Run walk-forward validation ---
all_metrics = {name: {"MAE": [], "RMSE": [], "MAPE": []} for name in models}

for fold, (train_idx, test_idx) in enumerate(tscv.split(ts)):
    train = ts.iloc[train_idx]
    test = ts.iloc[test_idx]
    h = len(test)

    print(f"  Fold {fold+1}: train={len(train)} days "
          f"({train.index[0].date()} to {train.index[-1].date()}), "
          f"test={len(test)} days")

    for name, model_fn in models.items():
        pred = model_fn(train, h)
        metrics = compute_metrics(test.values, pred)
        for metric_name, value in metrics.items():
            all_metrics[name][metric_name].append(value)

print()

# --- Average metrics across folds ---
print("=== Results (averaged across folds) ===\n")
print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
print(f"  {'-'*48}")

summary = {}
for name in models:
    avg_mae = np.mean(all_metrics[name]["MAE"])
    avg_rmse = np.mean(all_metrics[name]["RMSE"])
    avg_mape = np.mean(all_metrics[name]["MAPE"])
    summary[name] = {"MAE": avg_mae, "RMSE": avg_rmse, "MAPE": avg_mape}
    print(f"  {name:<22} {avg_mae:>8.2f} {avg_rmse:>8.2f} {avg_mape:>7.1f}%")

# Find winner for each metric
print()
for metric in ["MAE", "RMSE", "MAPE"]:
    best = min(summary, key=lambda x: summary[x][metric])
    print(f"  Best {metric}: {best} ({summary[best][metric]:.2f})")

# ============================================================
# Compare CV vs Walk-Forward (show leakage)
# ============================================================
print("\n=== Data Leakage Demonstration ===\n")
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# Create lag features
df = pd.DataFrame({"y": ts.values})
for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = df["y"].shift(lag)
df = df.dropna()

X = df.drop("y", axis=1).values
y_target = df["y"].values

# Standard KFold (WRONG)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = []
for train_idx, test_idx in kfold.split(X):
    model = Ridge()
    model.fit(X[train_idx], y_target[train_idx])
    pred = model.predict(X[test_idx])
    kfold_scores.append(np.sqrt(np.mean((y_target[test_idx] - pred) ** 2)))

# TimeSeriesSplit (CORRECT)
ts_scores = []
for train_idx, test_idx in TimeSeriesSplit(n_splits=5).split(X):
    model = Ridge()
    model.fit(X[train_idx], y_target[train_idx])
    pred = model.predict(X[test_idx])
    ts_scores.append(np.sqrt(np.mean((y_target[test_idx] - pred) ** 2)))

print(f"  K-Fold RMSE:           {np.mean(kfold_scores):.2f} (optimistically biased)")
print(f"  TimeSeriesSplit RMSE:  {np.mean(ts_scores):.2f} (honest estimate)")
print(f"  Bias: {(1 - np.mean(kfold_scores)/np.mean(ts_scores))*100:.1f}% "
      f"(K-Fold underestimates error)")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Walk-forward visualization
train_end = ts.iloc[:500]
test_end = ts.iloc[500:560]
axes[0, 0].plot(train_end.index[-100:], train_end[-100:], 'b-',
                linewidth=0.8, label="Train")
axes[0, 0].plot(test_end.index, test_end, 'g-', linewidth=1.5, label="Actual")
for name, model_fn in models.items():
    pred = model_fn(train_end, len(test_end))
    axes[0, 0].plot(test_end.index, pred, '--', linewidth=1.2, label=name)
axes[0, 0].set_title("Walk-Forward: One Fold")
axes[0, 0].legend(fontsize=7)

# Metric comparison
x = np.arange(len(models))
width = 0.25
names = list(models.keys())
axes[0, 1].bar(x - width, [summary[n]["MAE"] for n in names],
               width, label="MAE", color="steelblue")
axes[0, 1].bar(x, [summary[n]["RMSE"] for n in names],
               width, label="RMSE", color="coral")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
axes[0, 1].set_title("MAE vs RMSE by Model")
axes[0, 1].legend()

# MAPE comparison
axes[1, 0].barh(names, [summary[n]["MAPE"] for n in names], color="steelblue")
axes[1, 0].set_xlabel("MAPE (%)")
axes[1, 0].set_title("MAPE by Model (scale-independent)")
axes[1, 0].invert_yaxis()

# CV vs TimeSeriesSplit
axes[1, 1].bar(["K-Fold CV\n(WRONG)", "TimeSeriesSplit\n(CORRECT)"],
               [np.mean(kfold_scores), np.mean(ts_scores)],
               color=["salmon", "steelblue"])
axes[1, 1].set_ylabel("RMSE")
axes[1, 1].set_title("Data Leakage: K-Fold vs Time Series Split")

plt.tight_layout()
plt.savefig("forecasting_evaluation.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to forecasting_evaluation.png")
```

---

## Key Takeaways

- **Walk-forward validation is the only correct evaluation strategy for time series.** Train on the past, test on the future, slide forward, repeat. Never shuffle temporal data.
- **RMSE penalizes large errors; MAE treats all errors equally; MAPE gives percentages.** Choose based on your business context: is a big mistake more costly (use RMSE) or do you want scale-independent comparison (use MAPE)?
- **K-fold cross-validation produces optimistically biased estimates for time series.** Future data leaks into training through shuffling, making the model look better than it will actually perform.
- **Always compare against naive baselines.** If your sophisticated model cannot beat "repeat last value" or "repeat last week," it is adding complexity without value.
- **Evaluate across multiple time periods.** A model that works well in summer may fail in winter. Walk-forward validation with enough splits tests the model across different conditions.
