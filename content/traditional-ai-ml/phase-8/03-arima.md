# ARIMA

> Phase 8 — Time Series & Sequential Data | Kata 8.3

---

## Concept & Intuition

### What problem are we solving?

ARIMA (AutoRegressive Integrated Moving Average) is the workhorse model for univariate time series forecasting. It combines three components, each controlled by an order parameter. **AR(p)** uses p past values of the series to predict the current value -- it captures momentum and persistence. **I(d)** applies d rounds of differencing to make the series stationary -- it handles trends. **MA(q)** uses q past forecast errors to correct the current prediction -- it captures short-term shocks that echo forward.

The full model ARIMA(p, d, q) works as follows: first, difference the series d times to remove trend. Then model the differenced series as a linear combination of p past values (AR part) and q past errors (MA part). The result is a flexible framework that can handle a wide range of time series patterns by choosing the right (p, d, q) orders.

Choosing the orders is the key challenge. The differencing order d is determined by stationarity testing (ADF test): difference until stationary. The AR order p is read from the PACF (cutoff lag). The MA order q is read from the ACF (cutoff lag). Alternatively, you can use information criteria (AIC, BIC) to automatically search over combinations and select the best one.

### Why naive approaches fail

Using a simple average or linear regression ignores the temporal dependencies that ARIMA explicitly models. A linear regression of y on time captures the trend but misses the autoregressive structure (yesterday's value predicts today's). Conversely, just using yesterday's value (random walk) captures AR(1) but misses the trend and the MA error-correction component. ARIMA combines all three mechanisms into a single coherent framework.

### Mental models

- **AR as momentum**: if a stock went up yesterday, it is slightly more likely to go up today. The AR component captures this persistence.
- **I as detrending**: if sales are growing by 100 units per month, differencing converts the series from "cumulative sales" to "monthly change" -- which is much easier to model.
- **MA as shock absorption**: if an unexpected event (a viral tweet, a weather event) pushed sales up yesterday, the MA component models how that shock dissipates over the next few days.
- **ARIMA(p,d,q) as a recipe**: "use p days of history, difference d times, and correct for q past errors."

### Visual explanations

```
ARIMA(p, d, q) Components:

  AR(p): y_t = c + phi_1*y_{t-1} + phi_2*y_{t-2} + ... + phi_p*y_{t-p} + e_t
         "Current value depends on past values"

  I(d):  y'_t = y_t - y_{t-1}  (d=1)
         y''_t = y'_t - y'_{t-1}  (d=2)
         "Difference until stationary"

  MA(q): y_t = c + e_t + theta_1*e_{t-1} + ... + theta_q*e_{t-q}
         "Current value depends on past errors"

Order Selection Guide:
  d: ADF test (difference until p-value < 0.05)
  p: PACF cutoff lag (after differencing)
  q: ACF cutoff lag (after differencing)

  Or: use auto_arima / AIC minimization

Common Models:
  ARIMA(1,0,0) = AR(1)           (simple autoregression)
  ARIMA(0,1,0) = Random walk     (tomorrow = today + noise)
  ARIMA(0,1,1) = Exponential smoothing (approximately)
  ARIMA(1,1,1) = Common starting point for trended data
```

---

## Hands-on Exploration

1. Generate a time series with a known ARIMA structure (e.g., ARIMA(1,1,1)) and fit an ARIMA model to recover the true parameters.
2. Use the ACF/PACF-based approach to determine the model order: test for stationarity, difference if needed, then read p and q from the plots.
3. Use AIC-based model selection to automatically find the best (p,d,q) and compare it to your manual selection.
4. Generate forecasts with confidence intervals and evaluate forecast accuracy on a held-out test period.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# --- Generate ARIMA(1,1,1) process ---
n = 300
noise = np.random.normal(0, 1, n)

# ARIMA(1,1,1): y_t = y_{t-1} + phi*(y_{t-1} - y_{t-2}) + theta*e_{t-1} + e_t
phi, theta = 0.6, 0.4
y = np.zeros(n)
e = noise.copy()
for t in range(2, n):
    y[t] = y[t-1] + phi * (y[t-1] - y[t-2]) + theta * e[t-1] + e[t]

dates = pd.date_range("2022-01-01", periods=n, freq="D")
ts = pd.Series(y, index=dates)

# Split into train/test
train_size = 250
ts_train = ts[:train_size]
ts_test = ts[train_size:]

print("=== Generated ARIMA(1,1,1) Process ===")
print(f"  True parameters: phi={phi}, theta={theta}")
print(f"  Train: {len(ts_train)} obs, Test: {len(ts_test)} obs\n")

# --- Step 1: Test for stationarity ---
print("=== Step 1: Stationarity Testing ===\n")
adf_original = adfuller(ts_train, autolag="AIC")
print(f"  Original series:    ADF stat={adf_original[0]:.4f}, "
      f"p-value={adf_original[1]:.6f} "
      f"{'(stationary)' if adf_original[1] < 0.05 else '(NON-stationary)'}")

ts_diff = ts_train.diff().dropna()
adf_diff = adfuller(ts_diff, autolag="AIC")
print(f"  First difference:   ADF stat={adf_diff[0]:.4f}, "
      f"p-value={adf_diff[1]:.6f} "
      f"{'(stationary)' if adf_diff[1] < 0.05 else '(NON-stationary)'}")
print(f"  --> d = 1\n")

# --- Step 2: Read p and q from PACF/ACF of differenced series ---
print("=== Step 2: ACF/PACF of Differenced Series ===\n")
ci = 1.96 / np.sqrt(len(ts_diff))
acf_vals = acf(ts_diff, nlags=15)
pacf_vals = pacf(ts_diff, nlags=15, method="ywm")

sig_acf = [i for i in range(1, 16) if abs(acf_vals[i]) > ci]
sig_pacf = [i for i in range(1, 16) if abs(pacf_vals[i]) > ci]
print(f"  Significant ACF lags:  {sig_acf[:5]}")
print(f"  Significant PACF lags: {sig_pacf[:5]}")
print(f"  --> Suggested: p=1, q=1 (or try p=1, q=0 and p=0, q=1)\n")

# --- Step 3: Fit multiple ARIMA models and compare AIC ---
print("=== Step 3: Model Selection (AIC) ===\n")
orders_to_try = [
    (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
    (2, 1, 0), (0, 1, 2), (2, 1, 1), (1, 1, 2), (2, 1, 2)
]

results = []
for order in orders_to_try:
    try:
        model = ARIMA(ts_train, order=order)
        fitted = model.fit()
        results.append({
            "order": order,
            "aic": fitted.aic,
            "bic": fitted.bic,
        })
    except Exception:
        pass

results = sorted(results, key=lambda x: x["aic"])
print(f"  {'Order':<15} {'AIC':>10} {'BIC':>10}")
print(f"  {'-'*37}")
for r in results:
    marker = " <-- best" if r == results[0] else ""
    print(f"  {str(r['order']):<15} {r['aic']:>10.2f} {r['bic']:>10.2f}{marker}")

best_order = results[0]["order"]
print(f"\n  Best model: ARIMA{best_order}\n")

# --- Step 4: Fit best model and forecast ---
print("=== Step 4: Fit and Forecast ===\n")
model = ARIMA(ts_train, order=best_order)
fitted = model.fit()

print(f"  Fitted ARIMA{best_order}:")
print(f"    AR coefficients: {fitted.arparams}")
print(f"    MA coefficients: {fitted.maparams}")
print(f"    (True: phi={phi}, theta={theta})\n")

# Forecast
forecast = fitted.get_forecast(steps=len(ts_test))
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int(alpha=0.05)

# --- Evaluate ---
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(ts_test, pred_mean))
mae = mean_absolute_error(ts_test, pred_mean)

print(f"  Forecast Evaluation ({len(ts_test)} steps ahead):")
print(f"    RMSE: {rmse:.4f}")
print(f"    MAE:  {mae:.4f}\n")

# --- Residual diagnostics ---
residuals = fitted.resid
print("=== Residual Diagnostics ===")
print(f"  Mean: {residuals.mean():.4f} (should be ~0)")
print(f"  Std:  {residuals.std():.4f}")
print(f"  Skew: {residuals.skew():.4f} (should be ~0)")

# Ljung-Box test on residuals
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(f"  Ljung-Box p-values (should be > 0.05 = no autocorrelation):")
for lag, row in lb_test.iterrows():
    status = "OK" if row["lb_pvalue"] > 0.05 else "FAIL"
    print(f"    Lag {lag:>2}: p={row['lb_pvalue']:.4f} ({status})")

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original series + forecast
axes[0, 0].plot(ts_train.index, ts_train, 'b-', linewidth=0.8, label="Train")
axes[0, 0].plot(ts_test.index, ts_test, 'g-', linewidth=0.8, label="Test (actual)")
axes[0, 0].plot(pred_mean.index, pred_mean, 'r--', linewidth=1.5, label="Forecast")
axes[0, 0].fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                         alpha=0.2, color='r', label="95% CI")
axes[0, 0].set_title(f"ARIMA{best_order} Forecast")
axes[0, 0].legend(fontsize=8)

# ACF of differenced series
plot_acf(ts_diff, lags=15, ax=axes[0, 1], title="ACF of Differenced Series")

# PACF of differenced series
plot_pacf(ts_diff, lags=15, ax=axes[1, 0], title="PACF of Differenced Series",
          method="ywm")

# Residual plot
axes[1, 1].plot(residuals.index, residuals, linewidth=0.5)
axes[1, 1].axhline(0, color='r', linewidth=1)
axes[1, 1].set_title("Model Residuals")
axes[1, 1].set_xlabel("Date")

plt.tight_layout()
plt.savefig("arima.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to arima.png")
```

---

## Key Takeaways

- **ARIMA combines autoregression (p), differencing (d), and moving average (q).** Each component handles a different aspect of the time series: persistence, trend, and shocks.
- **Order selection follows a systematic process.** Test stationarity to find d, then read p from PACF and q from ACF of the differenced series. Or use AIC to search automatically.
- **AIC balances fit quality and model complexity.** Lower AIC is better. It penalizes models with too many parameters, preventing overfitting.
- **Residual diagnostics validate the model.** If residuals have no autocorrelation (Ljung-Box test passes), the model has captured all the temporal structure.
- **Forecast uncertainty grows with horizon.** The confidence interval widens for longer forecasts because errors accumulate. Short-term forecasts are always more reliable.
