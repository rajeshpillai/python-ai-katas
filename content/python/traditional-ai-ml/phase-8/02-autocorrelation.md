# Autocorrelation

> Phase 8 — Time Series & Sequential Data | Kata 8.2

---

## Concept & Intuition

### What problem are we solving?

Autocorrelation measures how much a time series is correlated with lagged versions of itself. If today's stock price is correlated with yesterday's price, the lag-1 autocorrelation is high. If weekly sales follow a pattern, the lag-7 autocorrelation will be high. Understanding autocorrelation is essential for two reasons: it reveals the temporal structure in your data, and it directly guides the choice of time series model (ARIMA, exponential smoothing, etc.).

There are two key tools. The **Autocorrelation Function (ACF)** measures the correlation between y_t and y_{t-k} for each lag k, including indirect effects through intermediate lags. The **Partial Autocorrelation Function (PACF)** measures the *direct* correlation between y_t and y_{t-k}, removing the effect of all intermediate lags. The distinction is critical: if lag-1 and lag-2 are both significant in the ACF, it could mean y_t directly depends on y_{t-2}, OR it could mean y_t depends on y_{t-1}, which depends on y_{t-2}. PACF distinguishes between these two scenarios.

Together, ACF and PACF plots are the diagnostic tools that tell you what kind of model to fit. A pure AR(p) process shows a cutoff at lag p in the PACF and gradual decay in the ACF. A pure MA(q) process shows a cutoff at lag q in the ACF and gradual decay in the PACF. These signatures are the fingerprints that identify the data-generating process.

### Why naive approaches fail

Looking at a raw time series plot and guessing the model structure is unreliable. A series might appear to have a 7-day cycle, but is that a direct dependence on 7 days ago, or a chain of daily dependencies that happens to create a weekly pattern? ACF and PACF answer this question quantitatively. Without them, you are guessing model orders, which leads to either underfitting (too few lags) or overfitting (too many lags).

### Mental models

- **ACF as total correlation**: like asking "how similar is today's weather to the weather k days ago?" This includes both direct effects and indirect chains.
- **PACF as direct causation**: like asking "how much does the weather k days ago directly affect today, after accounting for all the days in between?"
- **Fingerprinting**: different processes leave different signatures in ACF/PACF plots. Learning to read these signatures is like learning to identify species from their tracks.

### Visual explanations

```
ACF and PACF Signatures:

AR(1) process: y_t = 0.8 * y_{t-1} + noise
  ACF:   |||||||||  |||||||  |||||  |||  ||  |   (gradual decay)
  PACF:  |||||||||  |  .  .  .  .  .  .  .  .   (cuts off at lag 1)

MA(1) process: y_t = noise + 0.6 * noise_{t-1}
  ACF:   |||||||||  |  .  .  .  .  .  .  .  .   (cuts off at lag 1)
  PACF:  |||||||||  |||||||  |||||  |||  ||  |   (gradual decay)

AR(2) process: y_t = 0.5*y_{t-1} + 0.3*y_{t-2} + noise
  ACF:   |||||||||  |||||||  |||||  |||  ||  |   (gradual decay)
  PACF:  |||||||||  |||||  .  .  .  .  .  .  .   (cuts off at lag 2)

Seasonal (period=7):
  ACF:   ||||  |  |  ||||  |  |  ||||  |  |      (spikes every 7 lags)
  PACF:  ||||  |  |  ||||  .  .  .  .  .  .      (spikes at 7, then cuts off)

Blue shaded region = 95% confidence interval
Spikes outside this region are statistically significant
```

---

## Hands-on Exploration

1. Generate a pure AR(1) process and plot its ACF and PACF. Verify the theoretical signature: ACF decays gradually, PACF cuts off at lag 1.
2. Generate a pure MA(1) process and plot its ACF and PACF. Verify the opposite signature: ACF cuts off at lag 1, PACF decays gradually.
3. Generate a seasonal process (period=7) and observe the ACF spikes at multiples of 7.
4. Apply ACF/PACF analysis to a real dataset to identify the appropriate model order for ARIMA.

---

## Live Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(42)

# ============================================================
# Generate different types of processes
# ============================================================

n = 500

# --- AR(1): y_t = 0.8 * y_{t-1} + noise ---
ar1 = np.zeros(n)
for t in range(1, n):
    ar1[t] = 0.8 * ar1[t-1] + np.random.normal(0, 1)

# --- AR(2): y_t = 0.5 * y_{t-1} + 0.3 * y_{t-2} + noise ---
ar2 = np.zeros(n)
for t in range(2, n):
    ar2[t] = 0.5 * ar2[t-1] + 0.3 * ar2[t-2] + np.random.normal(0, 1)

# --- MA(1): y_t = noise + 0.6 * noise_{t-1} ---
noise = np.random.normal(0, 1, n)
ma1 = np.zeros(n)
for t in range(1, n):
    ma1[t] = noise[t] + 0.6 * noise[t-1]

# --- Seasonal (period=7) + AR(1) ---
seasonal_ar = np.zeros(n)
for t in range(7, n):
    seasonal_ar[t] = 0.5 * seasonal_ar[t-1] + 0.4 * seasonal_ar[t-7] + np.random.normal(0, 1)

# ============================================================
# Compute and display ACF/PACF values
# ============================================================
print("=== ACF/PACF Analysis ===\n")

processes = {
    "AR(1), phi=0.8": ar1,
    "AR(2), phi1=0.5, phi2=0.3": ar2,
    "MA(1), theta=0.6": ma1,
    "Seasonal AR (period=7)": seasonal_ar,
}

for name, series in processes.items():
    acf_vals = acf(series, nlags=15)
    pacf_vals = pacf(series, nlags=15)

    print(f"  {name}:")
    print(f"    ACF  (lags 1-10): ", end="")
    for v in acf_vals[1:11]:
        bar = "|" * int(abs(v) * 20)
        sign = "+" if v > 0 else "-"
        print(f"{sign}{bar:>8}", end=" ")
    print()
    print(f"    PACF (lags 1-10): ", end="")
    for v in pacf_vals[1:11]:
        bar = "|" * int(abs(v) * 20)
        sign = "+" if v > 0 else "-"
        print(f"{sign}{bar:>8}", end=" ")
    print("\n")

# ============================================================
# Identify model order from ACF/PACF
# ============================================================
print("=== Model Identification Rules ===\n")
print("  ACF cuts off at lag q, PACF decays --> MA(q)")
print("  ACF decays, PACF cuts off at lag p --> AR(p)")
print("  Both decay gradually              --> ARMA(p,q)")
print("  Spikes at seasonal lags           --> Seasonal component\n")

# Determine significant lags
for name, series in processes.items():
    n_obs = len(series)
    ci = 1.96 / np.sqrt(n_obs)  # 95% confidence interval

    acf_vals = acf(series, nlags=20)
    pacf_vals = pacf(series, nlags=20)

    sig_acf = [i for i in range(1, 21) if abs(acf_vals[i]) > ci]
    sig_pacf = [i for i in range(1, 21) if abs(pacf_vals[i]) > ci]

    print(f"  {name}:")
    print(f"    Significant ACF lags:  {sig_acf[:10]}")
    print(f"    Significant PACF lags: {sig_pacf[:10]}")
    print(f"    95% CI threshold: +/- {ci:.4f}\n")

# ============================================================
# Lag analysis: scatter plots
# ============================================================
print("=== Lag Scatter Analysis (AR(1)) ===\n")
for lag in [1, 2, 5]:
    corr = np.corrcoef(ar1[lag:], ar1[:-lag])[0, 1]
    print(f"  Lag {lag}: correlation = {corr:.4f}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(4, 3, figsize=(16, 16))

for row, (name, series) in enumerate(processes.items()):
    # Time series
    axes[row, 0].plot(series[:200], linewidth=0.8)
    axes[row, 0].set_title(f"{name}")
    axes[row, 0].set_xlabel("Time")

    # ACF
    plot_acf(series, lags=20, ax=axes[row, 1], title=f"ACF: {name}")

    # PACF
    plot_pacf(series, lags=20, ax=axes[row, 2], title=f"PACF: {name}",
              method="ywm")

plt.tight_layout()
plt.savefig("autocorrelation.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nPlot saved to autocorrelation.png")
```

---

## Key Takeaways

- **ACF measures total correlation at each lag; PACF measures direct correlation.** The difference reveals whether dependencies are direct or transmitted through intermediate lags.
- **ACF/PACF signatures identify the model type.** Gradual ACF decay + PACF cutoff at lag p = AR(p). ACF cutoff at lag q + gradual PACF decay = MA(q).
- **The 95% confidence band separates signal from noise.** Spikes within the band are not statistically significant. Only lags outside the band should be included in the model.
- **Seasonal patterns show up as spikes at multiples of the period.** A weekly pattern produces ACF spikes at lags 7, 14, 21, etc.
- **Always check ACF/PACF before fitting a model.** These plots are the time series equivalent of exploratory data analysis -- they tell you what structure exists before you try to model it.
