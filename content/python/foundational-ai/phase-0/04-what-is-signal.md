# What Is Signal?

> Phase 0 — Foundations (Before ML) | Kata 0.4

---

## Concept & Intuition

### What problem are we solving?

If noise is unwanted variation, **signal** is the meaningful pattern. Signal is the consistent, repeatable relationship between inputs and outputs. The entire goal of ML is to separate signal from noise.

### Why naive approaches fail

A model with 99% training accuracy might have found signal — or memorized noise. You need to test on **unseen data** to tell the difference. Training error alone is misleading.

### Mental models

- **Gold panning**: dirt/water = noise, gold nuggets = signal. Your model is the sieve.
- **Trend line in a scatter plot**: the trend is signal, the scatter around it is noise.

### Visual explanations

```
Fit complexity vs signal recovery:
  Degree 1 (line):     underfits — misses the curve
  Degree 2 (parabola): just right — captures the signal
  Degree 15 (wiggly):  overfits — memorizes noise
```

---

## Hands-on Exploration

1. Generate data with a known quadratic signal + noise
2. Fit polynomials of degree 1, 2, and 15 — see which recovers the signal
3. Test on new data — overfitting is revealed

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# True signal: a parabola
x = np.linspace(-3, 3, 30)
true_signal = 0.5 * x**2 - x + 2
noise = np.random.normal(0, 0.8, 30)
observed = true_signal + noise

# Fit polynomials of different degrees
print("--- Recovering signal with polynomial fits ---")
for degree in [1, 2, 5, 15]:
    coeffs = np.polyfit(x, observed, degree)
    fitted = np.polyval(coeffs, x)
    signal_err = np.sqrt(((fitted - true_signal) ** 2).mean())
    data_err = np.sqrt(((fitted - observed) ** 2).mean())
    print(f"\n  Degree {degree:>2d}:")
    print(f"    Fit to noisy data: {data_err:.3f}")
    print(f"    Recovery of signal: {signal_err:.3f}")

# Test on NEW data
print("\n--- Test on new data ---")
x_new = np.linspace(-3, 3, 50)
true_new = 0.5 * x_new**2 - x_new + 2
obs_new = true_new + np.random.normal(0, 0.8, 50)

for degree in [2, 15]:
    coeffs = np.polyfit(x, observed, degree)
    pred = np.polyval(coeffs, x_new)
    err = np.sqrt(((pred - obs_new) ** 2).mean())
    print(f"  Degree {degree:>2d} on new data: RMSE = {err:.3f}")

print("\n  Degree 2 generalizes (captured signal)")
print("  Degree 15 fails on new data (captured noise)")
```

---

## Key Takeaways

- **Signal is the consistent pattern** you want your model to learn.
- **Test on new data** to verify you captured signal, not noise.
- **Simpler models capture signal.** Complex models memorize noise.
- **Signal strength determines learnability.** Weak signals need more data.
