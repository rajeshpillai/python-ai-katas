# What Is Noise?

> Phase 0 — Foundations (Before ML) | Kata 0.3

---

## Concept & Intuition

### What problem are we solving?

Real-world data is never clean. Every measurement has **noise** — unwanted variation that obscures the true pattern. If a model learns the noise instead of the signal, it will fail on new data. This is overfitting.

### Why naive approaches fail

Without understanding noise, you might think a model that fits every point perfectly is the best. In reality, it memorized noise. A simpler model that ignores noise generalizes better.

### Mental models

- **Radio static**: music = signal, static = noise. A good tuner captures music and filters static.
- **Blurry photo**: the scene is signal, blur is noise. Over-sharpening creates artifacts (overfitting).

### Visual explanations

```
Clean signal:    + Noise:        = Observed data:
  ────────         ∼∼∼∼∼∼          ∼─∼──∼─∼∼─
  (true pattern)   (random)        (what we actually see)
```

---

## Hands-on Exploration

1. Generate a clean linear trend, add noise — see how the pattern gets obscured
2. Vary noise level — at what point is the signal unrecoverable?
3. Average multiple noisy samples — watch noise cancel out

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# True signal: y = 2x + 5
x = np.linspace(0, 10, 20)
true_signal = 2 * x + 5

# Add different noise levels
noise_low = np.random.normal(0, 1, 20)
noise_high = np.random.normal(0, 8, 20)

print("True signal (first 5):", true_signal[:5].round(2))
print("Low noise   (first 5):", (true_signal + noise_low)[:5].round(2))
print("High noise  (first 5):", (true_signal + noise_high)[:5].round(2))

# Signal-to-Noise Ratio
for name, noise in [("Low", noise_low), ("High", noise_high)]:
    snr = true_signal.std() / np.abs(noise).std()
    print(f"\n{name} noise: SNR = {snr:.2f} (higher = cleaner)")

# Averaging reduces noise
print("\n--- Averaging reduces noise ---")
for n_avg in [1, 5, 20, 100]:
    samples = np.array([true_signal + np.random.normal(0, 3, 20) for _ in range(n_avg)])
    averaged = samples.mean(axis=0)
    rmse = np.sqrt(((averaged - true_signal) ** 2).mean())
    print(f"  Average of {n_avg:>3d} samples: RMSE = {rmse:.3f}")
```

---

## Key Takeaways

- **Noise is unwanted variation** that obscures the true pattern.
- **All real data contains noise.** The question is how much.
- **SNR (Signal-to-Noise Ratio)** quantifies data quality.
- **Averaging reduces noise** — larger datasets are better.
- **A model that fits noise is overfitting.** It looks perfect on training data but fails on new data.
