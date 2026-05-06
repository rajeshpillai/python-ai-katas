# Visualizing Distributions

> Phase 0 — Foundations (Before ML) | Kata 0.5

---

## Concept & Intuition

### What problem are we solving?

You have a feature — say, heights of 1,000 people. What does it "look like"? A **distribution** describes how values are spread out. Understanding distributions is essential because ML algorithms make assumptions about data shape.

### Why naive approaches fail

Raw numbers tell you almost nothing. `[3.2, 5.1, 4.8, ...]` doesn't reveal structure. A histogram immediately shows shape, center, spread, and outliers.

### Mental models

| Statistic | Tells you | Intuition |
|-----------|-----------|-----------|
| Mean | Average | Balance point |
| Median | Middle value | Robust to outliers |
| Std dev | Spread | How far typical values are from the mean |

When mean ≈ median → symmetric. When they differ → skewed.

### Visual explanations

```
Normal:       Uniform:      Skewed:       Bimodal:
   ██                        ██            ██    ██
  ████        ████████       ████          ████  ████
 ██████       ████████       ██████       ██████████
████████      ████████       █████████   ████████████
```

---

## Hands-on Exploration

1. Generate normal, uniform, skewed distributions — compare histograms
2. Compare mean vs median for skewed data
3. See how statistics stabilize as sample size grows

---

## Live Code

```python
import numpy as np

np.random.seed(42)
n = 1000

normal = np.random.normal(loc=50, scale=10, size=n)
uniform = np.random.uniform(low=20, high=80, size=n)
skewed = np.random.exponential(scale=10, size=n) + 20

datasets = {"Normal": normal, "Uniform": uniform, "Skewed": skewed}

print(f"{'Name':<10} {'Mean':>8} {'Median':>8} {'Std':>8}")
print("=" * 40)
for name, data in datasets.items():
    print(f"{name:<10} {data.mean():>8.2f} {np.median(data):>8.2f} {data.std():>8.2f}")

# Text histogram
def text_hist(data, bins=12, width=35, title=""):
    counts, edges = np.histogram(data, bins=bins)
    mx = counts.max()
    print(f"\n  {title}")
    for i in range(len(counts)):
        bar = "#" * int(counts[i] / mx * width)
        print(f"  {edges[i]:>6.1f}-{edges[i+1]:>5.1f} | {bar}")

for name, data in datasets.items():
    text_hist(data, title=name)

# Skewness detection
print("\n--- Skewness ---")
for name, data in datasets.items():
    diff = data.mean() - np.median(data)
    label = "symmetric" if abs(diff) < 1 else ("right-skewed" if diff > 0 else "left-skewed")
    print(f"  {name}: mean - median = {diff:.2f} ({label})")

# 68-95 rule for normal distribution
within_1 = np.mean(np.abs(normal - normal.mean()) < normal.std()) * 100
within_2 = np.mean(np.abs(normal - normal.mean()) < 2 * normal.std()) * 100
print(f"\nNormal: {within_1:.0f}% within 1 std, {within_2:.0f}% within 2 std")
```

---

## Key Takeaways

- **Distributions describe data shape.** Always visualize before modeling.
- **Histograms reveal structure** that raw numbers hide.
- **Mean vs median detects skewness.**
- **68-95 rule**: for normal distributions, 68% within 1 std, 95% within 2 std.
- **More data = more stable statistics.**
