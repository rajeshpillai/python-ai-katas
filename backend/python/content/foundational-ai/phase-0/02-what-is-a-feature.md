# What Is a Feature?

> Phase 0 — Foundations (Before ML) | Kata 0.2

---

## Concept & Intuition

### What problem are we solving?

You have data — a table of observations. But which columns actually matter? If you're predicting house prices, does the owner's favorite movie matter? No. Does square footage? Absolutely. A **feature** is a measurable property you choose as input to a model.

### Why naive approaches fail

Including irrelevant features confuses models. Including redundant features wastes capacity. **Feature quality matters more than model complexity.**

### Mental models

Think of features as the **vocabulary** you give your model. If you can only use three words to describe a house, you'd pick "size," "location," "condition" — not "door color."

### Visual explanations

```
Correlation with price:
  sqft      ████████████ 0.95  ← strong signal
  bedrooms  ████████     0.78  ← useful
  age       ████         0.42  ← moderate
  fence_clr               0.02  ← noise
```

---

## Hands-on Exploration

1. Pick a prediction task — list 10 candidate features, rank by relevance
2. Compute correlation between each feature and the target
3. Try adding a random noise feature — observe near-zero correlation

---

## Live Code

```python
import numpy as np

# 8 houses: [sqft, bedrooms, age_years, distance_to_school]
features = np.array([
    [1400, 3, 15, 1.2],
    [1600, 3, 10, 0.8],
    [1700, 4, 25, 2.1],
    [1100, 2,  5, 0.5],
    [2100, 4, 30, 3.0],
    [1300, 2, 20, 1.8],
    [1800, 3,  8, 1.0],
    [2400, 5, 12, 0.3],
])
prices = np.array([250, 310, 340, 190, 420, 230, 350, 480])
names = ["sqft", "bedrooms", "age", "dist_school"]

print("Feature matrix shape:", features.shape)

# Inspect each feature
for i, name in enumerate(names):
    col = features[:, i]
    print(f"\n{name}: mean={col.mean():.1f}, std={col.std():.1f}, range=[{col.min()}, {col.max()}]")

# Which features correlate with price?
print("\n--- Correlation with price ---")
for i, name in enumerate(names):
    corr = np.corrcoef(features[:, i], prices)[0, 1]
    bar = "#" * int(abs(corr) * 20)
    print(f"  {name:>12}: {corr:+.3f}  {bar}")

# Bad feature: random noise
np.random.seed(42)
noise = np.random.randn(8)
print(f"\nRandom noise correlation: {np.corrcoef(noise, prices)[0, 1]:.3f} (meaningless)")
```

---

## Key Takeaways

- **Features are the inputs you choose.** They define what information the model sees.
- **Feature quality > model complexity.** Right features + simple model beats wrong features + complex model.
- **Correlation is a quick relevance check.** High |correlation| with target = likely useful.
- **More features isn't always better.** Irrelevant features add noise without value.
