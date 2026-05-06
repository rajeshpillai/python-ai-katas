# What Is Data?

> Phase 0 — Foundations (Before ML) | Kata 0.1

---

## Concept & Intuition

### What problem are we solving?

Before we build any model, we need to understand the raw material: **data**. Data is a collection of structured observations about the world. Every time you measure something — temperature, a pixel color, a stock price — you create data. A single measurement is a **data point**. A collection is a **dataset**.

### Why naive approaches fail

People often jump straight to models without understanding their data. This leads to debugging models when the real problem is bad data — missing values, wrong types, or features that don't make sense. **80% of ML work is data work.**

### Mental models

- **Data as a spreadsheet**: rows are observations, columns are measurements
- **Data as coordinates**: each data point is a point in multi-dimensional space — a house with (sqft, bedrooms, price) is a point in 3D space

### Visual explanations

```
Spreadsheet view:           Geometric view (2D):

  sqft  beds  price            price
  1400   3    250k              |  *
  1600   3    310k              |    *
  1700   4    340k              |      *
  1100   2    190k              | *
  2100   4    420k              |         *
                               +------------- sqft
```

---

## Hands-on Exploration

1. Create a dataset by hand — measure 5 objects near you (weight, size, color)
2. Inspect shape and dtype of numpy arrays — these are the first things to check
3. Compare raw data (messy) vs processed data (clean, normalized)

---

## Live Code

```python
import numpy as np

# --- Creating data from scratch ---
temperatures = np.array([14.2, 16.8, 21.5, 24.1, 20.3, 15.7])

print("Dataset:", temperatures)
print("Shape:", temperatures.shape)
print("Dtype:", temperatures.dtype)
print("Observations:", len(temperatures))

# --- Multiple measurements per observation ---
# 5 houses: [square_feet, bedrooms, price_thousands]
houses = np.array([
    [1400, 3, 250],
    [1600, 3, 310],
    [1700, 4, 340],
    [1100, 2, 190],
    [2100, 4, 420],
])

print("\nHouse dataset:")
print(houses)
print("Shape:", houses.shape)  # (5, 3) = 5 observations, 3 features

# --- Accessing parts ---
print("\nFirst house:", houses[0])
print("All prices:", houses[:, 2])
print("Sqft of house 3:", houses[2, 0])

# --- Basic statistics ---
print("\nPrice statistics:")
print("  Mean:", houses[:, 2].mean())
print("  Min:", houses[:, 2].min())
print("  Max:", houses[:, 2].max())
print("  Std:", houses[:, 2].std())
```

---

## Key Takeaways

- **Data is structured observations.** Rows = data points, columns = features.
- **Always check shape and dtype first.** These tell you the structure instantly.
- **Raw data needs processing.** Real-world data is messy — cleaning it is the bulk of the work.
- **Inspect before modeling.** Print shape, first rows, and basic statistics before doing anything else.
