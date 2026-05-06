# Distance-Based Prediction

> Phase 1 — What Does It Mean to Learn? | Kata 1.3

---

## Concept & Intuition

### What problem are we solving?

The mean predictor ignores all features -- it predicts the same value whether a house is 800 sqft or 3000 sqft. Clearly, similar houses should have similar prices. **Distance-based prediction** formalizes this intuition: to predict for a new house, find the most similar houses in your data and use their prices.

This is the k-Nearest Neighbors (k-NN) algorithm. For 1-NN, you find the single closest data point and copy its value. For k-NN, you find the k closest points and average their values. The key insight is that **proximity in feature space implies similarity in outcome** -- houses with similar square footage, bedrooms, and age tend to have similar prices.

This is our first model that actually uses input features to make different predictions for different inputs. It requires no training -- no parameters to fit, no equations to solve. It just remembers all the data and looks up neighbors at prediction time. This simplicity is both its strength and its weakness.

### Why naive approaches fail

The obvious challenge is **defining "similar."** If one feature is in thousands (sqft) and another is in single digits (bedrooms), raw Euclidean distance is dominated by sqft. A 100 sqft difference matters far less than a 100 bedroom difference, but raw distance treats them equally. Feature scaling is essential. Choosing k also matters -- too small and predictions are noisy, too large and you're back to predicting the mean.

### Mental models

- **Asking neighbors for restaurant advice**: you ask the 3 people most like you (similar taste, budget, dietary needs) and average their recommendations.
- **Interpolation on a map**: to estimate the temperature at an unmeasured location, average the temperatures of the nearest weather stations.

### Visual explanations

```
1-NN: predict the value of the SINGLE nearest neighbor

  sqft:  1100  1300  1400  1600  1800  2100  2400
  price:  190   230   250   310   350   420   480

  Query: sqft=1500 --> nearest is 1400 (dist=100) or 1600 (dist=100)
         --> predict 250 or 310 (depends on tie-breaking)

3-NN: average the 3 nearest neighbors
  Query: sqft=1500 --> neighbors: 1400(250), 1600(310), 1300(230)
         --> predict (250 + 310 + 230) / 3 = 263.3
```

---

## Hands-on Exploration

1. Implement 1-NN prediction using only numpy distance calculations
2. Compare 1-NN vs 3-NN vs 5-NN -- see how k affects smoothness
3. Observe what happens when k equals the dataset size (it becomes the mean predictor!)

---

## Live Code

```python
import numpy as np

np.random.seed(42)

# --- Dataset: houses with sqft and price ---
sqft_train = np.array([1100, 1300, 1400, 1600, 1700, 1800, 2100, 2400])
price_train = np.array([190, 230, 250, 310, 340, 350, 420, 480])

# --- k-NN predictor ---
def knn_predict(x_train, y_train, x_query, k):
    predictions = []
    for q in x_query:
        distances = np.abs(x_train - q)
        nearest_idx = np.argsort(distances)[:k]
        predictions.append(y_train[nearest_idx].mean())
    return np.array(predictions)

# --- Test queries ---
queries = np.array([1200, 1500, 1950, 2200])
print(f"Training data:  sqft={sqft_train}")
print(f"Training prices: {price_train}\n")

print(f"{'Query':>8}  {'1-NN':>8}  {'3-NN':>8}  {'5-NN':>8}  {'All(mean)':>9}")
print("=" * 50)
for k_val in [1, 3, 5, len(sqft_train)]:
    preds = knn_predict(sqft_train, price_train, queries, k_val)
    if k_val == 1:
        row_data = {1: preds}
    elif k_val == 3:
        row_data[3] = preds
    elif k_val == 5:
        row_data[5] = preds
    else:
        row_data["all"] = preds

for i, q in enumerate(queries):
    print(f"{q:>8}  {row_data[1][i]:>8.1f}  {row_data[3][i]:>8.1f}  "
          f"{row_data[5][i]:>8.1f}  {row_data['all'][i]:>9.1f}")

# --- Leave-one-out evaluation ---
print("\n--- Leave-one-out error for different k ---")
for k in [1, 3, 5]:
    errors = []
    for i in range(len(sqft_train)):
        x_loo = np.delete(sqft_train, i)
        y_loo = np.delete(price_train, i)
        pred = knn_predict(x_loo, y_loo, sqft_train[i:i+1], k)[0]
        errors.append((price_train[i] - pred) ** 2)
    rmse = np.sqrt(np.mean(errors))
    bar = "#" * int(rmse / 3)
    print(f"  k={k}: RMSE = {rmse:>6.1f}k  {bar}")

# --- k = N degenerates to mean predictor ---
mean_pred = knn_predict(sqft_train, price_train, queries, len(sqft_train))
print(f"\nk=N predictions: {mean_pred}")
print(f"Mean of prices:  {price_train.mean():.1f} (same for all queries!)")
```

---

## Key Takeaways

- **Similar inputs should produce similar outputs.** This is the core assumption of distance-based methods.
- **1-NN copies the nearest neighbor.** Simple but noisy -- sensitive to individual data points.
- **Larger k means smoother predictions.** Averaging more neighbors reduces variance but may miss local patterns.
- **k = N gives the mean predictor.** k-NN spans from memorization (k=1) to the baseline (k=N).
- **Feature scaling matters.** Raw distance is meaningless when features have different units and ranges.
