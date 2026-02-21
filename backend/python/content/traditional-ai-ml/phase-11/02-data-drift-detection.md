# Data Drift Detection

> Phase 11 — Productionizing ML | Kata 11.2

---

## Concept & Intuition

### What problem are we solving?

A model trained on historical data makes an implicit assumption: **future data will look like past data**. When this assumption breaks — when the statistical properties of incoming data shift — the model's performance degrades silently. This is **data drift**, and it is one of the most common reasons ML systems fail in production.

There are several types of drift. **Covariate drift** occurs when the input distribution P(X) changes (e.g., your customer demographics shift). **Concept drift** occurs when the relationship P(Y|X) changes (e.g., what makes a customer churn changes over time). **Prior probability drift** occurs when P(Y) changes (e.g., fraud becomes more common). All require different detection and mitigation strategies.

Detecting drift early is critical. Without monitoring, a model can silently decay for months before someone notices the business metrics dropping. By then, you have lost revenue, trust, or worse. A good drift detection system raises alerts when input distributions shift significantly, triggering investigation, retraining, or model rollback.

### Why naive approaches fail

Simply monitoring prediction accuracy is not enough — in many production systems, ground truth labels arrive with a delay (sometimes weeks or months). You need to detect drift from the **inputs alone**, before you know whether predictions are wrong. Statistical tests comparing reference distributions (from training data) to production distributions catch shifts early, even without labels.

### Mental models

- **Drift as concept aging**: A model is like a map. The map was accurate when drawn, but the terrain changes over time. Drift detection tells you when the map is outdated.
- **Two-sample testing**: "Are these two batches of data from the same distribution?" This is the core statistical question. If the answer is "no," drift has occurred.
- **Training distribution as the reference**: Everything the model learned came from the training data. Any significant deviation from that distribution is a potential problem.

### Visual explanations

```
Types of drift:

  Covariate drift (P(X) changes):
    Training:   age ~ N(35, 10)     income ~ N(50k, 15k)
    Production: age ~ N(45, 12)     income ~ N(70k, 20k)
    --> Model inputs look different, predictions may be unreliable

  Concept drift (P(Y|X) changes):
    Training:   high spending --> loyal customer
    Production: high spending --> customer about to churn (economy changed)
    --> Same inputs, different correct outputs

  Label drift (P(Y) changes):
    Training:   5% fraud rate
    Production: 15% fraud rate (new attack vector)
    --> Class balance shifted

Detection methods:
  Statistical tests:
    - KS test (Kolmogorov-Smirnov): compares CDFs, works per-feature
    - PSI (Population Stability Index): binned distribution comparison
    - Chi-squared: for categorical features
    - MMD (Maximum Mean Discrepancy): multivariate, kernel-based

  Monitoring pipeline:
    Training data --> compute reference statistics
                         |
    Production data --> compute window statistics --> compare --> ALERT?
                         |
                     [sliding window or batch]
```

---

## Hands-on Exploration

1. Generate two datasets with the same distribution. Run a KS test — it should not detect drift.
2. Shift the mean of one feature by 1 standard deviation. Run the KS test again — does it detect the shift?
3. Compute PSI for a gradually drifting feature. At what point does PSI exceed the alert threshold (0.2)?
4. Simulate concept drift: keep X the same but change the decision boundary. Can you detect this from X alone? (Hint: you cannot — this is why monitoring predictions and outcomes matters too.)

---

## Live Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# --- Generate reference (training) and production data ---
n_ref = 1000    # reference sample (from training)
n_prod = 500    # production sample (new data)
n_features = 5
feature_names = ['age', 'income', 'tenure', 'usage', 'support_calls']

# Reference distribution
ref_means = [35, 50000, 24, 150, 2]
ref_stds = [10, 15000, 12, 50, 1.5]
X_ref = np.column_stack([
    np.random.normal(m, s, n_ref) for m, s in zip(ref_means, ref_stds)
])

# @param drift_magnitude float 0.0 3.0 0.5
drift_magnitude = 0.5  # how many std devs to shift (0 = no drift)

# Production distribution (with drift in some features)
drift_features = [0, 1]  # age and income drift
prod_means = ref_means.copy()
prod_stds = ref_stds.copy()
for f in drift_features:
    prod_means[f] += drift_magnitude * ref_stds[f]

X_prod = np.column_stack([
    np.random.normal(m, s, n_prod) for m, s in zip(prod_means, prod_stds)
])

print(f"=== Data Drift Detection ===")
print(f"Reference samples: {n_ref}, Production samples: {n_prod}")
print(f"Drift magnitude: {drift_magnitude} std devs in features: {[feature_names[f] for f in drift_features]}\n")

# --- Method 1: Kolmogorov-Smirnov Test (per feature) ---
print("=== KS Test (per feature) ===")
print(f"{'Feature':>15}  {'KS Stat':>8}  {'p-value':>10}  {'Drift?':>8}  {'Ref Mean':>10}  {'Prod Mean':>10}")
print("-" * 70)

ks_results = {}
for i, name in enumerate(feature_names):
    ks_stat, p_value = stats.ks_2samp(X_ref[:, i], X_prod[:, i])
    drift_detected = p_value < 0.05
    ks_results[name] = {'stat': ks_stat, 'p_value': p_value, 'drift': drift_detected}
    marker = "YES" if drift_detected else "no"
    print(f"{name:>15}  {ks_stat:>8.4f}  {p_value:>10.6f}  {marker:>8}  "
          f"{np.mean(X_ref[:, i]):>10.1f}  {np.mean(X_prod[:, i]):>10.1f}")

# --- Method 2: Population Stability Index (PSI) ---
def compute_psi(reference, production, n_bins=10):
    """Compute PSI between reference and production distributions."""
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Compute proportions in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    prod_counts = np.histogram(production, bins=breakpoints)[0]

    # Add small constant to avoid division by zero
    ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
    prod_pct = (prod_counts + 1) / (len(production) + n_bins)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return psi

print(f"\n=== Population Stability Index (PSI) ===")
print(f"Thresholds: PSI < 0.1 = stable, 0.1-0.2 = moderate shift, > 0.2 = significant drift\n")
print(f"{'Feature':>15}  {'PSI':>8}  {'Status':>15}")
print("-" * 42)

for i, name in enumerate(feature_names):
    psi = compute_psi(X_ref[:, i], X_prod[:, i])
    if psi < 0.1:
        status = "Stable"
    elif psi < 0.2:
        status = "Moderate shift"
    else:
        status = "SIGNIFICANT DRIFT"
    print(f"{name:>15}  {psi:>8.4f}  {status:>15}")

# --- Method 3: Summary Statistics Monitoring ---
print(f"\n=== Summary Statistics Comparison ===")
print(f"{'Feature':>15}  {'Metric':>10}  {'Reference':>12}  {'Production':>12}  {'% Change':>10}")
print("-" * 62)

for i, name in enumerate(feature_names):
    for metric_name, metric_fn in [('Mean', np.mean), ('Std', np.std),
                                     ('Median', np.median)]:
        ref_val = metric_fn(X_ref[:, i])
        prod_val = metric_fn(X_prod[:, i])
        pct_change = 100 * (prod_val - ref_val) / (abs(ref_val) + 1e-10)
        flag = " !" if abs(pct_change) > 10 else ""
        print(f"{name:>15}  {metric_name:>10}  {ref_val:>12.2f}  {prod_val:>12.2f}  {pct_change:>9.1f}%{flag}")

# --- Simulate gradual drift over time ---
print(f"\n=== Gradual Drift Simulation ===")
print(f"Monitoring 'age' feature over 10 time windows\n")

window_size = 200
n_windows = 10
print(f"{'Window':>8}  {'Mean':>8}  {'KS p-val':>10}  {'PSI':>8}  {'Alert':>8}")
print("-" * 48)

for w in range(n_windows):
    # Gradually increasing drift
    window_drift = drift_magnitude * (w / (n_windows - 1))
    window_mean = ref_means[0] + window_drift * ref_stds[0]
    X_window = np.random.normal(window_mean, ref_stds[0], window_size)

    ks_stat, ks_p = stats.ks_2samp(X_ref[:, 0], X_window)
    psi = compute_psi(X_ref[:, 0], X_window)

    alert = "ALERT" if ks_p < 0.05 or psi > 0.2 else ""
    print(f"{w+1:>8}  {np.mean(X_window):>8.1f}  {ks_p:>10.6f}  {psi:>8.4f}  {alert:>8}")

# --- Multivariate drift detection ---
print(f"\n=== Multivariate Drift (feature correlations) ===")
ref_corr = np.corrcoef(X_ref.T)
prod_corr = np.corrcoef(X_prod.T)
corr_diff = np.abs(ref_corr - prod_corr)

print("Correlation matrix difference (|ref - prod|):")
print(f"{'':>15}", end="")
for name in feature_names:
    print(f"  {name[:6]:>8}", end="")
print()
for i, name in enumerate(feature_names):
    print(f"{name:>15}", end="")
    for j in range(n_features):
        val = corr_diff[i, j]
        marker = "*" if val > 0.1 else " "
        print(f"  {val:>7.3f}{marker}", end="")
    print()
print("(* = correlation shift > 0.1)")

# --- Summary and recommendations ---
print(f"\n=== Drift Detection Summary ===")
n_drifted_ks = sum(1 for r in ks_results.values() if r['drift'])
print(f"Features with detected drift (KS test): {n_drifted_ks}/{n_features}")
if n_drifted_ks > 0:
    drifted = [name for name, r in ks_results.items() if r['drift']]
    print(f"  Drifted features: {drifted}")
    print(f"\nRecommended actions:")
    print(f"  1. Investigate root cause of drift in {drifted}")
    print(f"  2. Evaluate model performance on recent labeled data")
    print(f"  3. Consider retraining model with recent data")
    print(f"  4. Set up automated alerts for PSI > 0.2")
else:
    print(f"No significant drift detected. Model inputs appear stable.")
```

---

## Key Takeaways

- **Data drift is the silent killer of ML models.** Performance can degrade for months before anyone notices, because drift happens in the inputs, not in a visible error message.
- **The KS test and PSI are complementary detection methods.** KS is a formal statistical test with p-values; PSI gives an interpretable stability score. Use both.
- **Monitor individual features AND multivariate relationships.** Two features can individually look fine while their correlation changes dramatically.
- **You often must detect drift without labels.** Ground truth may be delayed by weeks or months. Input-based drift detection gives early warning before you can measure accuracy.
- **Drift detection triggers action, not panic.** Not all drift degrades model performance. The workflow is: detect, investigate, evaluate impact, then retrain if needed.
