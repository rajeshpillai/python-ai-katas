# A/B Testing for Models

> Phase 11 — Productionizing ML | Kata 11.4

---

## Concept & Intuition

### What problem are we solving?

Your new model has better offline metrics (accuracy, AUC, F1) than the current production model. Should you deploy it? Not so fast. Offline metrics are computed on historical data, which may not reflect real-world performance. User behavior, feedback loops, and distribution shifts can all cause a model that looked great in testing to underperform in production. **A/B testing** is the gold standard for comparing models in the real world.

In an A/B test, you split live traffic between the current model (control, version A) and the new model (treatment, version B). You measure a **business metric** (click-through rate, revenue, churn rate) over a sufficient period and apply a **statistical test** to determine whether the difference is real or due to random chance. Only if the new model shows a statistically significant improvement do you roll it out to all users.

The key challenge is getting the statistics right. You need enough samples for **statistical power** (the ability to detect a real difference), you need to account for **multiple comparisons** (testing many metrics inflates false positive rates), and you need to run the test long enough — stopping early when results look good leads to inflated false positive rates (the "peeking problem").

### Why naive approaches fail

Deploying based on offline metrics alone ignores the gap between offline and online performance. Deploying without statistical rigor — "it looks better after a day" — leads to adopting models that are not actually better (false positives) or rejecting models that are (false negatives). The human temptation to peek at results and stop early when they look good is especially dangerous: it dramatically inflates the false positive rate beyond the nominal 5%.

### Mental models

- **A/B testing as a clinical trial for models**: Patients (users) are randomly assigned to treatment (new model) or control (old model). The outcome (metric) is measured. Statistics determine if the treatment worked.
- **P-value as surprise**: "If the models were equally good, how surprising would this result be?" A p-value below 0.05 means "quite surprising — probably a real difference."
- **Effect size matters more than p-value**: A statistically significant 0.01% improvement is real but probably not worth the engineering effort. Always consider practical significance alongside statistical significance.
- **Power is about not missing real improvements**: A test with low power (small sample) might fail to detect a 5% improvement. You need enough data to catch the effects you care about.

### Visual explanations

```
A/B Testing Setup:

  Live Traffic
      |
      +--[50%]--> Model A (current)  --> measure metric
      |
      +--[50%]--> Model B (new)      --> measure metric

  Statistical Test:
    H0: metric_A == metric_B  (no difference)
    H1: metric_A != metric_B  (there is a difference)

    If p-value < 0.05: reject H0, deploy B (if B is better)
    If p-value >= 0.05: keep A (insufficient evidence for B)

  Sample Size Calculation:
    Given:
      - Baseline conversion rate: 5%
      - Minimum detectable effect: 10% relative (0.5% absolute)
      - Significance level (alpha): 0.05
      - Power (1-beta): 0.80
    Required: ~30,000 users per group

  The peeking problem:
    Day 1: p=0.03  --> "Significant! Ship it!" (WRONG)
    Day 2: p=0.12  --> (would have been fine if you waited)
    Day 7: p=0.08  --> Still not significant
    Day 14: p=0.04 --> Truly significant now (proper conclusion)

    Checking daily at alpha=0.05 inflates actual false positive rate to ~25%!
```

---

## Hands-on Exploration

1. Simulate two models with identical conversion rates. Run an A/B test. How often does the test incorrectly declare a winner (should be ~5% at alpha=0.05)?
2. Simulate model B being 10% better than A. What sample size do you need to detect this with 80% power?
3. Simulate the peeking problem: check p-values daily for 14 days. How often do you see a "significant" result even when the models are identical?
4. Try a Bayesian A/B test instead. Compare the interpretation ("95% probability B is better") with the frequentist interpretation ("if they were equal, this result would be surprising").

---

## Live Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# --- A/B Test Simulation ---
# Simulating conversion rates for two models

# @param base_rate float 0.01 0.20 0.05
base_rate = 0.05       # Model A (control) conversion rate
# @param effect_size float 0.0 0.5 0.10
effect_size = 0.10     # Relative improvement of Model B (0.10 = 10% better)
# @param n_per_group int 500 50000 5000
n_per_group = 5000     # Users per group
alpha = 0.05           # Significance level

treatment_rate = base_rate * (1 + effect_size)

print(f"=== A/B Testing for Models ===")
print(f"Model A (control) rate:   {base_rate:.4f}")
print(f"Model B (treatment) rate: {treatment_rate:.4f}")
print(f"Relative effect:          {effect_size*100:.1f}%")
print(f"Absolute effect:          {(treatment_rate - base_rate)*100:.2f}%")
print(f"Users per group:          {n_per_group}")
print(f"Significance level:       {alpha}\n")

# --- Run the A/B test ---
conversions_A = np.random.binomial(1, base_rate, n_per_group)
conversions_B = np.random.binomial(1, treatment_rate, n_per_group)

rate_A = np.mean(conversions_A)
rate_B = np.mean(conversions_B)

# Two-proportion z-test
n_A = len(conversions_A)
n_B = len(conversions_B)
p_pooled = (np.sum(conversions_A) + np.sum(conversions_B)) / (n_A + n_B)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))
z_stat = (rate_B - rate_A) / se if se > 0 else 0
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"=== Test Results ===")
print(f"Model A conversion rate: {rate_A:.4f} ({np.sum(conversions_A)} / {n_A})")
print(f"Model B conversion rate: {rate_B:.4f} ({np.sum(conversions_B)} / {n_B})")
print(f"Observed lift:           {((rate_B/rate_A - 1)*100) if rate_A > 0 else 0:.2f}%")
print(f"Z-statistic:             {z_stat:.4f}")
print(f"P-value:                 {p_value:.6f}")
print(f"Significant (p < {alpha})? {'YES' if p_value < alpha else 'NO'}")

# Confidence interval for the difference
se_diff = np.sqrt(rate_A*(1-rate_A)/n_A + rate_B*(1-rate_B)/n_B)
ci_low = (rate_B - rate_A) - 1.96 * se_diff
ci_high = (rate_B - rate_A) + 1.96 * se_diff
print(f"95% CI for difference:   [{ci_low:.4f}, {ci_high:.4f}]")

# --- Power Analysis ---
print(f"\n=== Sample Size / Power Analysis ===")
from scipy.stats import norm as normal_dist

def required_sample_size(p1, p2, alpha=0.05, power=0.80):
    """Compute required sample size per group for two-proportion test."""
    z_alpha = normal_dist.ppf(1 - alpha / 2)
    z_beta = normal_dist.ppf(power)
    p_bar = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
    return int(np.ceil(n))

print(f"{'Effect Size':>12}  {'Treatment Rate':>15}  {'Required n/group':>17}  {'Total Users':>12}")
print("-" * 62)
for es in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    p2 = base_rate * (1 + es)
    if p2 > 0 and p2 < 1 and abs(base_rate - p2) > 1e-10:
        n_req = required_sample_size(base_rate, p2)
        print(f"{es*100:>11.0f}%  {p2:>15.4f}  {n_req:>17,}  {2*n_req:>12,}")

# --- The Peeking Problem ---
print(f"\n=== The Peeking Problem ===")
print("Simulating 1000 A/A tests (both groups identical) with daily peeking\n")

n_simulations = 1000
n_days = 14
users_per_day = 500
false_positive_no_peek = 0
false_positive_with_peek = 0

for sim in range(n_simulations):
    # Both groups have the SAME rate (this is an A/A test)
    all_A = np.random.binomial(1, base_rate, n_days * users_per_day)
    all_B = np.random.binomial(1, base_rate, n_days * users_per_day)

    # Proper test: look only at final result
    final_rate_A = np.mean(all_A)
    final_rate_B = np.mean(all_B)
    n_final = n_days * users_per_day
    p_pool = (np.sum(all_A) + np.sum(all_B)) / (2 * n_final)
    se_final = np.sqrt(p_pool * (1 - p_pool) * (2 / n_final)) if p_pool > 0 else 1
    z_final = (final_rate_B - final_rate_A) / se_final if se_final > 0 else 0
    p_final = 2 * (1 - stats.norm.cdf(abs(z_final)))
    if p_final < alpha:
        false_positive_no_peek += 1

    # Peeking: check every day, stop if significant
    peeked_significant = False
    for day in range(1, n_days + 1):
        n_so_far = day * users_per_day
        peek_A = all_A[:n_so_far]
        peek_B = all_B[:n_so_far]
        r_A = np.mean(peek_A)
        r_B = np.mean(peek_B)
        p_pool = (np.sum(peek_A) + np.sum(peek_B)) / (2 * n_so_far)
        se_p = np.sqrt(p_pool * (1 - p_pool) * (2 / n_so_far)) if p_pool > 0 else 1
        z_p = (r_B - r_A) / se_p if se_p > 0 else 0
        p_p = 2 * (1 - stats.norm.cdf(abs(z_p)))
        if p_p < alpha:
            peeked_significant = True
            break
    if peeked_significant:
        false_positive_with_peek += 1

fp_rate_proper = false_positive_no_peek / n_simulations
fp_rate_peeking = false_positive_with_peek / n_simulations

print(f"False positive rate (proper test, end only): {fp_rate_proper:.3f} (should be ~{alpha})")
print(f"False positive rate (peeking daily):         {fp_rate_peeking:.3f} (inflated!)")
print(f"Peeking inflates false positives by {fp_rate_peeking/max(fp_rate_proper, 0.001):.1f}x")

# --- Bayesian A/B Testing ---
print(f"\n=== Bayesian A/B Test ===")
# Using Beta-Binomial model
successes_A = np.sum(conversions_A)
failures_A = n_A - successes_A
successes_B = np.sum(conversions_B)
failures_B = n_B - successes_B

# Posterior distributions: Beta(successes + 1, failures + 1)
n_mc = 100000
posterior_A = np.random.beta(successes_A + 1, failures_A + 1, n_mc)
posterior_B = np.random.beta(successes_B + 1, failures_B + 1, n_mc)

prob_B_better = np.mean(posterior_B > posterior_A)
expected_lift = np.mean((posterior_B - posterior_A) / posterior_A)
lift_samples = (posterior_B - posterior_A) / posterior_A

print(f"P(Model B is better): {prob_B_better:.4f}")
print(f"Expected relative lift: {expected_lift*100:.2f}%")
print(f"95% credible interval for lift: [{np.percentile(lift_samples, 2.5)*100:.2f}%, "
      f"{np.percentile(lift_samples, 97.5)*100:.2f}%]")

# --- Decision Summary ---
print(f"\n=== Decision Summary ===")
if p_value < alpha and rate_B > rate_A:
    print(f"Frequentist: DEPLOY Model B (p={p_value:.4f} < {alpha})")
elif p_value < alpha and rate_B < rate_A:
    print(f"Frequentist: KEEP Model A (B is significantly worse)")
else:
    print(f"Frequentist: KEEP Model A (insufficient evidence, p={p_value:.4f})")

if prob_B_better > 0.95:
    print(f"Bayesian:    DEPLOY Model B ({prob_B_better:.1%} probability of being better)")
elif prob_B_better < 0.05:
    print(f"Bayesian:    KEEP Model A (B is likely worse)")
else:
    print(f"Bayesian:    INCONCLUSIVE ({prob_B_better:.1%} probability B is better, need more data)")
```

---

## Key Takeaways

- **A/B testing is the gold standard for comparing models in production.** Offline metrics are necessary but not sufficient — real-world performance can differ significantly.
- **Statistical significance protects against false positives.** Without proper testing, you might deploy a model that is not actually better, just luckier in a small sample.
- **Sample size determines what effects you can detect.** Small effects require large samples. Always compute the required sample size before starting a test.
- **Peeking at results inflates false positive rates dramatically.** Decide the test duration in advance and commit to it. If you must peek, use sequential testing methods that account for multiple looks.
- **Bayesian A/B tests provide more intuitive results.** "95% probability B is better" is easier to act on than "p < 0.05." Both approaches are valid; choose based on your team's comfort and the decision context.
