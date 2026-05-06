# Bayesian Thinking

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.1

---

## Concept & Intuition

### What problem are we solving?

Classical (frequentist) statistics treats model parameters as fixed but unknown numbers and makes inferences through repeated sampling. **Bayesian thinking** flips the script: parameters are treated as **random variables** with probability distributions. Before seeing data, we have a **prior** distribution expressing our beliefs. After observing data, we update those beliefs using **Bayes' theorem** to get a **posterior** distribution. The **likelihood** function connects data to parameters, telling us how probable the observed data is under different parameter values.

This framework is powerful because it naturally handles uncertainty. Instead of a single point estimate ("the coin is 60% heads"), you get a full distribution ("the coin is probably between 55% and 65% heads, with most of the probability around 60%"). This is especially valuable when data is scarce — the prior acts as a regularizer, pulling estimates toward reasonable values and preventing overfitting.

**Conjugate priors** are special prior distributions that, when combined with a particular likelihood, produce a posterior of the same family. For example, a Beta prior with a Binomial likelihood gives a Beta posterior. This makes the math clean and updates simple — just adjust the parameters of the distribution. Conjugacy is not required for Bayesian inference, but it makes everything analytically tractable.

### Why naive approaches fail

A frequentist point estimate from a small sample can be wildly wrong. If you flip a coin 3 times and get 3 heads, the maximum likelihood estimate is P(heads) = 1.0 — clearly an overreaction. A Bayesian approach with a reasonable prior (say, Beta(2, 2) centered at 0.5) would give a posterior of Beta(5, 2), with a mean around 0.71 — still shifted toward heads, but much more sensible. The prior prevents the estimate from going to extremes when data is limited.

### Mental models

- **Prior = what you knew before seeing data.** Posterior = what you know after. Likelihood = how the data updates your knowledge.
- **Bayes' theorem as a learning rule**: posterior is proportional to prior times likelihood. More data shifts the posterior away from the prior and toward the data.
- **The posterior becomes the new prior**: Bayesian updating is incremental. Yesterday's posterior is today's prior. This is how beliefs evolve with new evidence.
- **Conjugate priors as matching puzzle pieces**: Beta-Binomial, Normal-Normal, Gamma-Poisson — each pair fits together perfectly, making updates a simple arithmetic operation.

### Visual explanations

```
Bayes' Theorem:

  P(theta | data) = P(data | theta) * P(theta) / P(data)
  ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^   ^^^^^^^^^
  posterior          likelihood          prior

  Or more simply:
  posterior ∝ likelihood x prior

Example: Coin flip with Beta-Binomial conjugacy

  Prior:      Beta(2, 2)     -- mild belief: coin is fair-ish
  Data:       7 heads, 3 tails
  Likelihood: Binomial(10, theta)
  Posterior:  Beta(2+7, 2+3) = Beta(9, 5)

  Prior mean:     2/(2+2)   = 0.50
  MLE:            7/10      = 0.70
  Posterior mean: 9/(9+5)   = 0.64  (compromise between prior and data)

  As data grows, posterior concentrates:
  After 10 flips:   Beta(9, 5)       -- wide, uncertain
  After 100 flips:  Beta(72, 32)     -- narrower
  After 1000 flips: Beta(702, 302)   -- very concentrated near true value
```

---

## Hands-on Exploration

1. Start with a Beta(1, 1) prior (uniform). Flip a biased coin 10 times. Compute the posterior. Plot prior and posterior on the same axes.
2. Repeat with a strong prior Beta(50, 50) (very confident the coin is fair). How much does 10 flips shift the posterior?
3. Gradually add more data (10, 50, 200 flips). Watch the posterior narrow and converge to the true probability regardless of the prior.
4. Try a misspecified prior — e.g., Beta(1, 10) (strongly believes coin favors tails) with a coin that actually favors heads. How quickly does the data overcome the wrong prior?

---

## Live Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# --- Bayesian Coin Flip with Beta-Binomial Conjugacy ---

# True coin bias (unknown to the agent)
true_p = 0.65

# @param prior_alpha float 0.5 50.0 2.0
prior_alpha = 2.0  # Beta prior parameter alpha (pseudo-counts for heads)
# @param prior_beta float 0.5 50.0 2.0
prior_beta = 2.0   # Beta prior parameter beta (pseudo-counts for tails)
# @param n_flips int 5 500 50
n_flips = 50

# Generate coin flip data
flips = np.random.binomial(1, true_p, size=n_flips)
n_heads = np.sum(flips)
n_tails = n_flips - n_heads

# --- Prior ---
print("=== Bayesian Coin Flip ===")
print(f"True P(heads): {true_p}")
print(f"Data: {n_heads} heads, {n_tails} tails out of {n_flips} flips\n")

print(f"Prior:     Beta({prior_alpha:.1f}, {prior_beta:.1f})")
prior_mean = prior_alpha / (prior_alpha + prior_beta)
prior_var = (prior_alpha * prior_beta) / ((prior_alpha + prior_beta)**2 * (prior_alpha + prior_beta + 1))
print(f"  Mean: {prior_mean:.4f}, Std: {np.sqrt(prior_var):.4f}")

# --- Likelihood (MLE) ---
mle = n_heads / n_flips
print(f"\nMLE:       {mle:.4f}")

# --- Posterior (conjugate update) ---
post_alpha = prior_alpha + n_heads
post_beta = prior_beta + n_tails
post_mean = post_alpha / (post_alpha + post_beta)
post_var = (post_alpha * post_beta) / ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))

print(f"\nPosterior: Beta({post_alpha:.1f}, {post_beta:.1f})")
print(f"  Mean: {post_mean:.4f}, Std: {np.sqrt(post_var):.4f}")

# 95% credible interval
ci_low = stats.beta.ppf(0.025, post_alpha, post_beta)
ci_high = stats.beta.ppf(0.975, post_alpha, post_beta)
print(f"  95% Credible Interval: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  True value {true_p} in CI: {ci_low <= true_p <= ci_high}")

# --- Sequential Bayesian Updating ---
print("\n=== Sequential Bayesian Updates ===")
print(f"{'Flips Seen':>11}  {'Heads':>6}  {'Post Mean':>10}  {'Post Std':>9}  {'95% CI':>20}")
print("-" * 65)

a, b = prior_alpha, prior_beta
checkpoints = [1, 5, 10, 20, 50, 100, 200, 500]
checkpoints = [cp for cp in checkpoints if cp <= n_flips]

for i in range(n_flips):
    if flips[i] == 1:
        a += 1
    else:
        b += 1
    if (i + 1) in checkpoints:
        mean = a / (a + b)
        std = np.sqrt((a * b) / ((a + b)**2 * (a + b + 1)))
        lo = stats.beta.ppf(0.025, a, b)
        hi = stats.beta.ppf(0.975, a, b)
        heads_so_far = int(np.sum(flips[:i+1]))
        print(f"{i+1:>11}  {heads_so_far:>6}  {mean:>10.4f}  {std:>9.4f}  [{lo:.4f}, {hi:.4f}]")

# --- Compare different priors ---
print("\n=== Effect of Prior Strength ===")
priors = [
    ("Weak: Beta(1,1)", 1, 1),
    ("Moderate: Beta(5,5)", 5, 5),
    ("Strong: Beta(50,50)", 50, 50),
    ("Wrong: Beta(2,10)", 2, 10),
]
print(f"Data: {n_heads} heads, {n_tails} tails\n")
print(f"{'Prior':>25}  {'Prior Mean':>11}  {'Post Mean':>10}  {'Post Std':>9}")
print("-" * 60)
for name, a0, b0 in priors:
    pa = a0 + n_heads
    pb = b0 + n_tails
    pm = pa / (pa + pb)
    ps = np.sqrt((pa * pb) / ((pa + pb)**2 * (pa + pb + 1)))
    print(f"{name:>25}  {a0/(a0+b0):>11.4f}  {pm:>10.4f}  {ps:>9.4f}")

print(f"\nTrue value: {true_p}")
print("Notice: With enough data, all priors converge to a similar posterior.")
print("Strong priors require more data to overcome.")

# --- Posterior predictive ---
print("\n=== Posterior Predictive ===")
# Probability of getting heads on the next flip
# Under Beta-Binomial, P(next=H | data) = posterior mean
print(f"P(next flip = heads | data) = {post_mean:.4f}")
print(f"This is simply the posterior mean of theta.")

# Predict next 10 flips
n_predict = 10
# Beta-Binomial predictive distribution
from scipy.special import comb, beta as beta_fn
print(f"\nPredictive distribution for {n_predict} future flips:")
print(f"{'k heads':>8}  {'P(k heads)':>11}")
for k in range(n_predict + 1):
    # Beta-Binomial PMF
    p = comb(n_predict, k) * beta_fn(post_alpha + k, post_beta + n_predict - k) / beta_fn(post_alpha, post_beta)
    if p > 0.01:
        print(f"{k:>8}  {p:>11.4f}  {'*' * int(p * 50)}")
```

---

## Key Takeaways

- **Bayesian inference produces distributions, not point estimates.** This naturally quantifies uncertainty — critical when data is scarce or decisions are high-stakes.
- **The prior encodes domain knowledge.** A good prior regularizes estimates; a bad prior can be overcome with enough data.
- **Conjugate priors make updates trivial.** Beta-Binomial, Normal-Normal, and other conjugate pairs reduce Bayesian updating to simple arithmetic.
- **As data grows, the posterior concentrates around the truth.** The prior becomes irrelevant — this is the Bayesian consistency guarantee.
- **The posterior mean is a weighted compromise between the prior mean and the MLE.** The weights depend on the relative "strength" (effective sample size) of the prior vs. the data.
