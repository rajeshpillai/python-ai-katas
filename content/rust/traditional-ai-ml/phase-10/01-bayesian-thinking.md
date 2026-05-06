# Bayesian Thinking

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.1

---

## Concept & Intuition

### What problem are we solving?

Classical (frequentist) statistics treats model parameters as fixed but unknown values. Bayesian thinking treats parameters as random variables with probability distributions. Before seeing data, you have a **prior** belief about the parameter. After observing data, you update that belief using Bayes' theorem to get a **posterior** distribution. This posterior combines your prior knowledge with the evidence from data.

Bayes' theorem: P(theta|data) = P(data|theta) * P(theta) / P(data). The posterior is proportional to the likelihood times the prior. This simple formula has profound implications: it provides a principled way to incorporate prior knowledge, quantify uncertainty, make predictions, and update beliefs as new data arrives.

The Bayesian approach naturally handles small datasets (the prior prevents overfitting), provides uncertainty quantification (you get a distribution, not just a point estimate), and enables sequential updating (each observation updates the posterior, which becomes the prior for the next observation).

### Why naive approaches fail

Point estimates (like the sample mean) give you a single number with no indication of how reliable it is. With 3 data points, the sample mean could be far from the true value, but you have no way to know. Bayesian inference gives you a full posterior distribution: with 3 points, the posterior is wide (high uncertainty); with 1000 points, it is narrow (low uncertainty). The width of the posterior honestly communicates how much you should trust the estimate.

### Mental models

- **Prior as initial guess**: before flipping a coin, you believe P(heads) is around 0.5. This is your prior.
- **Likelihood as evidence**: you flip the coin 10 times and get 8 heads. The likelihood of the data is highest for P(heads) around 0.8.
- **Posterior as updated belief**: combining prior (0.5) and evidence (8/10), the posterior is somewhere between 0.5 and 0.8, depending on how strongly you held your prior.
- **More data shifts toward the evidence**: with 1000 flips (800 heads), the posterior is almost entirely determined by the data, regardless of the prior.

### Visual explanations

```
Bayes' Theorem:
  posterior = likelihood * prior / evidence
  P(theta|D) = P(D|theta) * P(theta) / P(D)

Coin Example:
  Prior: Beta(2, 2) -- believe coin is roughly fair
  Data: 8 heads, 2 tails
  Posterior: Beta(2+8, 2+2) = Beta(10, 4)

  Prior:     .....XXXX.....     (centered at 0.5, wide)
  Likelihood: ........XXXXX.    (peaked at 0.8)
  Posterior:  ......XXXXX...    (shifted toward 0.8, but pulled by prior)

  With more data (80 heads, 20 tails):
  Posterior:  .........XXXX.    (dominated by data, narrow)
```

---

## Hands-on Exploration

1. Define a Beta prior for a coin's bias. Compute the posterior after observing some flips. Visualize how the posterior shifts with more data.
2. Compare the Bayesian estimate (posterior mean) with the frequentist estimate (sample proportion) for small samples (n=5) and large samples (n=500).
3. Show the effect of different priors: a uniform prior (Beta(1,1)) vs a strong prior (Beta(50,50)). How much data is needed to overwhelm a strong prior?
4. Compute a 95% credible interval from the posterior and compare it to a frequentist confidence interval.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    // --- Beta distribution utilities ---
    // Log of the Beta function: ln(B(a,b))
    fn ln_gamma(x: f64) -> f64 {
        // Stirling's approximation for ln(Gamma(x))
        if x < 0.5 {
            let pi = std::f64::consts::PI;
            (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x)
        } else {
            let x = x - 1.0;
            let mut s = 1.0 + 1.0 / (12.0 * x + 1.0 / (10.0 * x));
            0.5 * (2.0 * pi).ln() + (x + 0.5) * x.ln() - x + s.ln()
        }
    }

    fn beta_mean(a: f64, b: f64) -> f64 { a / (a + b) }
    fn beta_var(a: f64, b: f64) -> f64 { (a * b) / ((a + b).powi(2) * (a + b + 1.0)) }

    // Approximate Beta CDF using numerical integration
    fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        if x >= 1.0 { return 1.0; }
        let n_steps = 1000;
        let dx = x / n_steps as f64;
        let mut integral = 0.0;
        for i in 0..n_steps {
            let t = (i as f64 + 0.5) * dx;
            let val = t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0);
            integral += val * dx;
        }
        // Normalize by Beta function
        let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        integral / ln_beta.exp()
    }

    // Find quantile by bisection
    fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            if beta_cdf(mid, a, b) < p { lo = mid; } else { hi = mid; }
        }
        (lo + hi) / 2.0
    }

    // --- Coin flipping example ---
    println!("=== Bayesian Thinking: Coin Bias Estimation ===\n");

    let true_bias = 0.65; // true probability of heads
    let n_flips = 100;

    // Generate data
    let flips: Vec<bool> = (0..n_flips)
        .map(|_| rand_f64(&mut rng) < true_bias)
        .collect();

    // --- Sequential Bayesian updating ---
    println!("=== Sequential Update (Prior: Beta(2,2)) ===\n");

    let prior_a = 2.0;
    let prior_b = 2.0;

    println!("{:>6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12}",
        "n", "Heads", "Tails", "Post.Mean", "Post.Std", "Freq.Est", "95% CI width");
    println!("{}", "-".repeat(72));

    let checkpoints = [1, 5, 10, 20, 50, 100];
    for &n in &checkpoints {
        if n > n_flips { continue; }
        let heads = flips[..n].iter().filter(|&&f| f).count() as f64;
        let tails = n as f64 - heads;

        let post_a = prior_a + heads;
        let post_b = prior_b + tails;

        let post_mean = beta_mean(post_a, post_b);
        let post_std = beta_var(post_a, post_b).sqrt();
        let freq_est = heads / n as f64;

        let ci_lo = beta_quantile(0.025, post_a, post_b);
        let ci_hi = beta_quantile(0.975, post_a, post_b);
        let ci_width = ci_hi - ci_lo;

        println!("{:>6} {:>8.0} {:>8.0} {:>10.4} {:>10.4} {:>10.4} {:>12.4}",
            n, heads, tails, post_mean, post_std, freq_est, ci_width);
    }
    println!("\nTrue bias: {}", true_bias);

    // --- Effect of different priors ---
    println!("\n=== Effect of Prior Strength ===\n");
    println!("Data: {} flips, {} heads", n_flips,
        flips.iter().filter(|&&f| f).count());

    let heads = flips.iter().filter(|&&f| f).count() as f64;
    let tails = n_flips as f64 - heads;

    let priors = vec![
        ("Uniform Beta(1,1)", 1.0, 1.0),
        ("Weak Beta(2,2)", 2.0, 2.0),
        ("Moderate Beta(10,10)", 10.0, 10.0),
        ("Strong Beta(50,50)", 50.0, 50.0),
        ("Very strong Beta(200,200)", 200.0, 200.0),
    ];

    println!("{:<28} {:>10} {:>10} {:>10}", "Prior", "Prior Mean", "Post Mean", "Shrinkage");
    println!("{}", "-".repeat(62));

    for (name, pa, pb) in &priors {
        let prior_mean = beta_mean(*pa, *pb);
        let post_a = pa + heads;
        let post_b = pb + tails;
        let post_mean = beta_mean(post_a, post_b);
        let freq = heads / n_flips as f64;
        let shrinkage = (post_mean - freq).abs() / (prior_mean - freq).abs() * 100.0;

        println!("{:<28} {:>10.3} {:>10.4} {:>9.1}%", name, prior_mean, post_mean, shrinkage);
    }
    println!("\nShrinkage: how much the posterior is pulled toward the prior (vs MLE)");
    println!("Strong priors need more data to overwhelm.");

    // --- Bayesian vs Frequentist with small samples ---
    println!("\n=== Small Sample: Bayesian vs Frequentist ===\n");
    println!("Scenario: 3 flips, all heads. What is P(heads)?\n");

    let small_heads = 3.0;
    let small_n = 3.0;

    let freq_est = small_heads / small_n;
    println!("Frequentist MLE: {:.3} (overconfident!)", freq_est);

    let bayes_a = 2.0 + small_heads; // Beta(2,2) prior
    let bayes_b = 2.0 + (small_n - small_heads);
    let bayes_est = beta_mean(bayes_a, bayes_b);
    let ci_lo = beta_quantile(0.025, bayes_a, bayes_b);
    let ci_hi = beta_quantile(0.975, bayes_a, bayes_b);

    println!("Bayesian (Beta(2,2) prior): {:.3}", bayes_est);
    println!("95% credible interval: [{:.3}, {:.3}]", ci_lo, ci_hi);
    println!("\nBayesian estimate is more sensible: it does not claim P=1.0");
    println!("The credible interval honestly shows high uncertainty.");

    // --- Posterior predictive ---
    println!("\n=== Posterior Predictive ===");
    println!("Q: Given {} flips ({} heads), what is P(next flip = heads)?",
        n_flips, flips.iter().filter(|&&f| f).count());

    let final_a = prior_a + heads;
    let final_b = prior_b + tails;
    let pred = beta_mean(final_a, final_b);
    println!("Posterior predictive P(next=heads) = {:.4}", pred);
    println!("This naturally accounts for parameter uncertainty.");

    println!();
    println!("kata_metric(\"bayesian_estimate\", {:.4})", beta_mean(final_a, final_b));
    println!("kata_metric(\"frequentist_estimate\", {:.4})", heads / n_flips as f64);
    println!("kata_metric(\"posterior_std\", {:.4})", beta_var(final_a, final_b).sqrt());
}
```

---

## Key Takeaways

- **Bayesian inference provides full uncertainty quantification** via posterior distributions, not just point estimates.
- **The prior encodes existing knowledge** and prevents overfitting with small samples. As data increases, the prior's influence diminishes.
- **Credible intervals have a direct probability interpretation**: "there is a 95% probability the parameter lies in this interval."
- **Sequential updating is natural**: each new observation updates the posterior, which becomes the prior for the next observation.
