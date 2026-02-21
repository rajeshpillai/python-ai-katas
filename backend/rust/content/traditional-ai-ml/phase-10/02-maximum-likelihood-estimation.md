# Maximum Likelihood Estimation

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.2

---

## Concept & Intuition

### What problem are we solving?

Given observed data, what parameter values make the data most probable? **Maximum Likelihood Estimation (MLE)** answers this by finding the parameter values that maximize the likelihood function -- the probability of observing the data given the parameters. MLE is the foundation of most statistical estimation and the default approach in frequentist statistics.

For a dataset {x1, x2, ..., xn} assumed to come from a distribution with parameter theta, the likelihood is L(theta) = Product of P(xi | theta). Since products are numerically unstable, we typically maximize the log-likelihood: ln L(theta) = Sum of ln P(xi | theta). The MLE is the theta that maximizes this sum.

MLE has attractive theoretical properties: it is **consistent** (converges to the true parameter as data increases), **efficient** (achieves the lowest possible variance among consistent estimators), and **asymptotically normal** (the sampling distribution becomes Gaussian for large samples). These properties make MLE the go-to estimation method when you have sufficient data.

### Why naive approaches fail

Guessing parameters or using ad-hoc methods can produce estimates that are far from optimal. MLE provides a principled, automatic framework for parameter estimation in any parametric model. However, MLE can overfit with small datasets (no regularization built in), and the likelihood can have multiple local maxima for complex models. These are where Bayesian methods (MAP estimation) offer advantages.

### Mental models

- **MLE as "best explanation"**: which parameter values provide the best explanation for the data we actually observed?
- **Log-likelihood as a surface**: imagine a landscape where height = log-likelihood. MLE is the peak of this landscape.
- **MLE as supervised learning**: linear regression with MSE loss is equivalent to MLE under a Gaussian noise assumption.

### Visual explanations

```
MLE for Gaussian (mean, variance):

  Data: [2.1, 3.5, 2.8, 3.2, 2.6]

  Likelihood for different means (fixed variance):
    mean=1.0:  L = very low  (data is far from 1.0)
    mean=2.0:  L = moderate
    mean=2.84: L = MAXIMUM   <-- MLE estimate
    mean=4.0:  L = low

  Log-likelihood surface:
    mean
    ^
    |     .
    |   .   .
    |  .     .
    | .       .
    |.    *    .   (* = MLE peak at mean=2.84)
    +----------> log-likelihood

  MLE for Gaussian: mean_MLE = sample mean, var_MLE = sample variance
```

---

## Hands-on Exploration

1. Generate data from a known Gaussian. Compute the log-likelihood for different parameter values and find the MLE numerically.
2. Verify that the MLE for a Gaussian equals the sample mean and sample variance.
3. Implement MLE for a Bernoulli (coin flip) model and show it equals the sample proportion.
4. Use gradient ascent to find the MLE for a more complex model where closed-form solutions do not exist.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_normal = |s: &mut u64, mean: f64, std: f64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    // --- Gaussian MLE ---
    println!("=== Maximum Likelihood Estimation ===\n");

    let true_mean = 5.0;
    let true_std = 2.0;
    let n = 100;

    let data: Vec<f64> = (0..n).map(|_| rand_normal(&mut rng, true_mean, true_std)).collect();

    // Log-likelihood function for Gaussian
    let gaussian_ll = |data: &[f64], mean: f64, var: f64| -> f64 {
        let n = data.len() as f64;
        -0.5 * n * (2.0 * pi * var).ln()
        - 0.5 * data.iter().map(|&x| (x - mean).powi(2) / var).sum::<f64>()
    };

    // MLE closed-form for Gaussian
    let mle_mean: f64 = data.iter().sum::<f64>() / n as f64;
    let mle_var: f64 = data.iter().map(|x| (x - mle_mean).powi(2)).sum::<f64>() / n as f64;

    println!("=== Gaussian MLE (n={}) ===", n);
    println!("True:  mean={:.2}, std={:.2}, var={:.2}", true_mean, true_std, true_std * true_std);
    println!("MLE:   mean={:.4}, std={:.4}, var={:.4}", mle_mean, mle_var.sqrt(), mle_var);
    println!("Log-likelihood at MLE: {:.2}", gaussian_ll(&data, mle_mean, mle_var));

    // Show log-likelihood for different mean values
    println!("\n=== Log-Likelihood vs Mean (fixed variance) ===\n");
    println!("{:>8} {:>15}", "Mean", "Log-Likelihood");
    println!("{}", "-".repeat(25));
    for mean_trial in [2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0] {
        let ll = gaussian_ll(&data, mean_trial, mle_var);
        let marker = if (mean_trial - mle_mean).abs() < 0.5 { " <-- near MLE" } else { "" };
        println!("{:>8.1} {:>15.2}{}", mean_trial, ll, marker);
    }

    // --- Bernoulli MLE ---
    println!("\n=== Bernoulli MLE (coin flip) ===\n");
    let true_p = 0.7;
    let n_flips = 50;
    let flips: Vec<f64> = (0..n_flips)
        .map(|_| if rand_f64(&mut rng) < true_p { 1.0 } else { 0.0 })
        .collect();
    let n_heads: f64 = flips.iter().sum();

    // MLE for Bernoulli = sample proportion
    let mle_p = n_heads / n_flips as f64;

    // Log-likelihood for Bernoulli
    let bernoulli_ll = |p: f64, heads: f64, n: f64| -> f64 {
        if p <= 0.0 || p >= 1.0 { return f64::NEG_INFINITY; }
        heads * p.ln() + (n - heads) * (1.0 - p).ln()
    };

    println!("True p: {}", true_p);
    println!("Data: {} flips, {} heads", n_flips, n_heads);
    println!("MLE p: {:.4}", mle_p);
    println!("Log-likelihood at MLE: {:.2}", bernoulli_ll(mle_p, n_heads, n_flips as f64));

    println!("\n{:>8} {:>15}", "p", "Log-Likelihood");
    println!("{}", "-".repeat(25));
    for p_trial in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let ll = bernoulli_ll(p_trial, n_heads, n_flips as f64);
        let marker = if (p_trial - mle_p).abs() < 0.1 { " <-- near MLE" } else { "" };
        println!("{:>8.1} {:>15.2}{}", p_trial, ll, marker);
    }

    // --- Gradient Ascent MLE for mixture model ---
    println!("\n=== Gradient Ascent MLE (Gaussian with unknown mean and variance) ===\n");

    // Start from wrong initial values
    let mut est_mean = 0.0;
    let mut est_var = 1.0;
    let lr_mean = 0.01;
    let lr_var = 0.001;

    println!("{:>5} {:>10} {:>10} {:>15}", "Iter", "Mean", "Var", "Log-Likelihood");
    println!("{}", "-".repeat(44));

    for iter in 0..200 {
        // Gradient of log-likelihood w.r.t. mean
        let grad_mean: f64 = data.iter()
            .map(|&x| (x - est_mean) / est_var)
            .sum::<f64>() / n as f64;

        // Gradient of log-likelihood w.r.t. variance
        let grad_var: f64 = data.iter()
            .map(|&x| -0.5 / est_var + 0.5 * (x - est_mean).powi(2) / est_var.powi(2))
            .sum::<f64>() / n as f64;

        est_mean += lr_mean * grad_mean;
        est_var += lr_var * grad_var;
        est_var = est_var.max(0.01); // keep variance positive

        if iter < 5 || iter == 9 || iter == 19 || iter == 49 || iter == 199 {
            let ll = gaussian_ll(&data, est_mean, est_var);
            println!("{:>5} {:>10.4} {:>10.4} {:>15.2}", iter + 1, est_mean, est_var, ll);
        }
    }

    println!("\nGradient Ascent MLE:  mean={:.4}, var={:.4}", est_mean, est_var);
    println!("Closed-form MLE:      mean={:.4}, var={:.4}", mle_mean, mle_var);
    println!("Match: mean_diff={:.6}, var_diff={:.6}",
        (est_mean - mle_mean).abs(), (est_var - mle_var).abs());

    // --- MLE vs Bayesian comparison ---
    println!("\n=== MLE vs Bayesian (small sample) ===\n");
    let small_data = &data[..5];
    let small_mean: f64 = small_data.iter().sum::<f64>() / 5.0;
    let small_var: f64 = small_data.iter().map(|x| (x - small_mean).powi(2)).sum::<f64>() / 5.0;

    // Bayesian with conjugate prior: Normal-Inverse-Gamma
    // Prior: mean ~ N(0, 10), simplified
    let prior_mean = 0.0;
    let prior_precision = 0.01; // 1/variance of prior on mean
    let data_precision = 5.0 / small_var;

    let posterior_mean = (prior_precision * prior_mean + data_precision * small_mean)
        / (prior_precision + data_precision);

    println!("Small sample (n=5): {:?}",
        small_data.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>());
    println!("MLE mean:      {:.4}", small_mean);
    println!("Bayesian mean: {:.4} (with weak prior at 0)", posterior_mean);
    println!("True mean:     {:.4}", true_mean);
    println!("\nWith small samples, MLE can be noisy. Bayesian estimate is regularized.");

    println!();
    println!("kata_metric(\"mle_mean\", {:.4})", mle_mean);
    println!("kata_metric(\"mle_var\", {:.4})", mle_var);
    println!("kata_metric(\"true_mean\", {:.1})", true_mean);
    println!("kata_metric(\"true_var\", {:.1})", true_std * true_std);
}
```

---

## Key Takeaways

- **MLE finds the parameter values that make the observed data most probable.** It is the default estimation method in statistics and machine learning.
- **For common distributions, MLE has closed-form solutions**: Gaussian mean = sample mean, Bernoulli p = sample proportion.
- **For complex models, gradient ascent on the log-likelihood** provides a general numerical approach to finding the MLE.
- **MLE is optimal for large samples** (consistent, efficient) but can overfit with small samples, where Bayesian methods offer advantages through regularization via priors.
