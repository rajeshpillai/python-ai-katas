# MAP Estimation

> Phase 10 — Probabilistic & Bayesian Methods | Kata 10.3

---

## Concept & Intuition

### What problem are we solving?

**Maximum A Posteriori (MAP) estimation** finds the parameter value that maximizes the posterior distribution -- the most probable parameter value given both the data and a prior belief. MAP sits between MLE (no prior) and full Bayesian inference (entire posterior distribution). It adds the benefits of prior regularization to the computational simplicity of point estimation.

MAP maximizes: P(theta|data) proportional to P(data|theta) * P(theta). Taking logarithms: theta_MAP = argmax [ln P(data|theta) + ln P(theta)]. The first term is the log-likelihood (same as MLE). The second term is the log-prior, which acts as a **regularization penalty**. A Gaussian prior on parameters is equivalent to L2 regularization (Ridge), and a Laplacian prior is equivalent to L1 regularization (Lasso).

This connection between Bayesian priors and regularization is one of the most important insights in machine learning. Every time you add a penalty term to a loss function, you are implicitly choosing a prior distribution over the parameters.

### Why naive approaches fail

MLE without regularization overfits on small datasets because it has no mechanism to prefer simpler models. Adding ad-hoc regularization (like weight decay) works but lacks a principled justification. MAP estimation provides that justification: the regularization strength corresponds to the certainty of your prior belief about the parameters. A strong prior (small variance) equals strong regularization; a weak prior (large variance) equals weak regularization.

### Mental models

- **MAP = MLE + penalty**: the prior adds a penalty term that pushes parameters toward "reasonable" values.
- **Gaussian prior = Ridge regression**: the log of a Gaussian prior is -lambda * theta^2, exactly the L2 penalty.
- **MAP as a compromise**: between what the data says (likelihood) and what you believe a priori (prior). With lots of data, MAP converges to MLE.

### Visual explanations

```
MAP vs MLE:

  log P(theta|data) = log P(data|theta) + log P(theta) - const
                      ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
                      log-likelihood      log-prior (regularization)

  MLE:  maximize log-likelihood only
  MAP:  maximize log-likelihood + log-prior

  Gaussian prior N(0, sigma^2):
    log P(theta) = -theta^2 / (2*sigma^2) + const
    MAP = argmax [log-likelihood - lambda * theta^2]
    where lambda = 1 / (2 * sigma^2)

  This is EXACTLY Ridge regression!

  Prior variance    Regularization
  Large (weak)  --> Small lambda (almost MLE)
  Small (strong)--> Large lambda (heavy regularization)
```

---

## Hands-on Exploration

1. Implement MLE and MAP for linear regression. Show that MAP with a Gaussian prior is equivalent to Ridge regression.
2. Generate a small, noisy dataset. Compare MLE (high variance, potentially overfit) with MAP (lower variance, regularized).
3. Vary the prior strength and observe the bias-variance trade-off: strong prior reduces variance but increases bias.
4. Show that as data increases, MLE and MAP converge -- the data overwhelms the prior.

---

## Live Code

```rust
fn main() {
    let pi = std::f64::consts::PI;

    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_normal = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*s >> 33) as f64 / 2147483648.0).max(1e-10);
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*s >> 33) as f64 / 2147483648.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * pi * u2).cos()
    };

    // --- Generate polynomial regression data ---
    let true_weights = [2.0, -1.5, 0.5]; // y = 2 - 1.5x + 0.5x^2
    let n = 15; // small sample to show regularization effect
    let noise_std = 1.5;

    let mut x_data: Vec<f64> = Vec::new();
    let mut y_data: Vec<f64> = Vec::new();

    for _ in 0..n {
        let x = rand_normal(&mut rng) * 2.0;
        let y = true_weights[0] + true_weights[1] * x + true_weights[2] * x * x
            + rand_normal(&mut rng) * noise_std;
        x_data.push(x);
        y_data.push(y);
    }

    // Build feature matrix: [1, x, x^2, x^3, x^4] (intentionally overparameterized)
    let degree = 6;
    let n_feat = degree + 1;

    let build_features = |x: &[f64], deg: usize| -> Vec<Vec<f64>> {
        x.iter().map(|&xi| {
            (0..=deg).map(|d| xi.powi(d as i32)).collect()
        }).collect()
    };

    let features = build_features(&x_data, degree);

    // --- MLE (Ordinary Least Squares) ---
    fn fit_linear(features: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
        let n = features.len();
        let n_feat = features[0].len();
        let mut w = vec![0.0; n_feat];
        let lr = 0.0001;

        for _ in 0..10000 {
            let mut grad = vec![0.0; n_feat];
            for i in 0..n {
                let pred: f64 = w.iter().zip(&features[i]).map(|(wi, xi)| wi * xi).sum();
                let err = pred - y[i];
                for j in 0..n_feat {
                    grad[j] += err * features[i][j];
                }
            }
            for j in 0..n_feat {
                grad[j] += lambda * w[j]; // L2 regularization gradient
                w[j] -= lr * 2.0 * grad[j] / n as f64;
            }
        }
        w
    }

    fn predict(features: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
        features.iter().map(|row| {
            w.iter().zip(row).map(|(wi, xi)| wi * xi).sum()
        }).collect()
    }

    fn mse(pred: &[f64], truth: &[f64]) -> f64 {
        pred.iter().zip(truth).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64
    }

    // Generate test data
    let n_test = 50;
    let test_x: Vec<f64> = (0..n_test).map(|_| rand_normal(&mut rng) * 2.0).collect();
    let test_y: Vec<f64> = test_x.iter().map(|&x| {
        true_weights[0] + true_weights[1] * x + true_weights[2] * x * x
        + rand_normal(&mut rng) * noise_std
    }).collect();
    let test_features = build_features(&test_x, degree);

    // --- Compare MLE vs MAP with different lambdas ---
    println!("=== MAP Estimation (MLE + Regularization) ===\n");
    println!("True model: y = {:.1} + {:.1}*x + {:.1}*x^2", true_weights[0], true_weights[1], true_weights[2]);
    println!("Fitting degree-{} polynomial with n={} samples\n", degree, n);

    let lambdas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0];

    println!("{:>8} {:>12} {:>12} {:>10}", "Lambda", "Train MSE", "Test MSE", "||w||");
    println!("{}", "-".repeat(44));

    let mut best_lambda = 0.0;
    let mut best_test_mse = f64::MAX;

    for &lambda in &lambdas {
        let w = fit_linear(&features, &y_data, lambda);
        let train_pred = predict(&features, &w);
        let test_pred = predict(&test_features, &w);

        let train_mse = mse(&train_pred, &y_data);
        let test_mse = mse(&test_pred, &test_y);
        let w_norm: f64 = w.iter().map(|wi| wi.powi(2)).sum::<f64>().sqrt();

        let label = if lambda == 0.0 { " (MLE)" }
            else if test_mse < best_test_mse { " (best)" }
            else { "" };

        println!("{:>8.2} {:>12.4} {:>12.4} {:>10.3}{}", lambda, train_mse, test_mse, w_norm, label);

        if test_mse < best_test_mse {
            best_test_mse = test_mse;
            best_lambda = lambda;
        }
    }

    // --- Show weight values ---
    println!("\n=== Learned Weights ===\n");
    println!("{:>8}  {}", "Lambda", (0..=degree).map(|d| format!("{:>8}", format!("w{}", d))).collect::<Vec<_>>().join(""));
    println!("{}", "-".repeat(8 + 9 * (degree + 1)));

    println!("{:>8}  {}", "True",
        (0..=degree).map(|d| {
            let v = if d < true_weights.len() { true_weights[d] } else { 0.0 };
            format!("{:>8.3}", v)
        }).collect::<Vec<_>>().join(""));

    for &lambda in &[0.0, 0.1, 1.0, 10.0] {
        let w = fit_linear(&features, &y_data, lambda);
        let label = if lambda == 0.0 { " MLE" }
            else { &format!(" l={}", lambda) };
        println!("{:>8}  {}", label,
            w.iter().map(|v| format!("{:>8.3}", v)).collect::<Vec<_>>().join(""));
    }

    // --- Bayesian interpretation ---
    println!("\n=== Bayesian Interpretation ===\n");
    println!("MAP with Gaussian prior N(0, sigma^2) <=> Ridge with lambda = 1/(2*sigma^2)\n");

    for &lambda in &[0.01, 0.1, 1.0, 10.0] {
        let prior_var = 1.0 / (2.0 * lambda);
        let prior_std = prior_var.sqrt();
        println!("  lambda={:>5.2} <=> prior N(0, {:.2}) [std={:.2}]  ({})",
            lambda, prior_var, prior_std,
            if lambda < 0.1 { "weak prior" }
            else if lambda < 1.0 { "moderate prior" }
            else { "strong prior" });
    }

    // --- Convergence: MLE and MAP agree with more data ---
    println!("\n=== MLE vs MAP Convergence with More Data ===\n");
    println!("{:>6} {:>12} {:>12} {:>12}", "n", "MLE w1", "MAP w1", "Difference");
    println!("{}", "-".repeat(46));

    for &n_data in &[5, 15, 50, 200, 500] {
        let mut xd: Vec<f64> = (0..n_data).map(|_| rand_normal(&mut rng) * 2.0).collect();
        let yd: Vec<f64> = xd.iter().map(|&x| {
            true_weights[0] + true_weights[1] * x + true_weights[2] * x * x
            + rand_normal(&mut rng) * noise_std
        }).collect();
        let feat = build_features(&xd, degree);

        let w_mle = fit_linear(&feat, &yd, 0.0);
        let w_map = fit_linear(&feat, &yd, 1.0);

        println!("{:>6} {:>12.4} {:>12.4} {:>12.4}",
            n_data, w_mle[1], w_map[1], (w_mle[1] - w_map[1]).abs());
    }
    println!("\nAs n increases, MLE and MAP converge (data overwhelms the prior).");

    println!();
    println!("kata_metric(\"best_lambda\", {:.2})", best_lambda);
    println!("kata_metric(\"best_test_mse\", {:.4})", best_test_mse);
    let mle_w = fit_linear(&features, &y_data, 0.0);
    let mle_test = mse(&predict(&test_features, &mle_w), &test_y);
    println!("kata_metric(\"mle_test_mse\", {:.4})", mle_test);
}
```

---

## Key Takeaways

- **MAP estimation adds a prior to MLE**, acting as regularization that prevents overfitting, especially with small datasets.
- **Gaussian prior = L2 regularization (Ridge).** Laplacian prior = L1 regularization (Lasso). Every regularizer has a Bayesian interpretation.
- **The prior strength (or regularization lambda) controls the bias-variance trade-off.** Stronger priors reduce variance but increase bias.
- **With enough data, MAP converges to MLE.** The prior becomes irrelevant as the likelihood dominates the posterior.
