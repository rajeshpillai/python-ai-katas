# Elastic Net

> Phase 2 — Supervised Learning: Regression | Kata 2.06

---

## Concept & Intuition

### What problem are we solving?

Elastic Net combines the strengths of Ridge (L2) and Lasso (L1) regularization into a single model. The penalty term is a weighted combination: lambda * (alpha * L1 + (1-alpha) * L2), where alpha controls the mix. When alpha=1, it is pure Lasso; when alpha=0, it is pure Ridge. Values in between give a hybrid that can both select features (L1) and handle correlated features gracefully (L2).

Lasso has a known limitation with correlated features: when two features are highly correlated, Lasso arbitrarily picks one and zeros out the other. This is unstable — a small change in data might flip which feature is selected. Ridge keeps both but cannot zero out irrelevant features. Elastic Net solves this by grouping correlated features (keeping all or none in a group) while still performing feature selection on uncorrelated noise features.

In this kata, we implement Elastic Net using coordinate descent and explore how the alpha parameter shapes the model's behavior across the Ridge-Lasso spectrum.

### Why naive approaches fail

Using only Lasso when features are correlated produces unstable feature selection. Using only Ridge when you need feature selection keeps too many features. Manually choosing between Ridge and Lasso for each problem is suboptimal. Elastic Net automates this tradeoff through the alpha parameter, which can be tuned via cross-validation just like lambda.

### Mental models

- **Alpha as a dial between L1 and L2**: alpha=0 is Ridge (circular constraint), alpha=1 is Lasso (diamond constraint), alpha=0.5 is a rounded diamond. The shape determines whether coefficients reach exactly zero.
- **Group effect**: When features are correlated, Elastic Net tends to select them as a group — either all in or all out. This is more stable than Lasso's arbitrary single-feature selection.
- **Two hyperparameters**: Lambda controls overall regularization strength, alpha controls the L1/L2 mix. A grid search over both gives the best model.

### Visual explanations

```
  alpha=0 (Ridge)     alpha=0.5 (Elastic)    alpha=1 (Lasso)

  beta2                beta2                   beta2
    |   __               |   _                   |  /\
    | /    \             | /   \                  | /  \
    ||      |            |/     \                 |/    \
    | \    /             |\     /                 |\    /
    |   --               |  \_/                   | \  /
    +-------- beta1      +-------- beta1          +-------- beta1

  No sparsity          Some sparsity           Max sparsity
  Handles correlation  Balanced                Feature selection
```

---

## Hands-on Exploration

1. Create data with both correlated features and irrelevant noise features.
2. Implement Elastic Net using coordinate descent.
3. Compare Ridge, Lasso, and Elastic Net on this data.
4. Sweep over alpha and lambda to find the best combination.

---

## Live Code

```rust
fn main() {
    println!("=== Elastic Net Regression ===\n");

    // Generate data with correlated features and noise
    // True: y = 3*x1 + 3*x2 + 2*x3 + 0*x4 + 0*x5
    // x1 and x2 are highly correlated
    let n = 50;
    let mut rng = 42u64;

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for _ in 0..n {
        rng = lcg(rng);
        let x1 = (rng as f64 / u64::MAX as f64) * 10.0;
        rng = lcg(rng);
        let x2 = x1 + (rng as f64 / u64::MAX as f64 - 0.5) * 1.0; // correlated with x1
        rng = lcg(rng);
        let x3 = (rng as f64 / u64::MAX as f64) * 10.0;
        rng = lcg(rng);
        let x4 = (rng as f64 / u64::MAX as f64) * 10.0; // noise
        rng = lcg(rng);
        let x5 = (rng as f64 / u64::MAX as f64) * 10.0; // noise

        rng = lcg(rng);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 3.0;
        let y = 3.0 * x1 + 3.0 * x2 + 2.0 * x3 + 5.0 + noise;

        features.push(vec![x1, x2, x3, x4, x5]);
        targets.push(y);
    }

    let names = vec!["x1", "x2", "x3", "x4", "x5"];
    let true_coeffs = vec![3.0, 3.0, 2.0, 0.0, 0.0];

    println!("True: y = 3*x1 + 3*x2 + 2*x3 + 0*x4 + 0*x5 + 5");
    println!("x1 and x2 are highly correlated\n");

    // Standardize
    let (scaled, means, stds) = standardize(&features);

    // Split
    let train_n = 38;
    let x_train: Vec<Vec<f64>> = scaled[..train_n].to_vec();
    let y_train: Vec<f64> = targets[..train_n].to_vec();
    let x_test: Vec<Vec<f64>> = scaled[train_n..].to_vec();
    let y_test: Vec<f64> = targets[train_n..].to_vec();

    let y_mean = y_train.iter().sum::<f64>() / y_train.len() as f64;
    let y_train_c: Vec<f64> = y_train.iter().map(|y| y - y_mean).collect();

    // Compare Ridge, Lasso, Elastic Net
    println!("--- Comparison: Ridge vs Lasso vs Elastic Net ---");
    println!("(lambda=1.0 for all)\n");

    let lambda = 1.0;

    // Ridge (alpha=0)
    let ridge_c = elastic_net_fit(&x_train, &y_train_c, lambda, 0.0, 1000, 1e-6);
    let ridge_preds = predict_with_intercept(&x_test, &ridge_c, y_mean);
    let ridge_mse = mse(&y_test, &ridge_preds);

    // Lasso (alpha=1)
    let lasso_c = elastic_net_fit(&x_train, &y_train_c, lambda, 1.0, 1000, 1e-6);
    let lasso_preds = predict_with_intercept(&x_test, &lasso_c, y_mean);
    let lasso_mse = mse(&y_test, &lasso_preds);

    // Elastic Net (alpha=0.5)
    let enet_c = elastic_net_fit(&x_train, &y_train_c, lambda, 0.5, 1000, 1e-6);
    let enet_preds = predict_with_intercept(&x_test, &enet_c, y_mean);
    let enet_mse = mse(&y_test, &enet_preds);

    println!("{:<15} {:>7} {:>7} {:>7} {:>7} {:>7} {:>10}",
        "Method", "x1", "x2", "x3", "x4", "x5", "Test MSE");
    println!("{}", "-".repeat(62));
    println!("{:<15} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>10.4}",
        "Ridge (a=0)", ridge_c[0], ridge_c[1], ridge_c[2], ridge_c[3], ridge_c[4], ridge_mse);
    println!("{:<15} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>10.4}",
        "Lasso (a=1)", lasso_c[0], lasso_c[1], lasso_c[2], lasso_c[3], lasso_c[4], lasso_mse);
    println!("{:<15} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>10.4}",
        "Elastic (a=.5)", enet_c[0], enet_c[1], enet_c[2], enet_c[3], enet_c[4], enet_mse);

    // Highlight the grouped selection problem
    println!("\n--- Correlated Feature Handling ---");
    println!("  x1 and x2 are correlated (both should have nonzero coefficients):");
    println!("  Lasso: x1={:.3}, x2={:.3} (may zero one out!)", lasso_c[0], lasso_c[1]);
    println!("  Elastic: x1={:.3}, x2={:.3} (keeps both)", enet_c[0], enet_c[1]);
    println!("  Ridge: x1={:.3}, x2={:.3} (keeps both)", ridge_c[0], ridge_c[1]);

    // Grid search over alpha and lambda
    println!("\n--- Grid Search: alpha x lambda ---");
    let alphas = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let lambdas = vec![0.1, 0.5, 1.0, 2.0, 5.0];

    println!("{:<8} {:<8} {:>10} {:>8}", "Alpha", "Lambda", "Test MSE", "# Zero");
    println!("{}", "-".repeat(36));

    let mut best_alpha = 0.0;
    let mut best_lambda_val = 0.0;
    let mut best_mse = f64::INFINITY;

    for &alpha in &alphas {
        for &lam in &lambdas {
            let c = elastic_net_fit(&x_train, &y_train_c, lam, alpha, 1000, 1e-6);
            let preds = predict_with_intercept(&x_test, &c, y_mean);
            let test_mse = mse(&y_test, &preds);
            let n_zero = c.iter().filter(|&&v| v.abs() < 1e-6).count();

            if test_mse < best_mse {
                best_mse = test_mse;
                best_alpha = alpha;
                best_lambda_val = lam;
            }

            println!("{:<8.2} {:<8.1} {:>10.4} {:>8}", alpha, lam, test_mse, n_zero);
        }
    }

    println!("\nBest: alpha={}, lambda={} (test MSE: {:.4})", best_alpha, best_lambda_val, best_mse);

    // Best model details
    let best_c = elastic_net_fit(&x_train, &y_train_c, best_lambda_val, best_alpha, 1000, 1e-6);
    println!("\n--- Best Model Coefficients ---");
    for (i, name) in names.iter().enumerate() {
        let status = if best_c[i].abs() < 1e-6 { "zero" } else { "nonzero" };
        println!("  {}: {:>8.4} ({}, true={:.1})", name, best_c[i], status, true_coeffs[i]);
    }

    kata_metric("ridge_test_mse", ridge_mse);
    kata_metric("lasso_test_mse", lasso_mse);
    kata_metric("elastic_test_mse", enet_mse);
    kata_metric("best_alpha", best_alpha);
    kata_metric("best_lambda", best_lambda_val);
    kata_metric("best_test_mse", best_mse);
}

fn elastic_net_fit(
    x: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = x.len() as f64;
    let p = x[0].len();
    let mut coeffs = vec![0.0; p];

    let l1_weight = alpha * lambda;
    let l2_weight = (1.0 - alpha) * lambda;

    for _ in 0..max_iter {
        let mut max_change = 0.0;

        for j in 0..p {
            let mut residuals: Vec<f64> = y.to_vec();
            for i in 0..x.len() {
                for k in 0..p {
                    if k != j {
                        residuals[i] -= coeffs[k] * x[i][k];
                    }
                }
            }

            let rho: f64 = (0..x.len())
                .map(|i| x[i][j] * residuals[i])
                .sum::<f64>() / n;

            let old_coeff = coeffs[j];

            // Elastic Net update: soft threshold then scale by L2
            let numerator = soft_threshold(rho, l1_weight);
            let denominator = 1.0 + l2_weight;
            coeffs[j] = numerator / denominator;

            max_change = max_change.max((coeffs[j] - old_coeff).abs());
        }

        if max_change < tol {
            break;
        }
    }

    coeffs
}

fn soft_threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold { x - threshold }
    else if x < -threshold { x + threshold }
    else { 0.0 }
}

fn predict_with_intercept(x: &[Vec<f64>], coeffs: &[f64], intercept: f64) -> Vec<f64> {
    x.iter().map(|row| {
        intercept + row.iter().zip(coeffs.iter()).map(|(xi, ci)| xi * ci).sum::<f64>()
    }).collect()
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len() as f64;
    let p = data[0].len();
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];
    for j in 0..p {
        let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        means[j] = col.iter().sum::<f64>() / n;
        stds[j] = (col.iter().map(|x| (x - means[j]).powi(2)).sum::<f64>() / n).sqrt();
    }
    let scaled = data.iter().map(|row| {
        row.iter().enumerate().map(|(j, &v)| {
            if stds[j].abs() < 1e-10 { 0.0 } else { (v - means[j]) / stds[j] }
        }).collect()
    }).collect();
    (scaled, means, stds)
}

fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>() / actual.len() as f64
}

fn lcg(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties, controlled by the alpha mixing parameter.
- It handles correlated features better than Lasso alone by encouraging grouped selection (keeping or removing correlated features together).
- The two hyperparameters (lambda for strength, alpha for L1/L2 mix) are typically tuned via cross-validation grid search.
- Elastic Net is the default choice when you are unsure whether Ridge or Lasso is more appropriate — it covers both extremes and everything in between.
