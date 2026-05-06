# Regularization: Lasso (L1)

> Phase 2 — Supervised Learning: Regression | Kata 2.05

---

## Concept & Intuition

### What problem are we solving?

Lasso regression (Least Absolute Shrinkage and Selection Operator) uses L1 regularization: adding the sum of absolute values of coefficients to the loss function. The key difference from Ridge is that Lasso can drive coefficients to exactly zero, effectively performing automatic feature selection. This makes Lasso invaluable when you suspect that only a few features truly matter among many.

The loss function is: Loss = MSE + lambda * sum(|beta_i|). The L1 penalty creates a diamond-shaped constraint region. The geometry of this diamond means that the optimal solution is more likely to land at a corner — where one or more coefficients are exactly zero — than in Ridge's circular constraint, which favors non-zero values everywhere.

In this kata, we implement Lasso using coordinate descent, the standard optimization method for L1 problems. Unlike Ridge, Lasso does not have a closed-form solution because the absolute value function is not differentiable at zero. Coordinate descent optimizes one coefficient at a time while holding the others fixed, making it both simple and efficient.

### Why naive approaches fail

You cannot solve Lasso with the Normal Equation because the L1 penalty makes the optimization non-smooth (not differentiable at zero). Standard gradient descent struggles because the gradient of |x| is undefined at x=0. Coordinate descent sidesteps this by reducing the multi-dimensional problem to a sequence of one-dimensional problems, each of which has a closed-form solution involving the "soft-thresholding" operator.

### Mental models

- **L1 as feature selector**: The L1 penalty acts like a budget on the sum of absolute coefficient values. Given a limited budget, the model prefers to spend it all on a few important features rather than spreading it thinly across many.
- **Soft thresholding**: The key operation in Lasso. If a coefficient is close to zero (within lambda of zero), it gets set exactly to zero. Otherwise, it is shrunk by lambda toward zero. This is why Lasso produces sparse solutions.
- **Diamond vs. circle**: Visualize the constraint regions: L1 creates a diamond (corners at axes), L2 creates a circle. The loss function contours are more likely to touch the diamond at a corner (where a coefficient is zero).

### Visual explanations

```
  L2 (Ridge) constraint:        L1 (Lasso) constraint:

  beta2                          beta2
    |    ___                       |   /\
    |  /     \                     |  /  \
    | |   *   |                    | /  * \
    |  \     /                     |/      \
    |    ---                       +--------\- beta1
    +----------- beta1              \      /
                                     \  /
  Circle: optimal rarely at axis      \/
                                   Diamond: optimal often at corner
                                   (coefficient = 0)
```

---

## Hands-on Exploration

1. Create a dataset with many features, only a few of which are truly relevant.
2. Implement the soft-thresholding operator.
3. Implement Lasso regression via coordinate descent.
4. Observe how Lasso identifies and eliminates irrelevant features.

---

## Live Code

```rust
fn main() {
    println!("=== Lasso Regression (L1 Regularization) ===\n");

    // Generate data: y = 3*x1 + 0*x2 + 2*x3 + 0*x4 + 0*x5 + 1*x6 + noise
    // Only features 1, 3, 6 are relevant; 2, 4, 5 are noise
    let n = 40;
    let p = 6;
    let true_coeffs = vec![3.0, 0.0, 2.0, 0.0, 0.0, 1.0];
    let true_intercept = 5.0;

    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for _ in 0..n {
        let mut row = Vec::new();
        for _ in 0..p {
            rng = lcg(rng);
            row.push((rng as f64 / u64::MAX as f64) * 10.0);
        }
        rng = lcg(rng);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
        let y = true_intercept + row.iter().zip(true_coeffs.iter())
            .map(|(x, c)| x * c).sum::<f64>() + noise;
        features.push(row);
        targets.push(y);
    }

    let feature_names: Vec<String> = (1..=p).map(|i| format!("x{}", i)).collect();
    println!("True coefficients: {:?}", true_coeffs);
    println!("Relevant features: x1 (3.0), x3 (2.0), x6 (1.0)");
    println!("Noise features: x2, x4, x5 (all 0.0)\n");

    // Standardize features
    let (scaled, means, stds) = standardize(&features);

    // Train-test split
    let train_n = 30;
    let x_train: Vec<Vec<f64>> = scaled[..train_n].to_vec();
    let y_train: Vec<f64> = targets[..train_n].to_vec();
    let x_test: Vec<Vec<f64>> = scaled[train_n..].to_vec();
    let y_test: Vec<f64> = targets[train_n..].to_vec();

    // Center target
    let y_mean = y_train.iter().sum::<f64>() / y_train.len() as f64;
    let y_train_c: Vec<f64> = y_train.iter().map(|y| y - y_mean).collect();
    let y_test_c: Vec<f64> = y_test.iter().map(|y| y - y_mean).collect();

    // Lambda sweep
    let lambdas = vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

    println!("{:<10} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>10} {:>8}",
        "Lambda", "x1", "x2", "x3", "x4", "x5", "x6", "Test MSE", "# Zero");
    println!("{}", "-".repeat(74));

    let mut best_lambda = 0.0;
    let mut best_test_mse = f64::INFINITY;

    for &lambda in &lambdas {
        let coeffs = lasso_coordinate_descent(&x_train, &y_train_c, lambda, 1000, 1e-6);

        let train_preds: Vec<f64> = x_train.iter().map(|row| {
            y_mean + row.iter().zip(coeffs.iter()).map(|(x, c)| x * c).sum::<f64>()
        }).collect();
        let test_preds: Vec<f64> = x_test.iter().map(|row| {
            y_mean + row.iter().zip(coeffs.iter()).map(|(x, c)| x * c).sum::<f64>()
        }).collect();

        let test_mse = mse(&y_test, &test_preds);
        let n_zero = coeffs.iter().filter(|&&c| c.abs() < 1e-6).count();

        println!("{:<10.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>10.4} {:>8}",
            lambda, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5],
            test_mse, n_zero);

        if test_mse < best_test_mse {
            best_test_mse = test_mse;
            best_lambda = lambda;
        }
    }

    println!("\nBest lambda: {} (test MSE: {:.4})\n", best_lambda, best_test_mse);

    // Show best model's feature selection
    let best_coeffs = lasso_coordinate_descent(&x_train, &y_train_c, best_lambda, 1000, 1e-6);

    println!("--- Feature Selection Result ---");
    for i in 0..p {
        let status = if best_coeffs[i].abs() < 1e-6 { "REMOVED" } else { "KEPT" };
        let true_status = if true_coeffs[i].abs() < 1e-6 { "noise" } else { "relevant" };
        let correct = (status == "REMOVED" && true_status == "noise") ||
                      (status == "KEPT" && true_status == "relevant");
        println!("  {}: coeff={:>7.3} {} (truly {}) {}",
            feature_names[i], best_coeffs[i], status, true_status,
            if correct { "CORRECT" } else { "WRONG" });
    }

    // Coefficient path visualization
    println!("\n--- Lasso Coefficient Path ---");
    let path_lambdas: Vec<f64> = (0..25).map(|i| 0.01 * (1.3_f64).powi(i)).collect();
    for feat in 0..p {
        print!("  x{}: ", feat + 1);
        for &lam in &path_lambdas {
            let c = lasso_coordinate_descent(&x_train, &y_train_c, lam, 500, 1e-5);
            let ch = if c[feat].abs() < 1e-6 { ' ' }
                     else if c[feat].abs() > 5.0 { '#' }
                     else if c[feat].abs() > 2.0 { 'o' }
                     else { '.' };
            print!("{}", ch);
        }
        let status = if true_coeffs[feat].abs() > 0.0 { "relevant" } else { "noise" };
        println!("  ({})", status);
    }
    println!("       small lambda ---------> large lambda");

    kata_metric("best_lambda", best_lambda);
    kata_metric("best_test_mse", best_test_mse);
    let n_selected = best_coeffs.iter().filter(|&&c| c.abs() > 1e-6).count();
    kata_metric("features_selected", n_selected as f64);
    kata_metric("total_features", p as f64);
}

fn lasso_coordinate_descent(
    x: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = x.len() as f64;
    let p = x[0].len();
    let mut coeffs = vec![0.0; p];

    for _ in 0..max_iter {
        let mut max_change = 0.0;

        for j in 0..p {
            // Compute partial residual (excluding feature j)
            let mut residuals: Vec<f64> = y.to_vec();
            for i in 0..x.len() {
                for k in 0..p {
                    if k != j {
                        residuals[i] -= coeffs[k] * x[i][k];
                    }
                }
            }

            // Compute rho_j = sum(x_ij * residual_i) / n
            let rho: f64 = (0..x.len())
                .map(|i| x[i][j] * residuals[i])
                .sum::<f64>() / n;

            // Soft thresholding
            let old_coeff = coeffs[j];
            coeffs[j] = soft_threshold(rho, lambda);

            max_change = max_change.max((coeffs[j] - old_coeff).abs());
        }

        if max_change < tol {
            break;
        }
    }

    coeffs
}

fn soft_threshold(rho: f64, lambda: f64) -> f64 {
    if rho > lambda {
        rho - lambda
    } else if rho < -lambda {
        rho + lambda
    } else {
        0.0
    }
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

- Lasso (L1 regularization) adds the sum of absolute coefficient values to the loss, producing sparse solutions where some coefficients are exactly zero.
- This automatic feature selection is Lasso's killer feature: it identifies which features matter and which are noise.
- Coordinate descent with soft thresholding is the standard algorithm for Lasso since the L1 penalty is not differentiable.
- Compared to Ridge (L2), Lasso is better when you believe the true model is sparse. Ridge is better when all features contribute but some are correlated.
