# Regularization: Ridge (L2)

> Phase 2 — Supervised Learning: Regression | Kata 2.04

---

## Concept & Intuition

### What problem are we solving?

Overfitting happens when a model fits the noise in the training data rather than the underlying signal. One effective countermeasure is regularization: adding a penalty term to the loss function that discourages overly complex models. Ridge regression (L2 regularization) adds the sum of squared coefficients to the loss, pushing all coefficients toward zero without eliminating any entirely.

The loss function becomes: Loss = MSE + lambda * sum(beta_i^2), where lambda controls the strength of regularization. When lambda is 0, we get ordinary least squares. As lambda increases, coefficients shrink toward zero, producing a simpler model that generalizes better. The challenge is finding the right lambda — too small and overfitting persists, too large and the model underfits.

Ridge regression also stabilizes the solution when features are correlated (multicollinearity). With correlated features, the OLS solution is unstable — small changes in data cause large changes in coefficients. Ridge regularization dampens this instability by adding lambda*I to the X'X matrix before inverting it.

### Why naive approaches fail

Simply removing features to reduce complexity is crude and loses information. Reducing the polynomial degree limits expressiveness. OLS with correlated features produces unreliable coefficients with huge magnitudes that cancel each other out. Ridge regression elegantly solves all three problems: it keeps all features but controls their influence, it allows complex models while preventing overfitting, and it stabilizes coefficients even with multicollinearity.

### Mental models

- **Regularization as a budget constraint**: Think of the coefficients as having a "budget." Ridge limits the total squared magnitude of all coefficients, forcing the model to allocate its budget wisely among features.
- **Ridge as shrinkage**: Every coefficient is pulled toward zero by a force proportional to lambda. Important features resist the pull (their gradient from the data is strong); unimportant features are pulled to near-zero.
- **The ridge in "Ridge"**: Adding lambda*I to X'X creates a "ridge" along the diagonal, ensuring the matrix is always invertible. This is where the name comes from.

### Visual explanations

```
  OLS coefficients:              Ridge coefficients (lambda=1):
  beta1 = 15.3                   beta1 = 4.2
  beta2 = -12.8                  beta2 = -3.1
  beta3 = 0.5                    beta3 = 0.3
  (large, unstable)              (smaller, stable)

  Loss landscape:
            OLS                           Ridge
  beta2 |  ---o---              beta2 |  ---o---
        | /       \                   | /       \
        |/    *    \                  |/ *  *    \   <- circular penalty
        |\   best  /                  |\  best   /      shrinks toward origin
        | \       /                   | \       /
        |  -------                    |  -------
        +---------- beta1             +---------- beta1

  * = OLS minimum               * = Ridge minimum (closer to origin)
```

---

## Hands-on Exploration

1. Generate data with multicollinear features to show OLS instability.
2. Implement Ridge regression with the closed-form solution.
3. Sweep across lambda values and observe coefficient shrinkage.
4. Compare OLS and Ridge on train/test performance.

---

## Live Code

```rust
fn main() {
    println!("=== Ridge Regression (L2 Regularization) ===\n");

    // Generate data with correlated features
    let n = 30;
    let mut rng = 42u64;

    let mut x1 = Vec::new();
    let mut x2 = Vec::new();
    let mut x3 = Vec::new();
    let mut y = Vec::new();

    for _ in 0..n {
        rng = lcg(rng);
        let val1 = (rng as f64 / u64::MAX as f64) * 10.0;
        rng = lcg(rng);
        let noise_corr = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
        let val2 = val1 * 0.8 + noise_corr; // x2 correlated with x1
        rng = lcg(rng);
        let val3 = (rng as f64 / u64::MAX as f64) * 10.0;
        rng = lcg(rng);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;

        let target = 3.0 * val1 + 2.0 * val2 + 1.0 * val3 + 5.0 + noise;
        x1.push(val1);
        x2.push(val2);
        x3.push(val3);
        y.push(target);
    }

    // Build feature matrix with intercept
    let features: Vec<Vec<f64>> = (0..n).map(|i| {
        vec![1.0, x1[i], x2[i], x3[i]]
    }).collect();

    // Split train/test
    let train_n = 22;
    let x_train: Vec<Vec<f64>> = features[..train_n].to_vec();
    let y_train: Vec<f64> = y[..train_n].to_vec();
    let x_test: Vec<Vec<f64>> = features[train_n..].to_vec();
    let y_test: Vec<f64> = y[train_n..].to_vec();

    // Check correlation between x1 and x2
    let corr_12 = correlation(&x1, &x2);
    println!("Correlation between x1 and x2: {:.4} (highly correlated!)\n", corr_12);

    // OLS (lambda = 0)
    println!("--- OLS (no regularization) ---");
    let ols_coeffs = ridge_fit(&x_train, &y_train, 0.0);
    let ols_train_preds = predict(&x_train, &ols_coeffs);
    let ols_test_preds = predict(&x_test, &ols_coeffs);
    println!("  Coefficients: [{:.3}, {:.3}, {:.3}, {:.3}]",
        ols_coeffs[0], ols_coeffs[1], ols_coeffs[2], ols_coeffs[3]);
    println!("  True: [5.0, 3.0, 2.0, 1.0]");
    println!("  Train MSE: {:.4}", mse(&y_train, &ols_train_preds));
    println!("  Test MSE:  {:.4}", mse(&y_test, &ols_test_preds));

    // Ridge with different lambda values
    let lambdas = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

    println!("\n--- Ridge Regression: Lambda Sweep ---");
    println!("{:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "Lambda", "b0", "b1", "b2", "b3", "Train MSE", "Test MSE");
    println!("{}", "-".repeat(68));

    let mut best_lambda = 0.0;
    let mut best_test_mse = f64::INFINITY;

    for &lambda in &lambdas {
        let coeffs = ridge_fit(&x_train, &y_train, lambda);
        let train_preds = predict(&x_train, &coeffs);
        let test_preds = predict(&x_test, &coeffs);
        let train_mse = mse(&y_train, &train_preds);
        let test_mse = mse(&y_test, &test_preds);

        println!("{:<10.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10.4} {:>10.4}",
            lambda, coeffs[0], coeffs[1], coeffs[2], coeffs[3], train_mse, test_mse);

        if test_mse < best_test_mse {
            best_test_mse = test_mse;
            best_lambda = lambda;
        }
    }

    println!("\nBest lambda: {} (test MSE: {:.4})", best_lambda, best_test_mse);

    // Coefficient path visualization
    println!("\n--- Coefficient Shrinkage Path ---");
    println!("(How coefficients change as lambda increases)\n");
    let path_lambdas: Vec<f64> = (0..20).map(|i| 10.0_f64.powf(-2.0 + i as f64 * 0.3)).collect();

    for feat_idx in 1..=3 {
        print!("  b{}: ", feat_idx);
        for &lam in &path_lambdas {
            let c = ridge_fit(&x_train, &y_train, lam);
            let normalized = (c[feat_idx].abs() / 10.0).min(1.0);
            let ch = if normalized > 0.7 { '#' }
                     else if normalized > 0.3 { 'o' }
                     else if normalized > 0.1 { '.' }
                     else { ' ' };
            print!("{}", ch);
        }
        println!("  (feature {})", feat_idx);
    }
    println!("       small lambda  ---------->  large lambda");

    // Coefficient magnitude comparison
    println!("\n--- L2 Norm of Coefficients ---");
    let ols_l2: f64 = ols_coeffs[1..].iter().map(|c| c * c).sum::<f64>().sqrt();
    let ridge_best = ridge_fit(&x_train, &y_train, best_lambda);
    let ridge_l2: f64 = ridge_best[1..].iter().map(|c| c * c).sum::<f64>().sqrt();
    println!("  OLS L2 norm:   {:.4}", ols_l2);
    println!("  Ridge L2 norm: {:.4}", ridge_l2);
    println!("  Shrinkage: {:.1}%", (1.0 - ridge_l2 / ols_l2) * 100.0);

    kata_metric("correlation_x1_x2", corr_12);
    kata_metric("ols_test_mse", mse(&y_test, &ols_test_preds));
    kata_metric("best_lambda", best_lambda);
    kata_metric("best_ridge_test_mse", best_test_mse);
    kata_metric("ols_l2_norm", ols_l2);
    kata_metric("ridge_l2_norm", ridge_l2);
}

fn ridge_fit(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    // beta = (X'X + lambda*I)^-1 X'y
    // Note: we don't regularize the intercept (index 0)
    let xt = transpose(x);
    let mut xtx = mat_mul(&xt, x);
    let p = xtx.len();

    // Add lambda to diagonal (skip intercept)
    for i in 1..p {
        xtx[i][i] += lambda;
    }

    let xty = mat_vec_mul(&xt, y);
    let xtx_inv = invert_matrix(&xtx);
    mat_vec_mul(&xtx_inv, &xty)
}

fn predict(x: &[Vec<f64>], coeffs: &[f64]) -> Vec<f64> {
    x.iter().map(|row| {
        row.iter().zip(coeffs.iter()).map(|(xi, ci)| xi * ci).sum()
    }).collect()
}

fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>() / actual.len() as f64
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        cov += (x[i] - mx) * (y[i] - my);
        vx += (x[i] - mx).powi(2);
        vy += (y[i] - my).powi(2);
    }
    cov / (vx.sqrt() * vy.sqrt())
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m.len();
    let cols = m[0].len();
    let mut r = vec![vec![0.0; rows]; cols];
    for i in 0..rows { for j in 0..cols { r[j][i] = m[i][j]; } }
    r
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (rows, cols, inner) = (a.len(), b[0].len(), b.len());
    let mut r = vec![vec![0.0; cols]; rows];
    for i in 0..rows { for j in 0..cols { for k in 0..inner {
        r[i][j] += a[i][k] * b[k][j];
    }}}
    r
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()).collect()
}

fn invert_matrix(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n { for j in 0..n { aug[i][j] = m[i][j]; } aug[i][n + i] = 1.0; }
    for col in 0..n {
        let mut max_r = col;
        for row in (col+1)..n { if aug[row][col].abs() > aug[max_r][col].abs() { max_r = row; } }
        aug.swap(col, max_r);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 { continue; }
        for j in 0..(2*n) { aug[col][j] /= pivot; }
        for row in 0..n { if row != col {
            let f = aug[row][col];
            for j in 0..(2*n) { aug[row][j] -= f * aug[col][j]; }
        }}
    }
    aug.iter().map(|row| row[n..].to_vec()).collect()
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

- Ridge regression adds an L2 penalty (sum of squared coefficients) to the loss function, discouraging large coefficient values.
- The lambda hyperparameter controls regularization strength: larger lambda means more shrinkage toward zero.
- Ridge is especially valuable when features are correlated (multicollinear), as it stabilizes the coefficient estimates.
- Unlike Lasso (L1), Ridge shrinks all coefficients toward zero but never sets them exactly to zero — it keeps all features in the model.
