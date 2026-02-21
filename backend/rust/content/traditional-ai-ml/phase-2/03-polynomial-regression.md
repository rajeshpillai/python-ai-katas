# Polynomial Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.03

---

## Concept & Intuition

### What problem are we solving?

Many real-world relationships are not linear. The trajectory of a thrown ball is parabolic. Drug dosage effectiveness often has a sweet spot (inverted U-shape). Population growth follows exponential curves before plateauing. When the data shows curvature, a straight line gives a poor fit no matter how you adjust the slope. Polynomial regression fits curves by including polynomial terms (x^2, x^3, etc.) as additional features.

The clever insight is that polynomial regression is still linear regression — just with engineered features. If we create new features x^2, x^3, etc., and feed them into the same linear regression machinery, we get a polynomial fit. This is our first example of feature engineering: transforming the input space to make a linear model capture nonlinear relationships.

In this kata, we implement polynomial regression by generating polynomial features and fitting them with OLS. We also explore the critical concept of model complexity: higher-degree polynomials fit the training data better but may oscillate wildly between data points (overfitting).

### Why naive approaches fail

Increasing the polynomial degree always reduces training error — a degree-N polynomial can perfectly fit N+1 points. But this perfect fit on training data means the model is memorizing noise rather than learning the underlying relationship. A degree-19 polynomial fit to 20 points will oscillate wildly between data points, making absurd predictions. The challenge is finding the right degree that captures the true relationship without fitting noise.

### Mental models

- **Feature engineering as lifting**: By adding x^2 and x^3 as features, we lift the data from a 1D line to a 3D space where the nonlinear relationship becomes a linear hyperplane.
- **Bias-variance tradeoff preview**: Low-degree polynomials underfit (high bias, low variance). High-degree polynomials overfit (low bias, high variance). The sweet spot balances both.
- **Occam's razor**: Among models with similar performance, prefer the simpler one. A degree-2 polynomial that fits nearly as well as degree-8 is almost always better.

### Visual explanations

```
  Degree 1 (underfit):    Degree 2 (good fit):     Degree 8 (overfit):

  y |     /               y |   * *                 y |  *  /\  *
    |   / *                 | *     *                 |*  /  \ *
    | /  *  *               |*       *                |  /    \/
    |/ *                    |         *               | /   *
    +-------- x             +---------*--- x          +--------- x

  Training MSE: high       Training MSE: low         Training MSE: ~0
  Test MSE: high           Test MSE: low             Test MSE: VERY HIGH
```

---

## Hands-on Exploration

1. Generate nonlinear data (quadratic with noise).
2. Fit polynomials of increasing degree (1, 2, 3, 5, 8).
3. Compare training error and observe how it decreases with degree.
4. Split data into train/test and observe the overfitting phenomenon.

---

## Live Code

```rust
fn main() {
    println!("=== Polynomial Regression ===\n");

    // Generate data: y = 2x^2 - 3x + 5 + noise
    let n = 25;
    let mut rng = 42u64;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for i in 0..n {
        let x = -3.0 + 6.0 * i as f64 / (n - 1) as f64;
        rng = lcg(rng);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 8.0;
        let y = 2.0 * x * x - 3.0 * x + 5.0 + noise;
        x_data.push(x);
        y_data.push(y);
    }

    println!("True function: y = 2x^2 - 3x + 5 + noise");
    println!("Generated {} data points\n", n);

    // Train-test split (first 18 train, last 7 test)
    let train_n = 18;
    let x_train: Vec<f64> = x_data[..train_n].to_vec();
    let y_train: Vec<f64> = y_data[..train_n].to_vec();
    let x_test: Vec<f64> = x_data[train_n..].to_vec();
    let y_test: Vec<f64> = y_data[train_n..].to_vec();

    println!("Train: {} samples, Test: {} samples\n", train_n, n - train_n);

    // Fit polynomials of different degrees
    let degrees = vec![1, 2, 3, 5, 8];

    println!("{:<10} {:>12} {:>12} {:>12}", "Degree", "Train MSE", "Test MSE", "Coefficients");
    println!("{}", "-".repeat(60));

    let mut results: Vec<(usize, f64, f64, Vec<f64>)> = Vec::new();

    for &degree in &degrees {
        let x_train_poly = polynomial_features(&x_train, degree);
        let x_test_poly = polynomial_features(&x_test, degree);

        let coeffs = ols_fit(&x_train_poly, &y_train);

        let train_preds = predict(&x_train_poly, &coeffs);
        let test_preds = predict(&x_test_poly, &coeffs);

        let train_mse = mse(&y_train, &train_preds);
        let test_mse = mse(&y_test, &test_preds);

        let coeff_str: String = coeffs.iter()
            .map(|c| format!("{:.2}", c))
            .collect::<Vec<_>>()
            .join(", ");

        println!("{:<10} {:>12.4} {:>12.4} [{}]", degree, train_mse, test_mse, coeff_str);
        results.push((degree, train_mse, test_mse, coeffs.clone()));
    }

    // Visualize the best fit (degree 2)
    println!("\n--- ASCII Plot: Degree 2 Fit ---");
    let best = &results[1]; // degree 2
    let x_poly = polynomial_features(&x_data, 2);
    let all_preds = predict(&x_poly, &best.3);
    ascii_plot(&x_data, &y_data, &all_preds);

    // Overfitting visualization
    println!("\n--- Overfitting Analysis ---");
    println!("{:<10} {:>12} {:>12} {:>12}", "Degree", "Train MSE", "Test MSE", "Ratio");
    println!("{}", "-".repeat(46));
    for &(degree, train_mse, test_mse, _) in &results {
        let ratio = if train_mse > 1e-10 { test_mse / train_mse } else { f64::INFINITY };
        let indicator = if ratio > 5.0 { " << OVERFIT" } else if ratio > 2.0 { " < warning" } else { "" };
        println!("{:<10} {:>12.4} {:>12.4} {:>11.2}x{}",
            degree, train_mse, test_mse, ratio, indicator);
    }

    // Polynomial feature details for degree 2
    println!("\n--- Degree 2: Coefficient Interpretation ---");
    let d2_coeffs = &results[1].3;
    println!("  y = {:.2} + ({:.2})x + ({:.2})x^2", d2_coeffs[0], d2_coeffs[1], d2_coeffs[2]);
    println!("  True: y = 5.0 + (-3.0)x + (2.0)x^2");
    println!("  Coefficient errors:");
    let true_coeffs = vec![5.0, -3.0, 2.0];
    for (i, (&estimated, &actual)) in d2_coeffs.iter().zip(true_coeffs.iter()).enumerate() {
        let label = match i { 0 => "intercept", 1 => "x", _ => "x^2" };
        println!("    {}: estimated={:.3}, true={:.1}, error={:.3}",
            label, estimated, actual, (estimated - actual).abs());
    }

    // Metrics
    kata_metric("degree2_train_mse", results[1].1);
    kata_metric("degree2_test_mse", results[1].2);
    kata_metric("degree8_train_mse", results[4].1);
    kata_metric("degree8_test_mse", results[4].2);
    kata_metric("best_degree", 2.0);
}

fn polynomial_features(x: &[f64], degree: usize) -> Vec<Vec<f64>> {
    x.iter().map(|&xi| {
        let mut row = vec![1.0]; // intercept
        for d in 1..=degree {
            row.push(xi.powi(d as i32));
        }
        row
    }).collect()
}

fn ols_fit(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let xt = transpose(x);
    let xtx = mat_mul(&xt, x);
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

fn ascii_plot(x: &[f64], y_actual: &[f64], y_pred: &[f64]) {
    let height = 12;
    let width = 50;

    let all_y: Vec<f64> = y_actual.iter().chain(y_pred.iter()).cloned().collect();
    let y_min = all_y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut grid = vec![vec![' '; width]; height];

    // Plot predictions (curve)
    for col in 0..width {
        let xv = x_min + (x_max - x_min) * col as f64 / (width - 1) as f64;
        let yv = y_pred.iter().zip(x.iter())
            .min_by_key(|(_, xi)| ((xi - xv).abs() * 1000.0) as i64)
            .map(|(yp, _)| *yp)
            .unwrap_or(0.0);
        let row = ((y_max - yv) / (y_max - y_min) * (height - 1) as f64).round() as i32;
        if row >= 0 && row < height as i32 {
            grid[row as usize][col] = '-';
        }
    }

    // Plot actual data points
    for i in 0..x.len() {
        let col = ((x[i] - x_min) / (x_max - x_min) * (width - 1) as f64).round() as usize;
        let row = ((y_max - y_actual[i]) / (y_max - y_min) * (height - 1) as f64).round() as i32;
        if row >= 0 && row < height as i32 && col < width {
            grid[row as usize][col] = '*';
        }
    }

    for row in &grid {
        let line: String = row.iter().collect();
        println!("  |{}", line);
    }
    println!("  +{}", "-".repeat(width));
    println!("   * = actual data,  - = polynomial fit");
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m.len();
    let cols = m[0].len();
    let mut result = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = m[i][j];
        }
    }
    result
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();
    let mut result = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| {
        row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

fn invert_matrix(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = m[i][j]; }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() { max_row = row; }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 { continue; }
        for j in 0..(2 * n) { aug[col][j] /= pivot; }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) { aug[row][j] -= factor * aug[col][j]; }
            }
        }
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

- Polynomial regression captures nonlinear relationships by adding polynomial features (x^2, x^3, ...) and fitting with standard linear regression.
- Higher-degree polynomials always reduce training error but may dramatically increase test error — this is overfitting.
- The right polynomial degree balances underfitting (too simple) and overfitting (too complex). Comparing train vs. test error reveals overfitting.
- Polynomial regression is our first concrete example of the bias-variance tradeoff, which is central to all of machine learning.
