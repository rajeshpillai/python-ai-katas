# Overfitting vs Underfitting

> Phase 1 — What Does It Mean to Learn? | Kata 1.7

---

## Concept & Intuition

### What problem are we solving?

Overfitting and underfitting are the two fundamental failure modes of machine learning. A model that underfits is too simple to capture the patterns in the data — it has high bias. A model that overfits memorizes the training data, including its noise — it has high variance. The goal is to find the sweet spot: a model complex enough to capture the signal but not so complex that it also captures the noise.

Underfitting is easy to detect: the model performs poorly on both training and test data. Overfitting is more insidious: the model performs well on training data but poorly on test data. The gap between training and test performance is the hallmark of overfitting. This is why we always evaluate models on held-out data that the model has never seen during training.

The bias-variance tradeoff is the formal framework for understanding this tension. Bias is the error from wrong assumptions (model too simple). Variance is the error from sensitivity to small fluctuations in the training set (model too complex). Total error = bias^2 + variance + irreducible noise. You cannot minimize both simultaneously — reducing one typically increases the other.

### Why naive approaches fail

A common mistake is to evaluate a model only on its training data. Training loss always decreases with model complexity — but test loss decreases, reaches a minimum, and then increases. Without a test set, you cannot detect overfitting. This is why train/test splits (and cross-validation) are non-negotiable in machine learning.

### Mental models

- **Polynomial degree as complexity**: Fitting polynomials of increasing degree to noisy data illustrates the tradeoff perfectly. Degree 1 (line) underfits curves. Degree 15 on 10 points overfits wildly.
- **Memorization vs. generalization**: Overfitting is like a student who memorizes exam answers instead of understanding concepts — they ace the practice exam but fail the real one.
- **The U-curve**: Plot test error against model complexity. It forms a U shape. The bottom of the U is the sweet spot.

### Visual explanations

```
  error │
        │ ╲                    ╱
        │  ╲  test error      ╱
        │   ╲               ╱
        │    ╲   ╱─────╲  ╱
        │     ╲╱         ╲╱   ← sweet spot
        │      ─────────────── training error
        └──────────────────────── model complexity
          underfitting   │    overfitting
                    optimal
```

---

## Hands-on Exploration

1. Generate noisy data from a known function.
2. Fit polynomials of increasing degree and observe training vs. test error.
3. Identify the sweet spot where test error is minimized.

---

## Live Code

```rust
fn main() {
    // === Overfitting vs Underfitting ===
    // Finding the right model complexity.

    // Pseudo-random number generator
    let mut seed: u64 = 42;
    let mut rand_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };

    // True function: y = 0.5x^2 - 2x + 3
    let true_fn = |x: f64| -> f64 { 0.5 * x * x - 2.0 * x + 3.0 };
    let noise_std = 1.5;

    // Generate training data
    let n_train = 15;
    let n_test = 30;

    let x_train: Vec<f64> = (0..n_train)
        .map(|i| -2.0 + 8.0 * i as f64 / (n_train - 1) as f64)
        .collect();
    let y_train: Vec<f64> = x_train.iter()
        .map(|&x| true_fn(x) + rand_f64() * noise_std)
        .collect();

    let x_test: Vec<f64> = (0..n_test)
        .map(|i| -2.0 + 8.0 * i as f64 / (n_test - 1) as f64)
        .collect();
    let y_test: Vec<f64> = x_test.iter()
        .map(|&x| true_fn(x) + rand_f64() * noise_std)
        .collect();

    println!("=== Overfitting vs Underfitting ===\n");
    println!("  True function: y = 0.5x² - 2x + 3");
    println!("  Training set: {} points, Test set: {} points", n_train, n_test);
    println!("  Noise std: {}\n", noise_std);

    // === Fit polynomials of different degrees ===
    // For a polynomial of degree d: y = c0 + c1*x + c2*x^2 + ... + cd*x^d
    // We solve: X * c = y (via normal equation)

    // Build Vandermonde matrix
    let vandermonde = |xs: &[f64], degree: usize| -> Vec<Vec<f64>> {
        xs.iter().map(|&x| {
            (0..=degree).map(|d| x.powi(d as i32)).collect()
        }).collect()
    };

    // Solve via Gaussian elimination (general case)
    let solve_linear = |a_orig: &Vec<Vec<f64>>, b_orig: &Vec<f64>| -> Option<Vec<f64>> {
        let n = b_orig.len();
        let m = a_orig[0].len();
        if n < m { return None; }

        // X^T X c = X^T y (for overdetermined systems)
        let mut ata = vec![vec![0.0; m]; m];
        let mut atb = vec![0.0; m];
        for i in 0..n {
            for j in 0..m {
                for k in 0..m {
                    ata[j][k] += a_orig[i][j] * a_orig[i][k];
                }
                atb[j] += a_orig[i][j] * b_orig[i];
            }
        }

        // Gaussian elimination with partial pivoting
        let mut a = ata;
        let mut b = atb;
        for col in 0..m {
            // Find pivot
            let mut max_val = a[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..m {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    max_row = row;
                }
            }
            a.swap(col, max_row);
            b.swap(col, max_row);

            if a[col][col].abs() < 1e-12 { return None; }

            for row in (col + 1)..m {
                let factor = a[row][col] / a[col][col];
                for k in col..m {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back substitution
        let mut x = vec![0.0; m];
        for i in (0..m).rev() {
            x[i] = b[i];
            for j in (i + 1)..m {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }
        Some(x)
    };

    // Evaluate polynomial
    let poly_eval = |x: f64, coeffs: &[f64]| -> f64 {
        coeffs.iter().enumerate()
            .map(|(d, &c)| c * x.powi(d as i32))
            .sum()
    };

    // Compute MSE
    let mse = |xs: &[f64], ys: &[f64], coeffs: &[f64]| -> f64 {
        let n = xs.len() as f64;
        xs.iter().zip(ys.iter())
            .map(|(&x, &y)| {
                let pred = poly_eval(x, coeffs);
                (y - pred) * (y - pred)
            })
            .sum::<f64>() / n
    };

    println!("=== Polynomial Fits of Increasing Degree ===\n");
    println!("  {:>6} {:>12} {:>12} {:>8}",
        "Degree", "Train MSE", "Test MSE", "Status");
    println!("  {:->6} {:->12} {:->12} {:->8}", "", "", "", "");

    let mut results: Vec<(usize, f64, f64)> = Vec::new();
    let mut best_test_mse = f64::INFINITY;
    let mut best_degree = 0;

    for degree in 1..=12 {
        let x_mat = vandermonde(&x_train, degree);
        if let Some(coeffs) = solve_linear(&x_mat, &y_train) {
            let train_mse = mse(&x_train, &y_train, &coeffs);
            let test_mse = mse(&x_test, &y_test, &coeffs);

            let status = if test_mse < best_test_mse * 1.05 && degree <= 4 {
                "good"
            } else if train_mse < 0.5 && test_mse > train_mse * 3.0 {
                "OVERFIT"
            } else if train_mse > 3.0 {
                "UNDERFIT"
            } else {
                ""
            };

            if test_mse < best_test_mse {
                best_test_mse = test_mse;
                best_degree = degree;
            }

            println!("  {:>6} {:>12.4} {:>12.4} {:>8}",
                degree, train_mse, test_mse, status);
            results.push((degree, train_mse, test_mse));
        }
    }

    println!("\n  Best test MSE at degree {}\n", best_degree);

    // === Visualize the bias-variance tradeoff ===
    println!("=== The Bias-Variance Tradeoff ===\n");

    let max_test = results.iter().map(|r| r.2).fold(0.0_f64, f64::max).min(50.0);

    for &(degree, train_mse, test_mse) in &results {
        let train_bar = (train_mse / max_test * 30.0).min(30.0) as usize;
        let test_bar = (test_mse / max_test * 30.0).min(30.0) as usize;
        let marker = if degree == best_degree { " ← sweet spot" } else { "" };
        println!("  degree {}:", degree);
        println!("    train: |{}| {:.2}", "█".repeat(train_bar), train_mse);
        println!("    test:  |{}| {:.2}{}",
            "░".repeat(test_bar), test_mse.min(999.0), marker);
    }

    println!();
    println!("  Training error always decreases with complexity.");
    println!("  Test error decreases, then INCREASES — the hallmark of overfitting.\n");

    // === Predictions comparison ===
    println!("=== Prediction Comparison at Sample Points ===\n");
    println!("  {:>4} {:>8} {:>10} {:>10} {:>10}",
        "x", "true", "degree 1", "degree 2", "degree 10");
    println!("  {:->4} {:->8} {:->10} {:->10} {:->10}", "", "", "", "", "");

    // Fit the specific degrees we want to compare
    let coeffs_1 = solve_linear(&vandermonde(&x_train, 1), &y_train).unwrap();
    let coeffs_2 = solve_linear(&vandermonde(&x_train, 2), &y_train).unwrap();
    let coeffs_10 = solve_linear(&vandermonde(&x_train, 10), &y_train);

    for &x in &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] {
        let true_y = true_fn(x);
        let pred_1 = poly_eval(x, &coeffs_1);
        let pred_2 = poly_eval(x, &coeffs_2);
        let pred_10 = match &coeffs_10 {
            Some(c) => format!("{:>10.2}", poly_eval(x, c)),
            None => "  N/A".to_string(),
        };
        println!("  {:>4.0} {:>8.2} {:>10.2} {:>10.2} {}",
            x, true_y, pred_1, pred_2, pred_10);
    }

    println!();
    println!("  Degree 1: too simple (underfitting — misses the curvature)");
    println!("  Degree 2: just right (captures the quadratic pattern)");
    println!("  Degree 10: too complex (overfitting — wild oscillations)");

    println!();
    println!("Key insight: Model complexity must match data complexity.");
    println!("Too simple → underfitting. Too complex → overfitting.");
    println!("Always evaluate on held-out test data.");
}
```

---

## Key Takeaways

- Underfitting (high bias) means the model is too simple to capture the pattern; overfitting (high variance) means it memorizes noise.
- Training error always decreases with model complexity, but test error forms a U-shape — the minimum is the sweet spot.
- The bias-variance tradeoff is the fundamental tension: reducing bias increases variance and vice versa.
- Always evaluate on held-out test data to detect overfitting — never rely solely on training performance.
