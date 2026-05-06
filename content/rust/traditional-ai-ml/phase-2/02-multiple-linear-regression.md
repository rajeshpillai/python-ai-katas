# Multiple Linear Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.02

---

## Concept & Intuition

### What problem are we solving?

Real-world outcomes depend on multiple factors, not just one. House prices depend on square footage, number of bedrooms, age, location, and more. Multiple linear regression extends simple linear regression to handle multiple input features: y = b0 + b1*x1 + b2*x2 + ... + bn*xn. Each coefficient bi represents the effect of feature xi on the target, holding all other features constant.

The key insight is that multiple regression disentangles the effects of correlated features. Square footage and number of bedrooms are correlated (bigger houses tend to have more bedrooms), but multiple regression separates their individual contributions. The coefficient for bedrooms answers: "How much does one extra bedroom affect price, for houses of the same square footage?"

In this kata, we implement multiple linear regression using the Normal Equation (the matrix-based closed-form solution) and gradient descent. Since we cannot use external crates, we implement the necessary matrix operations from scratch.

### Why naive approaches fail

Running separate simple regressions for each feature ignores the correlations between features. If square footage and bedrooms are correlated, their separate regression coefficients will be biased — each one absorbs some of the other's effect. Multiple regression handles this by fitting all coefficients simultaneously. However, when features are highly correlated (multicollinearity), the coefficient estimates become unstable, which motivates regularization techniques in later katas.

### Mental models

- **Hyperplane instead of line**: With two features, the model is a plane in 3D space. With more features, it is a hyperplane in N-dimensional space. We are always finding the flat surface that best fits the data.
- **Normal Equation as direct solve**: X'Xb = X'y. Solving this system of equations gives us the optimal coefficients directly. No iteration needed.
- **Feature importance from coefficients**: After standardizing features, the coefficient magnitudes indicate relative importance. The feature with the largest absolute coefficient has the strongest influence.

### Visual explanations

```
  Simple Regression:           Multiple Regression:
  y = m*x + b                 y = b0 + b1*x1 + b2*x2

  y |      /                   y
    |    /                     |    /----/
    |  /                       |  /----/   (a plane in 3D)
    |/                         |/----/
    +--------- x               +----------/--- x1
                              / x2

  1 feature -> line            2 features -> plane
                               N features -> hyperplane
```

---

## Hands-on Exploration

1. Create a multi-feature dataset with a known linear relationship.
2. Implement matrix operations: transpose, multiply, invert.
3. Solve the Normal Equation to find optimal coefficients.
4. Compare with gradient descent on the same problem.

---

## Live Code

```rust
fn main() {
    println!("=== Multiple Linear Regression ===\n");

    // Dataset: house price prediction
    // Features: sqft, bedrooms, bathrooms, age_years
    // Target: price (thousands)
    let feature_names = vec!["sqft", "bedrooms", "bathrooms", "age_years"];

    let features: Vec<Vec<f64>> = vec![
        vec![1400.0, 3.0, 2.0, 10.0],
        vec![1600.0, 3.0, 2.0, 5.0],
        vec![1700.0, 4.0, 2.5, 8.0],
        vec![1100.0, 2.0, 1.0, 25.0],
        vec![2100.0, 4.0, 3.0, 2.0],
        vec![1500.0, 3.0, 2.0, 15.0],
        vec![1800.0, 4.0, 2.5, 3.0],
        vec![950.0,  2.0, 1.0, 30.0],
        vec![2400.0, 5.0, 3.5, 1.0],
        vec![1300.0, 3.0, 1.5, 20.0],
        vec![1550.0, 3.0, 2.0, 12.0],
        vec![1900.0, 4.0, 3.0, 7.0],
        vec![2200.0, 5.0, 3.0, 4.0],
        vec![1200.0, 2.0, 1.5, 18.0],
        vec![1650.0, 3.0, 2.5, 9.0],
    ];

    let targets: Vec<f64> = vec![
        250.0, 310.0, 340.0, 180.0, 420.0,
        275.0, 365.0, 150.0, 500.0, 230.0,
        285.0, 380.0, 450.0, 210.0, 305.0,
    ];

    let n = features.len();
    let p = features[0].len();
    println!("Dataset: {} samples, {} features\n", n, p);

    // Method 1: Normal Equation
    // Add intercept column (column of 1s)
    let mut x_with_intercept: Vec<Vec<f64>> = features.iter().map(|row| {
        let mut new_row = vec![1.0]; // intercept
        new_row.extend(row);
        new_row
    }).collect();

    println!("--- Method 1: Normal Equation ---");
    let coeffs = normal_equation(&x_with_intercept, &targets);

    println!("  Intercept: {:.4}", coeffs[0]);
    for (i, name) in feature_names.iter().enumerate() {
        println!("  {} coefficient: {:.4}", name, coeffs[i + 1]);
    }

    let predictions = predict(&x_with_intercept, &coeffs);
    let mse_val = mse(&targets, &predictions);
    let r2_val = r_squared(&targets, &predictions);

    println!("  MSE: {:.4}", mse_val);
    println!("  R-squared: {:.4}", r2_val);

    // Method 2: Gradient Descent (with feature scaling)
    println!("\n--- Method 2: Gradient Descent ---");
    let (scaled_features, means, stds) = standardize(&features);
    let mut scaled_x: Vec<Vec<f64>> = scaled_features.iter().map(|row| {
        let mut new_row = vec![1.0];
        new_row.extend(row);
        new_row
    }).collect();

    let (gd_coeffs, loss_hist) = gradient_descent_multi(&scaled_x, &targets, 0.01, 2000);

    // Convert coefficients back to original scale
    let mut orig_coeffs = vec![gd_coeffs[0]];
    for i in 0..p {
        if stds[i].abs() > 1e-10 {
            let orig_coeff = gd_coeffs[i + 1] / stds[i];
            orig_coeffs.push(orig_coeff);
            orig_coeffs[0] -= orig_coeff * means[i];
        } else {
            orig_coeffs.push(0.0);
        }
    }

    println!("  Intercept: {:.4}", orig_coeffs[0]);
    for (i, name) in feature_names.iter().enumerate() {
        println!("  {} coefficient: {:.4}", name, orig_coeffs[i + 1]);
    }

    let gd_preds = predict(&x_with_intercept, &orig_coeffs);
    let gd_mse = mse(&targets, &gd_preds);
    let gd_r2 = r_squared(&targets, &gd_preds);
    println!("  MSE: {:.4}", gd_mse);
    println!("  R-squared: {:.4}", gd_r2);

    // Feature importance (using standardized coefficients)
    println!("\n--- Feature Importance (standardized coefficients) ---");
    let mut importance: Vec<(&str, f64)> = feature_names.iter()
        .enumerate()
        .map(|(i, &name)| (name, gd_coeffs[i + 1].abs()))
        .collect();
    importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, coeff) in &importance {
        let bar_len = (coeff / importance[0].1 * 30.0) as usize;
        println!("  {:<12} |{}", name, "#".repeat(bar_len));
    }

    // Predictions vs actuals
    println!("\n--- Predictions vs Actuals ---");
    println!("{:<10} {:>10} {:>10} {:>10}", "Sample", "Actual", "Predicted", "Error");
    println!("{}", "-".repeat(40));
    for i in 0..n {
        let error = targets[i] - predictions[i];
        println!("{:<10} {:>10.1} {:>10.1} {:>10.1}", i, targets[i], predictions[i], error);
    }

    kata_metric("normal_eq_r_squared", r2_val);
    kata_metric("normal_eq_mse", mse_val);
    kata_metric("gradient_descent_r_squared", gd_r2);
    kata_metric("n_features", p as f64);
}

fn normal_equation(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    // beta = (X'X)^-1 X'y
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

fn gradient_descent_multi(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    epochs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.len() as f64;
    let p = x[0].len();
    let mut coeffs = vec![0.0; p];
    let mut loss_history = Vec::new();

    for epoch in 0..epochs {
        let mut gradients = vec![0.0; p];
        let mut total_loss = 0.0;

        for i in 0..x.len() {
            let pred: f64 = x[i].iter().zip(coeffs.iter()).map(|(xi, ci)| xi * ci).sum();
            let error = pred - y[i];
            total_loss += error * error;

            for j in 0..p {
                gradients[j] += error * x[i][j];
            }
        }

        for j in 0..p {
            coeffs[j] -= lr * 2.0 * gradients[j] / n;
        }
        loss_history.push(total_loss / n);
    }

    (coeffs, loss_history)
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

    // Build augmented matrix [M | I]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }

        // Scale pivot row
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse
    aug.iter().map(|row| row[n..].to_vec()).collect()
}

fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>() / actual.len() as f64
}

fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    let mean_y = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_res: f64 = actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2)).sum();
    let ss_tot: f64 = actual.iter().map(|a| (a - mean_y).powi(2)).sum();
    if ss_tot < 1e-10 { return 0.0; }
    1.0 - ss_res / ss_tot
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Multiple linear regression extends simple regression to multiple features, fitting a hyperplane instead of a line.
- The Normal Equation provides an exact closed-form solution but requires matrix inversion, which can be slow for very large feature sets.
- Gradient descent is iterative but scales better to high dimensions. Feature scaling is essential for it to converge efficiently.
- Standardized coefficients reveal relative feature importance: which features contribute most to the prediction after accounting for scale differences.
