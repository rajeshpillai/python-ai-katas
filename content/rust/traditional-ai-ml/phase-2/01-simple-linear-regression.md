# Simple Linear Regression

> Phase 2 — Supervised Learning: Regression | Kata 2.01

---

## Concept & Intuition

### What problem are we solving?

Simple linear regression is the foundational algorithm of supervised learning. Given a set of (x, y) data points, we want to find the straight line y = mx + b that best fits the data. "Best fit" means minimizing the sum of squared differences between predicted and actual values. This is the Ordinary Least Squares (OLS) approach.

Despite its simplicity, linear regression is remarkably powerful. It is interpretable (the slope tells you exactly how much y changes per unit change in x), fast to compute (closed-form solution, no iteration needed), and serves as a baseline against which more complex models are measured. If linear regression already explains 95% of the variance, a neural network that explains 96% may not be worth the added complexity.

The closed-form solution for simple linear regression has been known for over 200 years: the slope m = Cov(x,y) / Var(x) and the intercept b = mean(y) - m * mean(x). In this kata, we derive and implement this solution, then also implement gradient descent to find the same answer iteratively — building the foundation for all future optimization-based learning.

### Why naive approaches fail

You might try to find the best line by trial and error — picking random slopes and intercepts and checking the fit. But the parameter space is continuous and two-dimensional, making random search inefficient. You might try minimizing the sum of absolute errors instead of squared errors, but the absolute value function is not differentiable at zero, making optimization harder. Squared error has a smooth, convex loss surface with a unique minimum that can be found exactly (OLS) or iteratively (gradient descent).

### Mental models

- **Regression as projection**: The best-fit line is the projection of the target vector onto the space of possible linear predictions. OLS finds the closest point in that space.
- **Gradient descent as ball rolling downhill**: The loss surface is a bowl. Gradient descent starts the ball at a random point and follows the steepest downhill direction. The learning rate controls step size.
- **R-squared as explained variance**: R^2 tells you what fraction of the target's variance is explained by the model. R^2 = 0.8 means 80% of the variation in y is captured by the linear relationship with x.

### Visual explanations

```
  y                           Loss Surface (MSE)
  |       *                   Loss
  |     *  *  /               |  \         /
  |   *   * /                 |   \       /
  |  *   /                    |    \     /
  | * * / *                   |     \   /
  |  */  *                    |      \_/  <- minimum
  | /  *                      |       |
  |/ *                        +-------+------
  +------------- x              slope (m)

  Left: data with best-fit line    Right: loss as function of slope
```

---

## Hands-on Exploration

1. Generate synthetic data with a known linear relationship plus noise.
2. Implement the closed-form OLS solution for slope and intercept.
3. Implement gradient descent to find the same parameters iteratively.
4. Compute R-squared, MSE, and visualize the fit.

---

## Live Code

```rust
fn main() {
    println!("=== Simple Linear Regression ===\n");

    // Generate data: y = 3x + 10 + noise
    let true_slope = 3.0;
    let true_intercept = 10.0;
    let n = 20;

    let mut rng_state: u64 = 42;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for i in 0..n {
        let x = i as f64 * 0.5 + 1.0;
        rng_state = lcg(rng_state);
        let noise = (rng_state as f64 / u64::MAX as f64 - 0.5) * 6.0;
        let y = true_slope * x + true_intercept + noise;
        x_data.push(x);
        y_data.push(y);
    }

    println!("True relationship: y = {:.1}x + {:.1} + noise", true_slope, true_intercept);
    println!("Generated {} data points\n", n);

    // Method 1: Closed-form OLS solution
    println!("--- Method 1: Ordinary Least Squares (closed-form) ---");
    let (ols_slope, ols_intercept) = ols_fit(&x_data, &y_data);
    println!("  Slope:     {:.4} (true: {:.1})", ols_slope, true_slope);
    println!("  Intercept: {:.4} (true: {:.1})", ols_intercept, true_intercept);

    let ols_predictions: Vec<f64> = x_data.iter().map(|&x| ols_slope * x + ols_intercept).collect();
    let ols_mse = mse(&y_data, &ols_predictions);
    let ols_r2 = r_squared(&y_data, &ols_predictions);
    println!("  MSE:       {:.4}", ols_mse);
    println!("  R-squared: {:.4}", ols_r2);

    // Method 2: Gradient Descent
    println!("\n--- Method 2: Gradient Descent ---");
    let learning_rate = 0.001;
    let epochs = 1000;
    let (gd_slope, gd_intercept, loss_history) =
        gradient_descent(&x_data, &y_data, learning_rate, epochs);
    println!("  Learning rate: {}, Epochs: {}", learning_rate, epochs);
    println!("  Slope:     {:.4} (true: {:.1})", gd_slope, true_slope);
    println!("  Intercept: {:.4} (true: {:.1})", gd_intercept, true_intercept);

    let gd_predictions: Vec<f64> = x_data.iter().map(|&x| gd_slope * x + gd_intercept).collect();
    let gd_mse = mse(&y_data, &gd_predictions);
    let gd_r2 = r_squared(&y_data, &gd_predictions);
    println!("  MSE:       {:.4}", gd_mse);
    println!("  R-squared: {:.4}", gd_r2);

    // Loss convergence
    println!("\n--- Gradient Descent Loss Convergence ---");
    let checkpoints = vec![0, 1, 5, 10, 50, 100, 500, 999];
    for &epoch in &checkpoints {
        if epoch < loss_history.len() {
            let bar_len = ((loss_history[epoch] / loss_history[0]) * 40.0).min(40.0) as usize;
            let bar: String = "#".repeat(bar_len);
            println!("  Epoch {:>4}: MSE = {:>10.4} |{}", epoch, loss_history[epoch], bar);
        }
    }

    // ASCII scatter plot with regression line
    println!("\n--- Scatter Plot with Regression Line ---");
    ascii_scatter(&x_data, &y_data, ols_slope, ols_intercept);

    // Residual analysis
    println!("\n--- Residuals ---");
    let residuals: Vec<f64> = y_data.iter().zip(ols_predictions.iter())
        .map(|(y, yhat)| y - yhat)
        .collect();

    let res_mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let res_std = (residuals.iter().map(|r| (r - res_mean).powi(2)).sum::<f64>()
        / residuals.len() as f64).sqrt();
    println!("  Residual mean: {:.4} (should be ~0)", res_mean);
    println!("  Residual std:  {:.4}", res_std);

    kata_metric("ols_slope", ols_slope);
    kata_metric("ols_intercept", ols_intercept);
    kata_metric("ols_r_squared", ols_r2);
    kata_metric("ols_mse", ols_mse);
    kata_metric("gd_mse", gd_mse);
}

fn ols_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    for i in 0..x.len() {
        cov_xy += (x[i] - mean_x) * (y[i] - mean_y);
        var_x += (x[i] - mean_x).powi(2);
    }

    let slope = cov_xy / var_x;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

fn gradient_descent(
    x: &[f64],
    y: &[f64],
    lr: f64,
    epochs: usize,
) -> (f64, f64, Vec<f64>) {
    let n = x.len() as f64;
    let mut slope = 0.0;
    let mut intercept = 0.0;
    let mut loss_history = Vec::new();

    for _ in 0..epochs {
        let mut grad_slope = 0.0;
        let mut grad_intercept = 0.0;
        let mut total_loss = 0.0;

        for i in 0..x.len() {
            let pred = slope * x[i] + intercept;
            let error = pred - y[i];
            grad_slope += error * x[i];
            grad_intercept += error;
            total_loss += error * error;
        }

        grad_slope = 2.0 * grad_slope / n;
        grad_intercept = 2.0 * grad_intercept / n;
        total_loss /= n;

        slope -= lr * grad_slope;
        intercept -= lr * grad_intercept;
        loss_history.push(total_loss);
    }

    (slope, intercept, loss_history)
}

fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>() / actual.len() as f64
}

fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    let mean_y = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_res: f64 = actual.iter().zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    let ss_tot: f64 = actual.iter().map(|a| (a - mean_y).powi(2)).sum();
    if ss_tot < 1e-10 { return 0.0; }
    1.0 - ss_res / ss_tot
}

fn ascii_scatter(x: &[f64], y: &[f64], slope: f64, intercept: f64) {
    let height = 15;
    let width = 40;

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut grid = vec![vec![' '; width]; height];

    // Plot regression line
    for col in 0..width {
        let xv = x_min + (x_max - x_min) * col as f64 / (width - 1) as f64;
        let yv = slope * xv + intercept;
        let row = ((y_max - yv) / (y_max - y_min) * (height - 1) as f64).round() as i32;
        if row >= 0 && row < height as i32 {
            grid[row as usize][col] = '-';
        }
    }

    // Plot data points (overwrite line)
    for i in 0..x.len() {
        let col = ((x[i] - x_min) / (x_max - x_min) * (width - 1) as f64).round() as usize;
        let row = ((y_max - y[i]) / (y_max - y_min) * (height - 1) as f64).round() as i32;
        if row >= 0 && row < height as i32 && col < width {
            grid[row as usize][col] = '*';
        }
    }

    for row in &grid {
        let line: String = row.iter().collect();
        println!("  |{}", line);
    }
    println!("  +{}", "-".repeat(width));
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

- Simple linear regression finds the best straight line through data by minimizing the sum of squared errors.
- The closed-form OLS solution is exact and instant. Gradient descent converges to the same answer iteratively, which generalizes to more complex models.
- R-squared measures what fraction of the target's variance the model explains. It ranges from 0 (no explanatory power) to 1 (perfect fit).
- Residual analysis (checking that residuals are centered at zero with no pattern) validates whether the linear assumption holds.
