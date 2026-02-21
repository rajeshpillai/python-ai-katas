# Regression Diagnostics

> Phase 2 — Supervised Learning: Regression | Kata 2.07

---

## Concept & Intuition

### What problem are we solving?

Fitting a regression model is only half the job. Before trusting a model's predictions, you need to verify that the model's assumptions are met and diagnose potential problems. Regression diagnostics are systematic checks that reveal whether your model is trustworthy: are residuals randomly distributed? Are there influential outliers warping the fit? Is the relationship truly linear? Do errors have constant variance (homoscedasticity)?

A model can have a high R-squared and still be deeply flawed. A classic example: fitting a linear model to quadratic data gives a decent R-squared but systematically wrong predictions (under-predicting in the middle, over-predicting at the extremes). Only residual analysis reveals this. Regression diagnostics transform "the model fits" into "the model fits correctly for the right reasons."

In this kata, we implement a comprehensive diagnostic toolkit: residual analysis, leverage and influence detection, normality tests, and heteroscedasticity checks.

### Why naive approaches fail

Relying solely on R-squared or MSE to evaluate a model misses critical problems. Anscombe's quartet famously demonstrates four datasets with identical R-squared, slopes, and intercepts — but wildly different patterns that only residual plots reveal. Without diagnostics, you might deploy a model that performs well on average metrics but fails catastrophically on specific subsets of data.

### Mental models

- **Residuals as a diagnostic window**: If the model is correct, residuals should be random noise — no patterns, no trends, no structure. Any pattern in the residuals indicates something the model missed.
- **Leverage vs. influence**: Leverage measures how far a point's features are from the center. Influence measures how much removing a point changes the model. High leverage + high residual = high influence = a point you should investigate.
- **Cook's distance as a summary**: Cook's distance combines leverage and residual magnitude into a single number measuring each point's influence on the entire model.

### Visual explanations

```
  Good residuals:              Bad residuals (nonlinearity):
  residual                     residual
    |  .  .  .                   |        . .
    |. . . .  .                  |  .  .
  --+------------ x           --+------------- x
    | .  . . .                   |          . .
    |.  .   .                    |  .   .

  Random scatter = good          Pattern = model misspecification

  Bad residuals (heteroscedasticity):
  residual
    |              . .
    |         .  .
  --+--.-.---------- x
    |         .  .
    |              . .
  Variance increases with x = problematic
```

---

## Hands-on Exploration

1. Fit a linear regression model to data with known issues.
2. Compute and analyze residuals: look for patterns, non-normality, heteroscedasticity.
3. Compute leverage scores and Cook's distance to identify influential points.
4. Apply corrective actions and compare the improved model.

---

## Live Code

```rust
fn main() {
    println!("=== Regression Diagnostics ===\n");

    // Dataset with some issues: nonlinearity, outlier, heteroscedasticity hint
    let x_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    ];

    // y has a slight quadratic component + one outlier at index 15
    let mut rng = 42u64;
    let y_data: Vec<f64> = x_data.iter().enumerate().map(|(i, &x)| {
        rng = lcg(rng);
        let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 3.0;
        let y = 2.0 * x + 0.1 * x * x + 5.0 + noise;
        if i == 15 { y + 25.0 } else { y } // inject outlier
    }).collect();

    let n = x_data.len();

    // Fit linear regression
    let (slope, intercept) = ols_fit(&x_data, &y_data);
    println!("--- Linear Model: y = {:.3}x + {:.3} ---\n", slope, intercept);

    let predictions: Vec<f64> = x_data.iter().map(|&x| slope * x + intercept).collect();
    let residuals: Vec<f64> = y_data.iter().zip(predictions.iter())
        .map(|(y, yhat)| y - yhat).collect();

    // 1. Residual Analysis
    println!("--- 1. Residual Analysis ---");
    let res_mean = residuals.iter().sum::<f64>() / n as f64;
    let res_std = (residuals.iter().map(|r| (r - res_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    println!("  Residual mean: {:.4} (should be ~0)", res_mean);
    println!("  Residual std:  {:.4}", res_std);

    // Standardized residuals
    let std_residuals: Vec<f64> = residuals.iter().map(|r| (r - res_mean) / res_std).collect();

    println!("\n  Residuals by observation:");
    println!("  {:<6} {:>8} {:>8} {:>8} {:>8} {:>12}",
        "Idx", "x", "y", "y_hat", "resid", "std_resid");
    println!("  {}", "-".repeat(56));
    for i in 0..n {
        let flag = if std_residuals[i].abs() > 2.0 { " <<" } else { "" };
        println!("  {:<6} {:>8.1} {:>8.1} {:>8.1} {:>8.2} {:>10.3}{}",
            i, x_data[i], y_data[i], predictions[i], residuals[i], std_residuals[i], flag);
    }

    // 2. Normality of residuals (Jarque-Bera style)
    println!("\n--- 2. Normality Check ---");
    let skewness = residuals.iter().map(|r| ((r - res_mean) / res_std).powi(3)).sum::<f64>() / n as f64;
    let kurtosis = residuals.iter().map(|r| ((r - res_mean) / res_std).powi(4)).sum::<f64>() / n as f64;
    println!("  Skewness: {:.4} (normal: 0)", skewness);
    println!("  Kurtosis: {:.4} (normal: 3)", kurtosis);
    let jb = n as f64 / 6.0 * (skewness.powi(2) + (kurtosis - 3.0).powi(2) / 4.0);
    println!("  Jarque-Bera statistic: {:.4} (smaller is more normal)", jb);

    // 3. Heteroscedasticity check
    println!("\n--- 3. Heteroscedasticity Check ---");
    let half = n / 2;
    let var_first: f64 = residuals[..half].iter().map(|r| r * r).sum::<f64>() / half as f64;
    let var_second: f64 = residuals[half..].iter().map(|r| r * r).sum::<f64>() / (n - half) as f64;
    let variance_ratio = var_second / var_first;
    println!("  Variance of residuals (first half):  {:.4}", var_first);
    println!("  Variance of residuals (second half): {:.4}", var_second);
    println!("  Ratio: {:.4} (1.0 = homoscedastic)", variance_ratio);
    if variance_ratio > 2.0 || variance_ratio < 0.5 {
        println!("  WARNING: Possible heteroscedasticity detected!");
    }

    // 4. Leverage and Influence
    println!("\n--- 4. Leverage and Cook's Distance ---");
    let leverages = compute_leverage(&x_data);
    let cooks = compute_cooks_distance(&residuals, &leverages, n, 2);
    let avg_leverage = 2.0 / n as f64;

    println!("  Average leverage: {:.4}", avg_leverage);
    println!("  High leverage threshold (2*p/n): {:.4}", 2.0 * 2.0 / n as f64);
    println!("  Cook's distance threshold (4/n): {:.4}\n", 4.0 / n as f64);

    let cooks_threshold = 4.0 / n as f64;
    let leverage_threshold = 2.0 * 2.0 / n as f64;

    println!("  {:<6} {:>10} {:>12} {:>10}", "Idx", "Leverage", "Cook's D", "Flags");
    println!("  {}", "-".repeat(42));
    for i in 0..n {
        let mut flags = String::new();
        if leverages[i] > leverage_threshold { flags.push_str("HiLev "); }
        if cooks[i] > cooks_threshold { flags.push_str("HiInfluence "); }
        if std_residuals[i].abs() > 2.0 { flags.push_str("HiResid "); }
        if !flags.is_empty() {
            println!("  {:<6} {:>10.4} {:>12.4} {}", i, leverages[i], cooks[i], flags);
        }
    }

    // 5. Nonlinearity test: residuals vs. fitted
    println!("\n--- 5. Nonlinearity Detection ---");
    let corr_resid_x = correlation(&x_data, &residuals);
    let resid_sq: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    let corr_resid_sq_x = correlation(&x_data, &resid_sq);
    println!("  Corr(residuals, x):    {:.4} (should be ~0)", corr_resid_x);
    println!("  Corr(residuals^2, x):  {:.4} (nonzero suggests nonlinearity)", corr_resid_sq_x);

    // 6. Model comparison: Linear vs Quadratic
    println!("\n--- 6. Model Improvement: Add x^2 term ---");
    let x_poly: Vec<Vec<f64>> = x_data.iter().map(|&x| vec![1.0, x, x * x]).collect();
    let poly_coeffs = ols_multi(&x_poly, &y_data);
    let poly_preds: Vec<f64> = x_poly.iter().map(|row|
        row.iter().zip(poly_coeffs.iter()).map(|(x, c)| x * c).sum()
    ).collect();
    let poly_residuals: Vec<f64> = y_data.iter().zip(poly_preds.iter())
        .map(|(y, yhat)| y - yhat).collect();

    let linear_r2 = r_squared(&y_data, &predictions);
    let poly_r2 = r_squared(&y_data, &poly_preds);
    let linear_mse = mse(&y_data, &predictions);
    let poly_mse = mse(&y_data, &poly_preds);

    println!("  Linear: R^2={:.4}, MSE={:.4}", linear_r2, linear_mse);
    println!("  Quadratic: R^2={:.4}, MSE={:.4}", poly_r2, poly_mse);
    println!("  Improvement: {:.1}% reduction in MSE",
        (1.0 - poly_mse / linear_mse) * 100.0);

    kata_metric("r_squared_linear", linear_r2);
    kata_metric("r_squared_quadratic", poly_r2);
    kata_metric("residual_skewness", skewness);
    kata_metric("variance_ratio", variance_ratio);
    kata_metric("jarque_bera", jb);
}

fn ols_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for i in 0..x.len() {
        cov += (x[i] - mx) * (y[i] - my);
        var += (x[i] - mx).powi(2);
    }
    let slope = cov / var;
    (slope, my - slope * mx)
}

fn ols_multi(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let xt = transpose(x);
    let xtx = mat_mul(&xt, x);
    let xty = mat_vec_mul(&xt, y);
    let inv = invert_matrix(&xtx);
    mat_vec_mul(&inv, &xty)
}

fn compute_leverage(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|xi| (xi - mx).powi(2)).sum();
    x.iter().map(|xi| 1.0 / n + (xi - mx).powi(2) / var).collect()
}

fn compute_cooks_distance(residuals: &[f64], leverages: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mse_val: f64 = residuals.iter().map(|r| r * r).sum::<f64>() / (n - p) as f64;
    (0..n).map(|i| {
        let h = leverages[i];
        residuals[i].powi(2) * h / (p as f64 * mse_val * (1.0 - h).powi(2))
    }).collect()
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
    if vx < 1e-10 || vy < 1e-10 { return 0.0; }
    cov / (vx.sqrt() * vy.sqrt())
}

fn mse(a: &[f64], p: &[f64]) -> f64 {
    a.iter().zip(p.iter()).map(|(ai, pi)| (ai - pi).powi(2)).sum::<f64>() / a.len() as f64
}

fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    let mean_y = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_res: f64 = actual.iter().zip(predicted.iter()).map(|(a, p)| (a - p).powi(2)).sum();
    let ss_tot: f64 = actual.iter().map(|a| (a - mean_y).powi(2)).sum();
    1.0 - ss_res / ss_tot
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (r, c) = (m.len(), m[0].len());
    let mut t = vec![vec![0.0; r]; c];
    for i in 0..r { for j in 0..c { t[j][i] = m[i][j]; } }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (r, c, k) = (a.len(), b[0].len(), b.len());
    let mut m = vec![vec![0.0; c]; r];
    for i in 0..r { for j in 0..c { for l in 0..k { m[i][j] += a[i][l] * b[l][j]; } } }
    m
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()).collect()
}

fn invert_matrix(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n { for j in 0..n { aug[i][j] = m[i][j]; } aug[i][n + i] = 1.0; }
    for col in 0..n {
        let mut mr = col;
        for row in (col+1)..n { if aug[row][col].abs() > aug[mr][col].abs() { mr = row; } }
        aug.swap(col, mr);
        let p = aug[col][col];
        if p.abs() < 1e-12 { continue; }
        for j in 0..(2*n) { aug[col][j] /= p; }
        for row in 0..n { if row != col { let f = aug[row][col]; for j in 0..(2*n) { aug[row][j] -= f * aug[col][j]; } } }
    }
    aug.iter().map(|row| row[n..].to_vec()).collect()
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- R-squared alone is insufficient to evaluate a regression model. Residual analysis reveals patterns that summary statistics miss.
- Good residuals are randomly scattered with constant variance. Patterns indicate model misspecification; increasing variance indicates heteroscedasticity.
- Cook's distance identifies influential points that disproportionately affect the model fit — these deserve investigation.
- Regression diagnostics are not optional post-hoc checks. They are essential validation steps that determine whether a model's predictions can be trusted.
