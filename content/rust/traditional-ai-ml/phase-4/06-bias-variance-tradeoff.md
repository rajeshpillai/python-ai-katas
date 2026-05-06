# Bias-Variance Tradeoff

> Phase 4 — Model Evaluation & Selection | Kata 4.06

---

## Concept & Intuition

### What problem are we solving?

The bias-variance tradeoff is the central tension in machine learning. Bias is the error from oversimplified assumptions — a linear model applied to curved data will always be wrong in systematic ways (underfitting). Variance is the error from sensitivity to training data — a complex model that memorizes one dataset will fail on another (overfitting). Total error = bias^2 + variance + irreducible noise.

You cannot minimize both simultaneously. Reducing bias (making the model more complex) increases variance. Reducing variance (making the model simpler) increases bias. The goal is to find the sweet spot where their sum is minimized. This tradeoff governs every modeling decision: choosing the algorithm, setting hyperparameters, deciding feature sets, and determining how much data you need.

In this kata, we empirically demonstrate the bias-variance tradeoff by training models of increasing complexity on multiple bootstrap samples, decomposing the total error into bias and variance components.

### Why naive approaches fail

Evaluating a model on training data tells you nothing about the tradeoff — training error always decreases with complexity. Evaluating on a single test set conflates bias and variance into a single number. To truly understand the tradeoff, you need to train the model on multiple different training sets (generated via bootstrap) and observe how the predictions vary (variance) and how far the average prediction is from the truth (bias).

### Mental models

- **Bias as systematic error**: If you train the model 100 times on different samples, bias is how far the average prediction is from the truth. High bias means the model consistently gets it wrong in the same direction.
- **Variance as instability**: How much do predictions change across different training sets? High variance means the model is unreliable — its output depends heavily on which data it saw.
- **The U-shaped curve**: Plot test error vs. model complexity. You see a U shape: high error on the left (too simple, high bias), low error in the middle (good balance), high error on the right (too complex, high variance).

### Visual explanations

```
  Error
    |
    |  \                    /
    |   \                  /   <- Variance
    |    \    ____        /
    |     \  /    \      /
    |      \/      \    /     <- Total Error = Bias^2 + Variance
    |               \  /
    |   Bias^2 ->    \/
    |                         <- Irreducible noise
    +--------------------------------- Model Complexity
       Simple                Complex
       (underfit)            (overfit)
```

---

## Hands-on Exploration

1. Generate a dataset with a known underlying function.
2. Fit models of increasing complexity (polynomial degree) on multiple bootstrap samples.
3. Decompose the error into bias^2, variance, and noise at each query point.
4. Visualize the U-shaped tradeoff curve.

---

## Live Code

```rust
fn main() {
    println!("=== Bias-Variance Tradeoff ===\n");

    // True function: y = sin(x) + noise
    let n = 50;
    let mut rng = 42u64;

    // Generate full dataset
    let x_full: Vec<f64> = (0..n).map(|i| i as f64 * 0.12).collect();
    let noise_std = 0.3;

    // Test points for evaluation
    let x_test: Vec<f64> = (0..20).map(|i| 0.3 + i as f64 * 0.25).collect();
    let y_true: Vec<f64> = x_test.iter().map(|&x| x.sin()).collect();

    let n_bootstrap = 50;
    let degrees = vec![1, 2, 3, 5, 8, 12];

    println!("True function: y = sin(x)");
    println!("Noise std: {:.2}", noise_std);
    println!("Bootstrap samples: {}", n_bootstrap);
    println!("Test points: {}\n", x_test.len());

    println!("{:<8} {:>10} {:>10} {:>10} {:>10}",
        "Degree", "Bias^2", "Variance", "Noise", "Total Err");
    println!("{}", "-".repeat(50));

    let mut decompositions: Vec<(usize, f64, f64, f64, f64)> = Vec::new();

    for &degree in &degrees {
        // Collect predictions from all bootstrap models
        let mut all_predictions: Vec<Vec<f64>> = vec![Vec::new(); x_test.len()];

        for _ in 0..n_bootstrap {
            // Generate bootstrap sample
            let mut x_boot = Vec::new();
            let mut y_boot = Vec::new();

            for _ in 0..n {
                rng = lcg(rng);
                let idx = (rng as usize) % n;
                let x = x_full[idx];
                rng = lcg(rng);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0 * noise_std * 1.7;
                let y = x.sin() + noise;
                x_boot.push(x);
                y_boot.push(y);
            }

            // Fit polynomial
            let x_poly = poly_features(&x_boot, degree);
            let coeffs = ols_fit(&x_poly, &y_boot);

            // Predict on test points
            let x_test_poly = poly_features(&x_test, degree);
            let preds = predict(&x_test_poly, &coeffs);

            for (j, pred) in preds.iter().enumerate() {
                all_predictions[j].push(*pred);
            }
        }

        // Compute bias^2 and variance at each test point
        let mut total_bias_sq = 0.0;
        let mut total_variance = 0.0;
        let noise_variance = noise_std * noise_std;

        for j in 0..x_test.len() {
            let preds = &all_predictions[j];
            let mean_pred = preds.iter().sum::<f64>() / preds.len() as f64;
            let bias_sq = (mean_pred - y_true[j]).powi(2);
            let variance = preds.iter().map(|p| (p - mean_pred).powi(2)).sum::<f64>()
                / preds.len() as f64;

            total_bias_sq += bias_sq;
            total_variance += variance;
        }

        let avg_bias_sq = total_bias_sq / x_test.len() as f64;
        let avg_variance = total_variance / x_test.len() as f64;
        let total_error = avg_bias_sq + avg_variance + noise_variance;

        decompositions.push((degree, avg_bias_sq, avg_variance, noise_variance, total_error));

        println!("{:<8} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            degree, avg_bias_sq, avg_variance, noise_variance, total_error);
    }

    // ASCII visualization of the tradeoff
    println!("\n--- Bias-Variance Tradeoff Curve ---");
    let max_err = decompositions.iter().map(|d| d.4).fold(0.0f64, f64::max);

    println!("  Error");
    let h = 12;
    for row in 0..h {
        let level = max_err * (h - row) as f64 / h as f64;
        print!("  {:.2} |", level);
        for d in &decompositions {
            let bias_bar = if d.1 >= level { 'B' } else { ' ' };
            let var_bar = if d.1 + d.2 >= level { 'V' } else if d.1 >= level { 'B' } else { ' ' };
            let total_bar = if d.4 >= level { '|' } else { ' ' };
            print!(" {}{}{} ", bias_bar, var_bar, total_bar);
        }
        println!();
    }
    print!("       +");
    for _ in &decompositions { print!("-----"); }
    println!();
    print!("        ");
    for d in &decompositions { print!(" d={:<2}", d.0); }
    println!();
    println!("  B=Bias^2, V=Variance, |=Total");

    // Find the sweet spot
    let best = decompositions.iter().min_by(|a, b| a.4.partial_cmp(&b.4).unwrap()).unwrap();
    println!("\n--- Sweet Spot ---");
    println!("  Best degree: {} (total error: {:.4})", best.0, best.4);
    println!("  Bias^2: {:.4}, Variance: {:.4}", best.1, best.2);

    // Practical implications
    println!("\n--- Practical Implications ---");
    println!("  If model underfits (high bias):");
    println!("    - Increase model complexity (higher degree, more features)");
    println!("    - Reduce regularization");
    println!("    - More training data will NOT help much");
    println!("  If model overfits (high variance):");
    println!("    - Decrease model complexity (lower degree, fewer features)");
    println!("    - Increase regularization");
    println!("    - More training data WILL help");

    kata_metric("best_degree", best.0 as f64);
    kata_metric("best_total_error", best.4);
    kata_metric("best_bias_squared", best.1);
    kata_metric("best_variance", best.2);
}

fn poly_features(x: &[f64], degree: usize) -> Vec<Vec<f64>> {
    x.iter().map(|&xi| {
        let mut row = vec![1.0];
        for d in 1..=degree { row.push(xi.powi(d as i32)); }
        row
    }).collect()
}

fn ols_fit(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let xt = transpose(x);
    let mut xtx = mat_mul(&xt, x);
    // Add small ridge for numerical stability
    for i in 0..xtx.len() { xtx[i][i] += 1e-8; }
    let xty = mat_vec_mul(&xt, y);
    let inv = invert_matrix(&xtx);
    mat_vec_mul(&inv, &xty)
}

fn predict(x: &[Vec<f64>], c: &[f64]) -> Vec<f64> {
    x.iter().map(|r| r.iter().zip(c.iter()).map(|(a,b)| a*b).sum()).collect()
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (r,c)=(m.len(),m[0].len()); let mut t=vec![vec![0.0;r];c];
    for i in 0..r{for j in 0..c{t[j][i]=m[i][j];}} t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let(r,c,k)=(a.len(),b[0].len(),b.len()); let mut m=vec![vec![0.0;c];r];
    for i in 0..r{for j in 0..c{for l in 0..k{m[i][j]+=a[i][l]*b[l][j];}}} m
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|r| r.iter().zip(v.iter()).map(|(a,b)| a*b).sum()).collect()
}

fn invert_matrix(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n=m.len(); let mut a=vec![vec![0.0;2*n];n];
    for i in 0..n{for j in 0..n{a[i][j]=m[i][j];} a[i][n+i]=1.0;}
    for c in 0..n{
        let mut mr=c; for r in (c+1)..n{if a[r][c].abs()>a[mr][c].abs(){mr=r;}} a.swap(c,mr);
        let p=a[c][c]; if p.abs()<1e-12{continue;}
        for j in 0..(2*n){a[c][j]/=p;}
        for r in 0..n{if r!=c{let f=a[r][c]; for j in 0..(2*n){a[r][j]-=f*a[c][j];}}}
    }
    a.iter().map(|r| r[n..].to_vec()).collect()
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Total prediction error decomposes into bias^2 (systematic error from model assumptions), variance (sensitivity to training data), and irreducible noise.
- Increasing model complexity reduces bias but increases variance. The optimal model minimizes their sum.
- Underfitting (high bias) is solved by more complexity or features. Overfitting (high variance) is solved by less complexity, more regularization, or more data.
- The bias-variance tradeoff is the theoretical foundation for all model selection decisions in machine learning.
