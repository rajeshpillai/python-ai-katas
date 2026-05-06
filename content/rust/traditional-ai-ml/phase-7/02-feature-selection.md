# Feature Selection

> Phase 7 — Feature Engineering & Pipelines | Kata 7.2

---

## Concept & Intuition

### What problem are we solving?

More features is not always better. Irrelevant or redundant features add noise, increase computation, cause overfitting, and make models harder to interpret. Feature selection identifies which features actually contribute to prediction quality and discards the rest. This is different from feature *creation* (adding features) -- selection is about removing what does not help.

There are three main approaches. **Filter methods** rank features independently of the model using statistical measures (correlation, mutual information, variance). They are fast but ignore feature interactions. **Wrapper methods** (forward selection, backward elimination) evaluate feature subsets by actually training models, giving better results but at much higher computational cost. **Embedded methods** perform selection as part of model training -- L1 regularization (Lasso) drives irrelevant feature weights to exactly zero, providing automatic feature selection.

The curse of dimensionality makes feature selection critical. With many irrelevant features, the model must search a vast, mostly useless space. By focusing on the truly informative features, you reduce the effective dimensionality and make the learning problem much easier.

### Why naive approaches fail

Using all available features seems safe -- "let the model sort it out." But this fails because: (1) noise features can trick the model into finding spurious patterns, (2) redundant features can cause instability in linear models (multicollinearity), and (3) more features means more parameters to fit, requiring more data to avoid overfitting. A model with 100 features and 200 samples is far more prone to overfitting than one with 10 features and 200 samples.

### Mental models

- **Signal-to-noise ratio**: each irrelevant feature is noise. More noise means the signal (true pattern) is harder to detect.
- **Occam's razor for features**: the simplest feature set that explains the data is usually the most generalizable.
- **Filter/wrapper/embedded as a spectrum**: filters are fast but crude, wrappers are slow but thorough, embedded methods strike a balance.

### Visual explanations

```
Feature Selection Approaches:

Filter Methods (fast, model-independent):
  Feature  Correlation  Variance  --> Rank --> Select top K
  F1       0.85         high      --> 1
  F2       0.72         high      --> 2
  F3       0.03         high      --> 5 (low correlation = drop)
  F4       0.68         low       --> 4
  F5       0.01         very low  --> 6 (drop: low variance)

Wrapper Methods (slow, model-dependent):
  {} -> {F1} -> {F1,F2} -> {F1,F2,F4} -> stop
  Start with nothing, greedily add the feature that most improves CV score

Embedded (L1/Lasso):
  Train with L1 penalty --> weights: [0.82, 0.0, 0.65, 0.0, 0.0, 0.41]
  Zero weights = automatically eliminated features
```

---

## Hands-on Exploration

1. Generate a dataset with 10 features where only 3 are truly informative. Train a model on all features and observe the test performance.
2. Implement correlation-based filtering. Rank features by absolute correlation with the target and select the top K.
3. Implement forward selection: start with no features, add one at a time, keeping the one that most improves cross-validation MSE.
4. Use L1 (Lasso) regularization and observe which feature weights go to zero. Compare the selected features to the true informative ones.

---

## Live Code

```rust
fn main() {
    // --- RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    // --- Generate dataset: 10 features, only 3 informative ---
    let n = 300;
    let n_features = 10;
    let informative = vec![0, 3, 7]; // true informative features
    let true_weights = vec![2.5, 0.0, 0.0, -1.8, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0];

    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut target: Vec<f64> = Vec::new();

    for _ in 0..n {
        let row: Vec<f64> = (0..n_features)
            .map(|_| rand_f64(&mut rng) * 2.0 - 1.0)
            .collect();
        let y: f64 = row.iter().zip(&true_weights)
            .map(|(x, w)| x * w).sum::<f64>()
            + (rand_f64(&mut rng) - 0.5) * 0.5; // noise
        data.push(row);
        target.push(y);
    }

    let split = 200;

    // --- Helper: compute correlation ---
    fn correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx: f64 = x.iter().sum::<f64>() / n;
        let my: f64 = y.iter().sum::<f64>() / n;
        let cov: f64 = x.iter().zip(y).map(|(xi, yi)| (xi - mx) * (yi - my)).sum::<f64>() / n;
        let sx: f64 = (x.iter().map(|xi| (xi - mx).powi(2)).sum::<f64>() / n).sqrt();
        let sy: f64 = (y.iter().map(|yi| (yi - my).powi(2)).sum::<f64>() / n).sqrt();
        if sx < 1e-10 || sy < 1e-10 { return 0.0; }
        cov / (sx * sy)
    }

    // --- Helper: simple linear regression with selected features ---
    fn fit_and_eval(
        data: &[Vec<f64>], target: &[f64], selected: &[usize],
        split: usize,
    ) -> (f64, f64) {
        // Simple: predict using mean of target + weighted sum
        // Use normal equations: w = (X^T X)^{-1} X^T y
        // For simplicity, use gradient descent
        let n_feat = selected.len();
        if n_feat == 0 {
            let mean = target[..split].iter().sum::<f64>() / split as f64;
            let mse: f64 = target[split..].iter().map(|y| (y - mean).powi(2)).sum::<f64>()
                / (target.len() - split) as f64;
            return (mse, mse);
        }

        // Standardize
        let mut means = vec![0.0; n_feat];
        let mut stds = vec![1.0; n_feat];
        for (j, &f) in selected.iter().enumerate() {
            means[j] = data[..split].iter().map(|r| r[f]).sum::<f64>() / split as f64;
            let var: f64 = data[..split].iter().map(|r| (r[f] - means[j]).powi(2)).sum::<f64>()
                / split as f64;
            stds[j] = var.sqrt().max(1e-8);
        }
        let y_mean: f64 = target[..split].iter().sum::<f64>() / split as f64;

        let mut w = vec![0.0; n_feat];
        let lr = 0.01;

        for _ in 0..1000 {
            let mut grad = vec![0.0; n_feat];
            for i in 0..split {
                let mut pred = 0.0;
                for (j, &f) in selected.iter().enumerate() {
                    pred += w[j] * (data[i][f] - means[j]) / stds[j];
                }
                let err = pred - (target[i] - y_mean);
                for j in 0..n_feat {
                    grad[j] += err * (data[i][selected[j]] - means[j]) / stds[j];
                }
            }
            for j in 0..n_feat {
                w[j] -= lr * 2.0 * grad[j] / split as f64;
            }
        }

        // Evaluate
        let eval = |start: usize, end: usize| -> f64 {
            let mut total = 0.0;
            for i in start..end {
                let mut pred = y_mean;
                for (j, &f) in selected.iter().enumerate() {
                    pred += w[j] * (data[i][f] - means[j]) / stds[j];
                }
                total += (target[i] - pred).powi(2);
            }
            total / (end - start) as f64
        };

        (eval(0, split), eval(split, data.len()))
    }

    // --- 1. Filter Method: Correlation-based ranking ---
    println!("=== Filter Method: Correlation Ranking ===\n");

    let mut correlations: Vec<(usize, f64)> = Vec::new();
    for f in 0..n_features {
        let col: Vec<f64> = data[..split].iter().map(|r| r[f]).collect();
        let corr = correlation(&col, &target[..split]).abs();
        correlations.push((f, corr));
    }
    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("{:>5} {:>12} {:>12} {:>12}", "Feat", "|Corr|", "True Weight", "Informative?");
    println!("{}", "-".repeat(45));
    for &(f, corr) in &correlations {
        let bar: String = std::iter::repeat('|').take((corr * 30.0) as usize).collect();
        let info = if informative.contains(&f) { "YES" } else { "" };
        println!("{:>5} {:>12.4} {:>12.1} {:>12} {}", f, corr, true_weights[f], info, bar);
    }

    // Evaluate with top-K features
    println!("\n  Test MSE with top-K features (filter):");
    for k in [1, 2, 3, 5, 10] {
        let selected: Vec<usize> = correlations.iter().take(k).map(|&(f, _)| f).collect();
        let (_, test_mse) = fit_and_eval(&data, &target, &selected, split);
        let features_str: Vec<String> = selected.iter().map(|f| format!("F{}", f)).collect();
        println!("    K={:>2}: MSE={:.4}  features={:?}", k, test_mse, features_str);
    }

    // --- 2. Forward Selection (Wrapper) ---
    println!("\n=== Forward Selection (Wrapper Method) ===\n");

    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..n_features).collect();

    println!("{:>5} {:>8} {:>12} {:>10}", "Step", "Added", "Test MSE", "Features");
    println!("{}", "-".repeat(40));

    for step in 0..n_features.min(5) {
        let mut best_mse = f64::MAX;
        let mut best_feat = remaining[0];

        for &f in &remaining {
            let mut trial = selected.clone();
            trial.push(f);
            let (_, test_mse) = fit_and_eval(&data, &target, &trial, split);
            if test_mse < best_mse {
                best_mse = test_mse;
                best_feat = f;
            }
        }

        selected.push(best_feat);
        remaining.retain(|&f| f != best_feat);

        let feat_str: Vec<String> = selected.iter().map(|f| format!("F{}", f)).collect();
        println!("{:>5} {:>8} {:>12.4} {:>10}", step + 1,
            format!("F{}", best_feat), best_mse, feat_str.join(","));
    }

    // --- 3. L1 (Lasso) via coordinate descent ---
    println!("\n=== Embedded Method: L1 Regularization (Lasso) ===\n");

    let lambdas = [0.001, 0.01, 0.1, 0.5, 1.0];

    // Standardize
    let mut feat_means = vec![0.0; n_features];
    let mut feat_stds = vec![1.0; n_features];
    for f in 0..n_features {
        feat_means[f] = data[..split].iter().map(|r| r[f]).sum::<f64>() / split as f64;
        let var: f64 = data[..split].iter().map(|r| (r[f] - feat_means[f]).powi(2)).sum::<f64>()
            / split as f64;
        feat_stds[f] = var.sqrt().max(1e-8);
    }
    let y_mean: f64 = target[..split].iter().sum::<f64>() / split as f64;

    for &lam in &lambdas {
        // Coordinate descent for Lasso
        let mut w = vec![0.0; n_features];

        for _ in 0..500 {
            for f in 0..n_features {
                // Compute residual without feature f
                let mut residual_sum = 0.0;
                let mut xx_sum = 0.0;

                for i in 0..split {
                    let x_f = (data[i][f] - feat_means[f]) / feat_stds[f];
                    let mut pred = 0.0;
                    for j in 0..n_features {
                        if j != f {
                            pred += w[j] * (data[i][j] - feat_means[j]) / feat_stds[j];
                        }
                    }
                    let residual = (target[i] - y_mean) - pred;
                    residual_sum += residual * x_f;
                    xx_sum += x_f * x_f;
                }

                // Soft thresholding
                let rho = residual_sum / split as f64;
                let threshold = lam;
                if rho > threshold {
                    w[f] = (rho - threshold) / (xx_sum / split as f64);
                } else if rho < -threshold {
                    w[f] = (rho + threshold) / (xx_sum / split as f64);
                } else {
                    w[f] = 0.0;
                }
            }
        }

        let non_zero: Vec<usize> = (0..n_features).filter(|&f| w[f].abs() > 1e-6).collect();
        let n_selected = non_zero.len();

        if lam == 0.01 || lam == 0.1 || lam == 0.5 {
            println!("Lambda = {:.3}: {} features selected", lam, n_selected);
            for f in 0..n_features {
                if w[f].abs() > 1e-6 {
                    let info = if informative.contains(&f) { " (informative)" } else { "" };
                    println!("    F{}: weight={:+.4}{}", f, w[f], info);
                }
            }
            println!();
        }
    }

    // --- Summary ---
    let all_feats: Vec<usize> = (0..n_features).collect();
    let (_, mse_all) = fit_and_eval(&data, &target, &all_feats, split);
    let (_, mse_true) = fit_and_eval(&data, &target, &informative, split);

    println!("=== Summary ===");
    println!("True informative features: {:?}", informative);
    println!("All 10 features MSE:       {:.4}", mse_all);
    println!("True 3 features MSE:       {:.4}", mse_true);
    println!("Fewer features, better generalization!");

    println!();
    println!("kata_metric(\"mse_all_features\", {:.4})", mse_all);
    println!("kata_metric(\"mse_selected_features\", {:.4})", mse_true);
}
```

---

## Key Takeaways

- **Feature selection removes irrelevant and redundant features.** This reduces overfitting, improves interpretability, and speeds up training.
- **Filter methods are fast but ignore interactions.** Correlation ranking is a good starting point but may miss features that are only useful in combination.
- **Wrapper methods evaluate feature subsets with actual model performance.** Forward selection and backward elimination are thorough but computationally expensive.
- **L1 regularization provides automatic embedded feature selection.** Lasso drives irrelevant feature weights to exactly zero, combining model training and feature selection in one step.
