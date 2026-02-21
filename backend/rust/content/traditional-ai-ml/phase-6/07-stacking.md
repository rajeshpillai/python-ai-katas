# Stacking

> Phase 6 — Ensemble Methods | Kata 6.7

---

## Concept & Intuition

### What problem are we solving?

Bagging and boosting combine models of the same type (many trees). Stacking (stacked generalization) combines models of *different* types using a meta-learner. The idea: a decision tree, a linear regression, and a k-NN model each capture different aspects of the data. Instead of simply averaging their predictions, a meta-learner (often a simple linear model) learns the *optimal* way to combine them -- giving more weight to models that are accurate in different regions of the input space.

Stacking works in two levels. Level 0 contains the base models (diverse learners). Level 1 is the meta-learner that takes the base model predictions as input features and learns to produce the final prediction. The critical detail is that the level-0 predictions used to train the meta-learner must be **out-of-fold** predictions (obtained via cross-validation on the training set) to avoid data leakage.

This architecture can capture complementary strengths: a linear model handles smooth trends, a tree captures sharp boundaries, and a distance-based model captures local patterns. The meta-learner automatically discovers which base model to trust in which situation.

### Why naive approaches fail

Averaging predictions equally ignores the fact that different models excel in different situations. Using the base models' training-set predictions to train the meta-learner causes catastrophic overfitting -- a model that memorizes training data gets high weight even though it will fail on new data. The cross-validation trick (out-of-fold predictions) is essential: it gives the meta-learner an honest view of each base model's generalization ability.

### Mental models

- **Panel of experts**: each expert (base model) has different expertise. The moderator (meta-learner) knows which expert to listen to depending on the question.
- **Cross-validation as honesty**: using out-of-fold predictions is like evaluating each expert on questions they have not seen, so the moderator gets an unbiased view of their skills.
- **Feature engineering via models**: the base model predictions are essentially learned features. The meta-learner performs feature combination on these high-level representations.

### Visual explanations

```
Level 0 (base models):              Level 1 (meta-learner):
  Training data                       Out-of-fold predictions
  +----------+                        +---------+---------+---------+
  |          | --> Linear Reg ------> | pred_LR | pred_DT | pred_KN | --> Meta-learner
  |  X, y    | --> Decision Tree ---> |         |         |         |     (e.g., Ridge)
  |          | --> K-NN ------------> |         |         |         |     --> Final pred
  +----------+                        +---------+---------+---------+

Cross-validation for out-of-fold predictions:
  Fold 1: Train on folds 2-5, predict fold 1
  Fold 2: Train on folds 1,3-5, predict fold 2
  ...
  Fold 5: Train on folds 1-4, predict fold 5
  --> Concatenate: out-of-fold predictions for ALL training data
```

---

## Hands-on Exploration

1. Train three diverse base models (linear regression, decision tree, simple k-NN) on the same dataset. Compare their individual performances.
2. Generate out-of-fold predictions using 5-fold cross-validation for each base model.
3. Train a meta-learner (ridge regression) on the stacked out-of-fold predictions. Compare stacking performance to individual models and simple averaging.
4. Examine the meta-learner weights to see which base models it values most.

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

    // --- Generate regression dataset ---
    let n = 200;
    let true_fn = |x: f64| -> f64 {
        if x < 1.0 { 2.0 * x }                   // linear region
        else if x < 2.5 { 2.0 + (x - 1.0).powi(2) }  // quadratic region
        else { 4.25 - 0.5 * (x - 2.5) }           // different linear region
    };

    let mut x_data: Vec<f64> = (0..n).map(|_| rand_f64(&mut rng) * 4.0).collect();
    x_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y_data: Vec<f64> = x_data.iter()
        .map(|&x| true_fn(x) + (rand_f64(&mut rng) - 0.5) * 0.4)
        .collect();

    let split = 150;
    let train_x = &x_data[..split];
    let train_y = &y_data[..split];
    let test_x = &x_data[split..];
    let test_y = &y_data[split..];

    let mse = |pred: &[f64], truth: &[f64]| -> f64 {
        pred.iter().zip(truth).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64
    };

    // --- Base Model 1: Linear Regression (y = a*x + b) ---
    fn fit_linear(x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let sx: f64 = x.iter().sum();
        let sy: f64 = y.iter().sum();
        let sxy: f64 = x.iter().zip(y).map(|(xi, yi)| xi * yi).sum();
        let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
        let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let b = (sy - a * sx) / n;
        (a, b)
    }
    fn predict_linear(x: &[f64], a: f64, b: f64) -> Vec<f64> {
        x.iter().map(|&xi| a * xi + b).collect()
    }

    // --- Base Model 2: Piecewise constant (decision tree with 8 regions) ---
    fn fit_piecewise(x: &[f64], y: &[f64], n_regions: usize) -> Vec<(f64, f64, f64)> {
        let n = x.len();
        let region_size = n / n_regions;
        let mut regions = Vec::new();
        let mut sorted: Vec<(f64, f64)> = x.iter().zip(y).map(|(&xi, &yi)| (xi, yi)).collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for r in 0..n_regions {
            let start = r * region_size;
            let end = if r == n_regions - 1 { n } else { (r + 1) * region_size };
            let lo = if r == 0 { f64::NEG_INFINITY } else { sorted[start].0 };
            let hi = if r == n_regions - 1 { f64::INFINITY } else { sorted[end - 1].0 };
            let mean: f64 = sorted[start..end].iter().map(|(_, y)| y).sum::<f64>()
                / (end - start) as f64;
            regions.push((lo, hi, mean));
        }
        regions
    }
    fn predict_piecewise(x: &[f64], regions: &[(f64, f64, f64)]) -> Vec<f64> {
        x.iter().map(|&xi| {
            for &(lo, hi, val) in regions {
                if xi >= lo && xi <= hi { return val; }
            }
            regions.last().unwrap().2
        }).collect()
    }

    // --- Base Model 3: K-NN regression (k=5) ---
    fn predict_knn(train_x: &[f64], train_y: &[f64], test_x: &[f64], k: usize) -> Vec<f64> {
        test_x.iter().map(|&xi| {
            let mut dists: Vec<(f64, f64)> = train_x.iter().zip(train_y)
                .map(|(&tx, &ty)| ((xi - tx).abs(), ty))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let sum: f64 = dists.iter().take(k).map(|(_, y)| y).sum();
            sum / k as f64
        }).collect()
    }

    // --- Individual model performance ---
    let (a, b) = fit_linear(train_x, train_y);
    let pred_lr_test = predict_linear(test_x, a, b);
    let mse_lr = mse(&pred_lr_test, test_y);

    let regions = fit_piecewise(train_x, train_y, 8);
    let pred_dt_test = predict_piecewise(test_x, &regions);
    let mse_dt = mse(&pred_dt_test, test_y);

    let pred_knn_test = predict_knn(train_x, train_y, test_x, 5);
    let mse_knn = mse(&pred_knn_test, test_y);

    println!("=== Individual Model Performance ===");
    println!("Linear Regression MSE: {:.4}", mse_lr);
    println!("Decision Tree MSE:     {:.4}", mse_dt);
    println!("K-NN (k=5) MSE:        {:.4}", mse_knn);

    // --- Simple averaging ---
    let pred_avg: Vec<f64> = (0..test_x.len()).map(|i| {
        (pred_lr_test[i] + pred_dt_test[i] + pred_knn_test[i]) / 3.0
    }).collect();
    let mse_avg = mse(&pred_avg, test_y);
    println!("Simple Average MSE:    {:.4}", mse_avg);

    // --- Stacking with cross-validation ---
    println!("\n=== Stacking (5-fold CV for out-of-fold predictions) ===");

    let n_folds = 5;
    let fold_size = split / n_folds;

    // Generate out-of-fold predictions for each base model
    let mut oof_lr = vec![0.0; split];
    let mut oof_dt = vec![0.0; split];
    let mut oof_knn = vec![0.0; split];

    for fold in 0..n_folds {
        let val_start = fold * fold_size;
        let val_end = if fold == n_folds - 1 { split } else { (fold + 1) * fold_size };

        // Build fold train/val sets
        let mut fold_train_x = Vec::new();
        let mut fold_train_y = Vec::new();
        let mut fold_val_x = Vec::new();

        for i in 0..split {
            if i >= val_start && i < val_end {
                fold_val_x.push(train_x[i]);
            } else {
                fold_train_x.push(train_x[i]);
                fold_train_y.push(train_y[i]);
            }
        }

        // Fit base models on fold training data
        let (fa, fb) = fit_linear(&fold_train_x, &fold_train_y);
        let fold_regions = fit_piecewise(&fold_train_x, &fold_train_y, 8);

        // Predict on fold validation data (out-of-fold)
        let lr_preds = predict_linear(&fold_val_x, fa, fb);
        let dt_preds = predict_piecewise(&fold_val_x, &fold_regions);
        let knn_preds = predict_knn(&fold_train_x, &fold_train_y, &fold_val_x, 5);

        for (j, i) in (val_start..val_end).enumerate() {
            oof_lr[i] = lr_preds[j];
            oof_dt[i] = dt_preds[j];
            oof_knn[i] = knn_preds[j];
        }
    }

    // --- Train meta-learner (Ridge Regression) on OOF predictions ---
    // Meta-features: [oof_lr, oof_dt, oof_knn]
    // Simple closed-form ridge: w = (X^T X + lambda I)^{-1} X^T y

    // Build meta-feature matrix
    let n_models = 3;
    let mut xtx = vec![vec![0.0; n_models]; n_models];
    let mut xty = vec![0.0; n_models];

    for i in 0..split {
        let feats = [oof_lr[i], oof_dt[i], oof_knn[i]];
        for j in 0..n_models {
            for k in 0..n_models {
                xtx[j][k] += feats[j] * feats[k];
            }
            xty[j] += feats[j] * train_y[i];
        }
    }

    // Add ridge penalty
    let ridge_lambda = 0.1;
    for j in 0..n_models {
        xtx[j][j] += ridge_lambda * split as f64;
    }

    // Solve 3x3 system using Cramer's rule
    fn det3(m: &[Vec<f64>]) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    let d = det3(&xtx);
    let mut meta_weights = vec![0.0; n_models];
    for col in 0..n_models {
        let mut m = xtx.clone();
        for row in 0..n_models {
            m[row][col] = xty[row];
        }
        meta_weights[col] = det3(&m) / d;
    }

    println!("\nMeta-learner weights:");
    let model_names = ["Linear Reg", "Decision Tree", "K-NN"];
    for i in 0..n_models {
        let bar_len = (meta_weights[i].abs() * 30.0) as usize;
        let bar: String = std::iter::repeat('|').take(bar_len).collect();
        println!("  {:<15}: {:+.4}  {}", model_names[i], meta_weights[i], bar);
    }

    // --- Final stacking predictions on test set ---
    // Refit base models on full training data
    let (fa, fb) = fit_linear(train_x, train_y);
    let final_lr = predict_linear(test_x, fa, fb);
    let final_regions = fit_piecewise(train_x, train_y, 8);
    let final_dt = predict_piecewise(test_x, &final_regions);
    let final_knn = predict_knn(train_x, train_y, test_x, 5);

    let pred_stacked: Vec<f64> = (0..test_x.len()).map(|i| {
        meta_weights[0] * final_lr[i] + meta_weights[1] * final_dt[i] + meta_weights[2] * final_knn[i]
    }).collect();
    let mse_stacked = mse(&pred_stacked, test_y);

    println!("\n=== Final Comparison ===");
    println!("{:<25} {:>10}", "Method", "Test MSE");
    println!("{}", "-".repeat(37));
    println!("{:<25} {:>10.4}", "Linear Regression", mse_lr);
    println!("{:<25} {:>10.4}", "Decision Tree", mse_dt);
    println!("{:<25} {:>10.4}", "K-NN (k=5)", mse_knn);
    println!("{:<25} {:>10.4}", "Simple Average", mse_avg);
    println!("{:<25} {:>10.4}", "Stacking", mse_stacked);

    let best_single = mse_lr.min(mse_dt).min(mse_knn);
    println!("\nStacking vs best single model: {:.1}% improvement",
        (1.0 - mse_stacked / best_single) * 100.0);
    println!("Stacking vs simple average:    {:.1}% improvement",
        (1.0 - mse_stacked / mse_avg) * 100.0);

    // --- Show how each model contributes in different regions ---
    println!("\n=== Model contributions by region ===");
    for &xi in &[0.5, 1.5, 2.0, 3.0, 3.5] {
        let lr_p = fa * xi + fb;
        let dt_p = predict_piecewise(&[xi], &final_regions)[0];
        let knn_p = predict_knn(train_x, train_y, &[xi], 5)[0];
        let stacked = meta_weights[0] * lr_p + meta_weights[1] * dt_p + meta_weights[2] * knn_p;
        let true_val = true_fn(xi);
        println!("  x={:.1}: LR={:.2}, DT={:.2}, KNN={:.2} -> stacked={:.2} (true={:.2})",
            xi, lr_p, dt_p, knn_p, stacked, true_val);
    }

    println!();
    println!("kata_metric(\"mse_best_single\", {:.4})", best_single);
    println!("kata_metric(\"mse_average\", {:.4})", mse_avg);
    println!("kata_metric(\"mse_stacked\", {:.4})", mse_stacked);
}
```

---

## Key Takeaways

- **Stacking combines diverse model types through a learned meta-learner.** Instead of simple averaging, the meta-learner discovers the optimal combination weights.
- **Out-of-fold predictions prevent data leakage.** The meta-learner must be trained on predictions the base models made on data they did not train on.
- **Diversity is more important than individual accuracy.** A weak model that makes different errors than the others can still improve the stack significantly.
- **The meta-learner should be simple.** A linear model or shallow tree as the meta-learner avoids second-level overfitting. The complexity should be in the base models, not the combiner.
