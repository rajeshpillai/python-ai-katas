# Loss Functions

> Phase 1 — What Does It Mean to Learn? | Kata 1.5

---

## Concept & Intuition

### What problem are we solving?

A loss function measures how wrong a prediction is. It takes the true value and the predicted value and returns a single number: zero if the prediction is perfect, and larger values for worse predictions. The choice of loss function defines what "good" means for your model — it is arguably the most important design decision in machine learning.

Different loss functions penalize errors differently. Mean Squared Error (MSE) penalizes large errors quadratically — an error of 10 is 100 times worse than an error of 1. This makes MSE sensitive to outliers. Mean Absolute Error (MAE) penalizes errors linearly — an error of 10 is only 10 times worse than an error of 1, making it more robust to outliers. Huber loss combines both: it behaves like MSE for small errors and MAE for large errors.

For classification, we use different losses. Cross-entropy loss measures the distance between predicted probabilities and true labels. It penalizes confident wrong predictions severely — predicting 0.99 when the true class is 0 incurs a massive loss. This encourages the model to be well-calibrated: confident when correct and uncertain when unsure.

### Why naive approaches fail

Using the wrong loss function can lead to models that optimize for the wrong thing. MSE on data with outliers will warp the model to accommodate extreme values. Accuracy as a loss function is not differentiable and has flat gradients almost everywhere, making optimization impossible. The loss function must match your problem and be compatible with your optimization method.

### Mental models

- **Loss as a grading rubric**: The loss function is how you grade your model. MSE is a harsh teacher (big mistakes are punished severely). MAE is more lenient (all mistakes punished proportionally).
- **Loss landscape**: For a given model, varying the parameters creates a "landscape" of loss values. Training means finding the lowest point in this landscape.
- **Loss function = problem definition**: Changing the loss function changes what the model learns. The same data with different losses produces different models.

### Visual explanations

```
  Loss functions for a single prediction:

  loss │
       │ \    MSE          .  MAE
       │  \  (x²)        /  (|x|)
       │   \            /
       │    \          /
       │     \        /
       │      ╲      /
       │       ╲____/   ← Huber (smooth transition)
       └──────────────── error

  MSE punishes large errors much more than small ones.
  MAE treats all errors proportionally.
  Huber blends both behaviors.
```

---

## Hands-on Exploration

1. Implement MSE, MAE, and Huber loss functions from scratch.
2. Apply each to the same set of predictions and observe how they differ, especially for outliers.
3. See how each loss function leads to a different "optimal" prediction for the same data.

---

## Live Code

```rust
fn main() {
    // === Loss Functions ===
    // How we measure "wrongness" defines what the model learns.

    // Loss functions
    let mse_loss = |true_vals: &[f64], pred_vals: &[f64]| -> f64 {
        let n = true_vals.len() as f64;
        true_vals.iter().zip(pred_vals.iter())
            .map(|(t, p)| (t - p) * (t - p))
            .sum::<f64>() / n
    };

    let mae_loss = |true_vals: &[f64], pred_vals: &[f64]| -> f64 {
        let n = true_vals.len() as f64;
        true_vals.iter().zip(pred_vals.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>() / n
    };

    let huber_loss = |true_vals: &[f64], pred_vals: &[f64], delta: f64| -> f64 {
        let n = true_vals.len() as f64;
        true_vals.iter().zip(pred_vals.iter())
            .map(|(t, p)| {
                let err = (t - p).abs();
                if err <= delta {
                    0.5 * err * err
                } else {
                    delta * (err - 0.5 * delta)
                }
            })
            .sum::<f64>() / n
    };

    // === Compare losses on clean data ===
    let targets = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let predictions = vec![110.0, 190.0, 310.0, 380.0, 520.0];

    println!("=== Loss Functions on Clean Data ===\n");
    println!("  Targets:     {:?}", targets);
    println!("  Predictions: {:?}", predictions);
    println!();
    println!("  MSE:   {:.2}", mse_loss(&targets, &predictions));
    println!("  MAE:   {:.2}", mae_loss(&targets, &predictions));
    println!("  Huber: {:.2} (delta=15)\n", huber_loss(&targets, &predictions, 15.0));

    // === Now add an outlier ===
    let targets_outlier = vec![100.0, 200.0, 300.0, 400.0, 500.0, 300.0];
    let preds_outlier = vec![110.0, 190.0, 310.0, 380.0, 520.0, 1000.0]; // last one is wildly off

    println!("=== Effect of an Outlier (true=300, pred=1000) ===\n");
    println!("  MSE:   {:.2}  ← blown up by outlier", mse_loss(&targets_outlier, &preds_outlier));
    println!("  MAE:   {:.2}  ← moderate increase", mae_loss(&targets_outlier, &preds_outlier));
    println!("  Huber: {:.2}  ← controlled by delta\n",
        huber_loss(&targets_outlier, &preds_outlier, 50.0));

    // === Per-point loss comparison ===
    println!("=== Per-Point Loss Comparison ===\n");
    println!("  {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "true", "pred", "error", "MSE", "MAE", "Huber");
    println!("  {:->6} {:->6} {:->8} {:->8} {:->8} {:->8}", "", "", "", "", "", "");

    for (&t, &p) in targets_outlier.iter().zip(preds_outlier.iter()) {
        let err = (t - p).abs();
        let mse_pt = (t - p) * (t - p);
        let mae_pt = err;
        let delta = 50.0;
        let huber_pt = if err <= delta { 0.5 * err * err } else { delta * (err - 0.5 * delta) };

        println!("  {:>6.0} {:>6.0} {:>+8.0} {:>8.0} {:>8.0} {:>8.0}",
            t, p, p - t, mse_pt, mae_pt, huber_pt);
    }

    // === What does each loss function optimize? ===
    println!("\n=== Optimal Constant Prediction Under Each Loss ===\n");
    println!("  Data with outlier: {:?}\n", targets_outlier);

    let sweep_min = 50.0;
    let sweep_max = 800.0;
    let steps = 100;

    let mut best_mse = (0.0, f64::INFINITY);
    let mut best_mae = (0.0, f64::INFINITY);
    let mut best_huber = (0.0, f64::INFINITY);

    for i in 0..=steps {
        let pred = sweep_min + (sweep_max - sweep_min) * i as f64 / steps as f64;
        let preds: Vec<f64> = vec![pred; targets_outlier.len()];

        let mse = mse_loss(&targets_outlier, &preds);
        let mae = mae_loss(&targets_outlier, &preds);
        let huber = huber_loss(&targets_outlier, &preds, 50.0);

        if mse < best_mse.1 { best_mse = (pred, mse); }
        if mae < best_mae.1 { best_mae = (pred, mae); }
        if huber < best_huber.1 { best_huber = (pred, huber); }
    }

    let mean = targets_outlier.iter().sum::<f64>() / targets_outlier.len() as f64;
    let mut sorted = targets_outlier.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0;

    println!("  MSE   → optimal prediction ≈ {:.0} (actual mean = {:.0})", best_mse.0, mean);
    println!("  MAE   → optimal prediction ≈ {:.0} (actual median = {:.0})", best_mae.0, median);
    println!("  Huber → optimal prediction ≈ {:.0} (between mean and median)\n", best_huber.0);
    println!("  The outlier (1000) pulls the MSE-optimal prediction toward it,");
    println!("  but has less effect on MAE and Huber.\n");

    // === Binary cross-entropy loss ===
    println!("=== Binary Cross-Entropy (Classification) ===\n");

    let bce_loss = |true_label: f64, pred_prob: f64| -> f64 {
        let p = pred_prob.max(1e-15).min(1.0 - 1e-15); // clamp for numerical stability
        -(true_label * p.ln() + (1.0 - true_label) * (1.0 - p).ln())
    };

    println!("  True label = 1 (positive class):");
    for &prob in &[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
        let loss = bce_loss(1.0, prob);
        let bar_len = (loss * 10.0).min(50.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("    pred={:.2}: loss={:.4}  |{}|", prob, loss, bar);
    }

    println!("\n  True label = 0 (negative class):");
    for &prob in &[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
        let loss = bce_loss(0.0, prob);
        let bar_len = (loss * 10.0).min(50.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("    pred={:.2}: loss={:.4}  |{}|", prob, loss, bar);
    }

    println!();
    println!("  Cross-entropy severely punishes confident WRONG predictions.");
    println!("  Predicting 0.99 when the truth is 0 → very high loss.\n");

    println!("Key insight: The loss function defines what 'good' means.");
    println!("MSE → mean, MAE → median, cross-entropy → calibrated probabilities.");
}
```

---

## Key Takeaways

- A loss function quantifies prediction error — it is the objective that the model minimizes during training.
- MSE penalizes large errors quadratically (sensitive to outliers), MAE penalizes linearly (robust to outliers), and Huber loss blends both behaviors.
- The choice of loss function determines the optimal prediction: MSE optimizes for the mean, MAE for the median.
- For classification, cross-entropy loss encourages well-calibrated probabilities and severely punishes confident wrong predictions.
