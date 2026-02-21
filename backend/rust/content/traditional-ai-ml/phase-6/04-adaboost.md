# AdaBoost

> Phase 6 — Ensemble Methods | Kata 6.4

---

## Concept & Intuition

### What problem are we solving?

AdaBoost (Adaptive Boosting) is the original boosting algorithm that formalized the idea of combining weak learners. It provides a principled framework for determining (1) how much each model should focus on previous mistakes (sample re-weighting) and (2) how much influence each model should have in the final vote (model weighting). The algorithm is elegant: it has just one hyperparameter (the number of rounds T) and comes with theoretical guarantees about training error convergence.

At each round, AdaBoost trains a weak learner on weighted training data, computes the weighted error rate, calculates a model weight alpha that is higher for more accurate models, and updates sample weights so that misclassified examples become more prominent. The final prediction is a weighted majority vote of all weak learners.

The mathematical beauty of AdaBoost is that it can be shown to minimize an exponential loss function. Each round provably reduces the training error, and the algorithm is remarkably resistant to overfitting in practice -- even after hundreds of rounds, test error often continues to decrease or remains flat rather than increasing.

### Why naive approaches fail

Equal-weight voting of many stumps ignores the fact that some stumps are much better than others. A stump with 49% accuracy should have nearly zero influence, while one with 80% accuracy should dominate. AdaBoost's alpha weighting handles this automatically. Similarly, uniform sample weighting means the algorithm keeps making the same mistakes on hard examples. AdaBoost's adaptive re-weighting forces the ensemble to confront its weaknesses rather than repeatedly exploiting easy examples.

### Mental models

- **The alpha formula as confidence**: alpha = 0.5 * ln((1-err)/err). When error is low, alpha is high (confident model). When error is 0.5 (random guessing), alpha is zero (worthless model). When error exceeds 0.5, alpha is negative (contrarian model -- flip its predictions).
- **Exponential penalty**: misclassified samples get their weights multiplied by e^alpha, which grows quickly. After a few rounds of being misclassified, a sample's weight becomes enormous, forcing the next learner to get it right.
- **Margin maximization**: AdaBoost implicitly maximizes the margin -- the confidence of correct classifications. This is related to why it resists overfitting.

### Visual explanations

```
AdaBoost Algorithm:
  Initialize: w_i = 1/N for all samples

  For t = 1, 2, ..., T:
    1. Train weak learner h_t on weighted data
    2. Compute weighted error: err_t = sum(w_i * I(h_t(x_i) != y_i))
    3. Compute model weight: alpha_t = 0.5 * ln((1-err_t)/err_t)
    4. Update weights: w_i *= exp(-alpha_t * y_i * h_t(x_i))
    5. Normalize weights: w_i /= sum(w_j)

  Final prediction: H(x) = sign( sum(alpha_t * h_t(x)) )

Alpha as a function of error:
  err=0.01 --> alpha=2.30  (very confident)
  err=0.10 --> alpha=1.10  (confident)
  err=0.30 --> alpha=0.42  (moderate)
  err=0.49 --> alpha=0.02  (barely useful)
  err=0.50 --> alpha=0.00  (worthless)
```

---

## Hands-on Exploration

1. Implement the full AdaBoost algorithm with decision stumps as weak learners. Train on a 2D classification dataset.
2. Plot alpha values across rounds. Verify that early rounds (easy splits) get higher alpha than later rounds (harder splits).
3. Track training and test error across rounds. Observe whether test error keeps decreasing even after training error reaches zero.
4. Visualize the sample weight distribution after 10 rounds. Confirm that the hardest examples (near the decision boundary) have the highest weights.

---

## Live Code

```rust
fn main() {
    // --- RNG ---
    let mut rng: u64 = 12345;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };

    // --- Dataset: two interleaving half-moons ---
    let n = 300;
    let mut features: Vec<[f64; 2]> = Vec::new();
    let mut labels: Vec<i32> = Vec::new(); // +1 / -1

    let pi = std::f64::consts::PI;
    for i in 0..n {
        let noise_x = (rand_f64(&mut rng) - 0.5) * 0.3;
        let noise_y = (rand_f64(&mut rng) - 0.5) * 0.3;
        if i < n / 2 {
            let angle = pi * (i as f64) / (n as f64 / 2.0);
            features.push([angle.cos() + noise_x, angle.sin() + noise_y]);
            labels.push(1);
        } else {
            let angle = pi * ((i - n / 2) as f64) / (n as f64 / 2.0);
            features.push([1.0 - angle.cos() + noise_x, 0.5 - angle.sin() + noise_y]);
            labels.push(-1);
        }
    }

    // Train/test split
    let split = 200;
    let train_x = &features[..split];
    let train_y = &labels[..split];
    let test_x = &features[split..];
    let test_y = &labels[split..];

    // --- Decision stump ---
    fn train_stump(x: &[[f64; 2]], y: &[i32], w: &[f64]) -> (usize, f64, i32) {
        let mut best_err = f64::MAX;
        let mut best = (0_usize, 0.0_f64, 1_i32);

        for feat in 0..2 {
            let mut thresholds: Vec<f64> = x.iter().map(|r| r[feat]).collect();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            thresholds.dedup();

            for win in thresholds.windows(2) {
                let thresh = (win[0] + win[1]) / 2.0;
                for &pol in &[1_i32, -1] {
                    let mut err = 0.0;
                    for i in 0..x.len() {
                        let pred = if pol as f64 * (x[i][feat] - thresh) > 0.0 { 1 } else { -1 };
                        if pred != y[i] {
                            err += w[i];
                        }
                    }
                    if err < best_err {
                        best_err = err;
                        best = (feat, thresh, pol);
                    }
                }
            }
        }
        best
    }

    fn predict_stump(x: &[f64; 2], feat: usize, thresh: f64, pol: i32) -> i32 {
        if pol as f64 * (x[feat] - thresh) > 0.0 { 1 } else { -1 }
    }

    // --- AdaBoost ---
    let t_max = 30;
    let mut weights: Vec<f64> = vec![1.0 / split as f64; split];
    let mut learners: Vec<(usize, f64, i32, f64)> = Vec::new(); // feat, thresh, pol, alpha
    let mut train_errors: Vec<f64> = Vec::new();
    let mut test_errors: Vec<f64> = Vec::new();
    let mut alphas: Vec<f64> = Vec::new();

    println!("=== AdaBoost Training ===");
    println!("{:>5} {:>10} {:>10} {:>10} {:>10}", "Round", "W.Error", "Alpha", "TrainErr", "TestErr");
    println!("{}", "-".repeat(50));

    for t in 0..t_max {
        // 1. Train weak learner
        let (feat, thresh, pol) = train_stump(train_x, train_y, &weights);

        // 2. Weighted error
        let mut err = 0.0;
        for i in 0..split {
            if predict_stump(&train_x[i], feat, thresh, pol) != train_y[i] {
                err += weights[i];
            }
        }
        err = err.max(1e-10).min(1.0 - 1e-10);

        // 3. Model weight
        let alpha = 0.5 * ((1.0 - err) / err).ln();
        alphas.push(alpha);

        // 4. Update sample weights
        for i in 0..split {
            let pred = predict_stump(&train_x[i], feat, thresh, pol);
            weights[i] *= (-alpha * train_y[i] as f64 * pred as f64).exp();
        }

        // 5. Normalize
        let w_sum: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= w_sum;
        }

        learners.push((feat, thresh, pol, alpha));

        // Evaluate ensemble
        let ensemble_predict = |data: &[[f64; 2]]| -> Vec<i32> {
            data.iter().map(|sample| {
                let score: f64 = learners.iter()
                    .map(|&(f, t, p, a)| a * predict_stump(sample, f, t, p) as f64)
                    .sum();
                if score > 0.0 { 1 } else { -1 }
            }).collect()
        };

        let train_preds = ensemble_predict(train_x);
        let test_preds = ensemble_predict(test_x);

        let train_err = train_preds.iter().zip(train_y)
            .filter(|(p, t)| p != t).count() as f64 / split as f64;
        let test_err = test_preds.iter().zip(test_y)
            .filter(|(p, t)| p != t).count() as f64 / test_y.len() as f64;

        train_errors.push(train_err);
        test_errors.push(test_err);

        if t < 5 || t == 9 || t == 19 || t == t_max - 1 {
            println!("{:>5} {:>10.4} {:>10.3} {:>10.4} {:>10.4}",
                t + 1, err, alpha, train_err, test_err);
        }
    }

    // --- Alpha analysis ---
    println!("\n=== Alpha Values (model confidence) ===");
    println!("First 5 alphas: {:?}", &alphas[..5].iter().map(|a| format!("{:.3}", a)).collect::<Vec<_>>());
    println!("Last 5 alphas:  {:?}", &alphas[alphas.len()-5..].iter().map(|a| format!("{:.3}", a)).collect::<Vec<_>>());
    let avg_alpha: f64 = alphas.iter().sum::<f64>() / alphas.len() as f64;
    println!("Average alpha:  {:.3}", avg_alpha);

    // --- Weight distribution analysis ---
    println!("\n=== Final Weight Distribution ===");
    let max_w = weights.iter().cloned().fold(0.0_f64, f64::max);
    let min_w = weights.iter().cloned().fold(f64::MAX, f64::min);
    let uniform = 1.0 / split as f64;
    println!("Uniform weight: {:.6}", uniform);
    println!("Max weight:     {:.6} ({:.1}x uniform)", max_w, max_w / uniform);
    println!("Min weight:     {:.6} ({:.3}x uniform)", min_w, min_w / uniform);

    // Show top-10 highest weighted samples
    let mut indexed: Vec<(usize, f64)> = weights.iter().enumerate().map(|(i, &w)| (i, w)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nHardest samples (highest weights):");
    for &(idx, w) in indexed.iter().take(5) {
        println!("  sample {:>3}: x=[{:+.2}, {:+.2}], y={:+}, weight={:.6}",
            idx, train_x[idx][0], train_x[idx][1], train_y[idx], w);
    }

    // --- Margin analysis ---
    println!("\n=== Margin Analysis ===");
    let mut margins: Vec<f64> = Vec::new();
    let alpha_sum: f64 = alphas.iter().sum();
    for i in 0..split {
        let score: f64 = learners.iter()
            .map(|&(f, t, p, a)| a * predict_stump(&train_x[i], f, t, p) as f64)
            .sum();
        margins.push(train_y[i] as f64 * score / alpha_sum);
    }
    margins.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min_margin = margins[0];
    let median_margin = margins[margins.len() / 2];
    let mean_margin: f64 = margins.iter().sum::<f64>() / margins.len() as f64;
    let neg_count = margins.iter().filter(|&&m| m < 0.0).count();
    println!("Min margin:       {:.4}", min_margin);
    println!("Median margin:    {:.4}", median_margin);
    println!("Mean margin:      {:.4}", mean_margin);
    println!("Negative margins: {} (misclassified)", neg_count);

    // --- Summary ---
    println!("\n=== Summary ===");
    println!("Final train error: {:.4}", train_errors.last().unwrap());
    println!("Final test error:  {:.4}", test_errors.last().unwrap());
    println!("Number of rounds:  {}", t_max);
    println!("Number of stumps:  {}", learners.len());

    println!();
    println!("kata_metric(\"train_error\", {:.4})", train_errors.last().unwrap());
    println!("kata_metric(\"test_error\", {:.4})", test_errors.last().unwrap());
    println!("kata_metric(\"avg_alpha\", {:.4})", avg_alpha);
    println!("kata_metric(\"min_margin\", {:.4})", min_margin);
}
```

---

## Key Takeaways

- **AdaBoost adaptively re-weights samples to focus on mistakes.** Misclassified examples get exponentially increasing weight, forcing subsequent learners to prioritize them.
- **The alpha parameter gives better models more voting power.** A model with low error gets high alpha; a model at random-chance accuracy gets zero alpha.
- **AdaBoost minimizes exponential loss.** The algorithm can be derived as coordinate descent on an exponential loss function, giving it a solid theoretical foundation.
- **Surprisingly resistant to overfitting.** Even after training error reaches zero, test error often continues to improve as boosting increases the classification margin.
