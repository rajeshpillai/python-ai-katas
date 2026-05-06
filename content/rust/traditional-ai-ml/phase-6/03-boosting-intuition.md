# Boosting Intuition

> Phase 6 — Ensemble Methods | Kata 6.3

---

## Concept & Intuition

### What problem are we solving?

Bagging builds many independent models in parallel and averages them. Boosting takes a fundamentally different approach: it builds models **sequentially**, where each new model focuses on the mistakes of the previous ones. If the first model gets some examples wrong, the second model pays extra attention to those hard examples. The third model focuses on whatever the first two still get wrong. This iterative error-correction process means boosting can reduce *both* bias and variance.

The key insight is that combining many weak learners (models barely better than random guessing) can produce a strong learner with arbitrarily high accuracy. Each weak learner contributes a small improvement, and the cumulative effect of many such improvements is dramatic. This is sometimes called "the strength of weak learnability."

Boosting algorithms differ in how they define "focus on mistakes." AdaBoost re-weights training examples. Gradient Boosting fits new models to the residual errors. But the core principle is the same: sequential, error-correcting ensemble construction.

### Why naive approaches fail

Simply retraining the same model on the same data produces the same result every time. Bagging introduces diversity through random sampling, but each tree is still independently trained with no knowledge of what other trees got wrong. Boosting creates *directed diversity*: each new model is specifically designed to correct the ensemble's current weaknesses. This makes boosting particularly effective at reducing bias -- it can learn complex patterns that no single weak learner could capture.

### Mental models

- **Iterative tutoring**: a student takes a test, gets some questions wrong, then studies specifically those topics. Each round of study (each new model) targets the remaining gaps.
- **Additive correction**: think of boosting as building a prediction one layer at a time. The first layer gets the big picture; each subsequent layer adds finer corrections.
- **Weighted voting**: in boosting, better models get more voting power. A model that achieves high accuracy on the hard examples gets a louder voice in the final ensemble.

### Visual explanations

```
Bagging (parallel, independent):
  Model 1 ----\
  Model 2 -----+--> Average --> Final Prediction
  Model 3 ----/
  (each model trained independently)

Boosting (sequential, error-correcting):
  Model 1 --> errors --> Model 2 --> errors --> Model 3 --> ...
    |                     |                      |
    v                     v                      v
  pred_1     +          pred_2     +           pred_3    = Final Prediction
  (weight_1)            (weight_2)             (weight_3)

Focusing on mistakes:
  Round 1: Train on all data equally
           [o o o x x]  (x = misclassified)

  Round 2: Upweight the mistakes
           [o o o X X]  (X = higher weight)
           New model focuses on the hard examples

  Round 3: Upweight remaining mistakes
           Final ensemble combines all models
```

---

## Hands-on Exploration

1. Create a dataset where a single decision stump is clearly insufficient. Train one stump and observe its ~60% accuracy.
2. Manually implement a two-round boosting process: train stump 1, identify its errors, increase the weight of misclassified points, train stump 2 on the reweighted data.
3. Combine both stumps and observe the accuracy improvement. Compare to bagging two stumps.
4. Extend to 10 rounds and plot how the ensemble accuracy improves with each additional weak learner.

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

    // --- Generate a nonlinear classification dataset ---
    let n = 200;
    let mut x: Vec<[f64; 2]> = Vec::new();
    let mut y: Vec<i32> = Vec::new(); // +1 or -1

    for _ in 0..n {
        let x1 = rand_f64(&mut rng) * 4.0 - 2.0;
        let x2 = rand_f64(&mut rng) * 4.0 - 2.0;
        // Nonlinear boundary: circle
        let label = if x1 * x1 + x2 * x2 < 2.0 { 1 } else { -1 };
        // Add noise: flip 10% of labels
        let label = if rand_f64(&mut rng) < 0.1 { -label } else { label };
        x.push([x1, x2]);
        y.push(label);
    }

    let split = 150;
    let train_x = &x[..split];
    let train_y = &y[..split];
    let test_x = &x[split..];
    let test_y = &y[split..];

    // --- Decision stump: best threshold on one feature ---
    fn train_stump(
        x: &[[f64; 2]], y: &[i32], weights: &[f64],
    ) -> (usize, f64, i32) {
        // Returns (feature, threshold, polarity)
        let mut best_err = f64::MAX;
        let mut best = (0_usize, 0.0_f64, 1_i32);

        for feat in 0..2 {
            let mut vals: Vec<f64> = x.iter().map(|r| r[feat]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals.dedup();

            for &thresh in &vals {
                for &polarity in &[1_i32, -1] {
                    let mut err = 0.0;
                    for i in 0..x.len() {
                        let pred = if (polarity as f64) * (x[i][feat] - thresh) > 0.0 {
                            1
                        } else {
                            -1
                        };
                        if pred != y[i] {
                            err += weights[i];
                        }
                    }
                    if err < best_err {
                        best_err = err;
                        best = (feat, thresh, polarity);
                    }
                }
            }
        }
        best
    }

    fn predict_stump(sample: &[f64; 2], feat: usize, thresh: f64, polarity: i32) -> i32 {
        if (polarity as f64) * (sample[feat] - thresh) > 0.0 { 1 } else { -1 }
    }

    // --- Single stump baseline ---
    let uniform_weights: Vec<f64> = vec![1.0 / split as f64; split];
    let (s_feat, s_thresh, s_pol) = train_stump(train_x, train_y, &uniform_weights);

    let single_preds: Vec<i32> = test_x.iter()
        .map(|s| predict_stump(s, s_feat, s_thresh, s_pol))
        .collect();
    let single_acc = single_preds.iter().zip(test_y).filter(|(p, t)| p == t).count() as f64
        / test_y.len() as f64;

    println!("=== Single Decision Stump ===");
    println!("Feature: {}, Threshold: {:.2}, Polarity: {}", s_feat, s_thresh, s_pol);
    println!("Test accuracy: {:.4}", single_acc);
    println!();

    // --- Manual Boosting (step by step) ---
    println!("=== Manual Boosting (step by step) ===");

    let n_rounds = 20;
    let mut weights: Vec<f64> = vec![1.0 / split as f64; split];
    let mut stumps: Vec<(usize, f64, i32, f64)> = Vec::new(); // (feat, thresh, pol, alpha)

    for round in 0..n_rounds {
        // Train weighted stump
        let (feat, thresh, pol) = train_stump(train_x, train_y, &weights);

        // Compute weighted error
        let mut err = 0.0;
        for i in 0..split {
            let pred = predict_stump(&train_x[i], feat, thresh, pol);
            if pred != train_y[i] {
                err += weights[i];
            }
        }
        err = err.max(1e-10).min(1.0 - 1e-10);

        // Compute model weight (alpha)
        let alpha = 0.5 * ((1.0 - err) / err).ln();

        // Update sample weights
        for i in 0..split {
            let pred = predict_stump(&train_x[i], feat, thresh, pol);
            if pred != train_y[i] {
                weights[i] *= (alpha).exp();
            } else {
                weights[i] *= (-alpha).exp();
            }
        }

        // Normalize weights
        let w_sum: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= w_sum;
        }

        stumps.push((feat, thresh, pol, alpha));

        // Evaluate ensemble so far on test set
        let ensemble_preds: Vec<i32> = test_x.iter().map(|sample| {
            let score: f64 = stumps.iter()
                .map(|&(f, t, p, a)| a * predict_stump(sample, f, t, p) as f64)
                .sum();
            if score > 0.0 { 1 } else { -1 }
        }).collect();
        let ens_acc = ensemble_preds.iter().zip(test_y)
            .filter(|(p, t)| p == t).count() as f64 / test_y.len() as f64;

        if round < 5 || round == n_rounds - 1 || round == 9 {
            println!(
                "Round {:>2}: feat={}, thresh={:+.2}, alpha={:.3}, err={:.3}, ensemble_acc={:.4}",
                round + 1, feat, thresh, alpha, err, ens_acc
            );
        }
    }

    // --- Final ensemble evaluation ---
    let final_preds: Vec<i32> = test_x.iter().map(|sample| {
        let score: f64 = stumps.iter()
            .map(|&(f, t, p, a)| a * predict_stump(sample, f, t, p) as f64)
            .sum();
        if score > 0.0 { 1 } else { -1 }
    }).collect();
    let final_acc = final_preds.iter().zip(test_y)
        .filter(|(p, t)| p == t).count() as f64 / test_y.len() as f64;

    println!("\n=== Final Results ===");
    println!("Single stump accuracy:            {:.4}", single_acc);
    println!("Boosted ensemble ({} stumps):     {:.4}", n_rounds, final_acc);
    println!("Improvement:                      {:.1} percentage points",
        (final_acc - single_acc) * 100.0);

    // --- Show how accuracy builds up ---
    println!("\n=== Accuracy vs Number of Boosting Rounds ===");
    for rounds in [1, 2, 3, 5, 10, 15, 20].iter() {
        if *rounds > n_rounds { continue; }
        let preds: Vec<i32> = test_x.iter().map(|sample| {
            let score: f64 = stumps.iter().take(*rounds)
                .map(|&(f, t, p, a)| a * predict_stump(sample, f, t, p) as f64)
                .sum();
            if score > 0.0 { 1 } else { -1 }
        }).collect();
        let acc = preds.iter().zip(test_y).filter(|(p, t)| p == t).count() as f64
            / test_y.len() as f64;
        println!("  {:>2} rounds --> accuracy = {:.4}", rounds, acc);
    }

    // --- Weight distribution after boosting ---
    println!("\n=== Sample Weight Distribution (top 10 highest) ===");
    let mut indexed_weights: Vec<(usize, f64)> = weights.iter().enumerate()
        .map(|(i, &w)| (i, w)).collect();
    indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(idx, w) in indexed_weights.iter().take(10) {
        let bar: String = std::iter::repeat('#')
            .take((w * split as f64 * 20.0) as usize).collect();
        println!("  sample {:>3}: w={:.5} {} label={:+}", idx, w, bar, train_y[idx]);
    }

    println!();
    println!("kata_metric(\"single_stump_accuracy\", {:.4})", single_acc);
    println!("kata_metric(\"boosted_accuracy\", {:.4})", final_acc);
    println!("kata_metric(\"n_rounds\", {})", n_rounds);
}
```

---

## Key Takeaways

- **Boosting builds models sequentially, each correcting the previous ensemble's errors.** This is fundamentally different from bagging's parallel, independent approach.
- **Weak learners can combine into a strong learner.** Even simple decision stumps, when combined through boosting, can learn complex nonlinear boundaries.
- **Sample re-weighting is the key mechanism.** Hard-to-classify examples get increasing weight, forcing subsequent models to focus on them.
- **Boosting reduces both bias and variance.** Unlike bagging which primarily reduces variance, boosting can also reduce bias by iteratively adding complexity.
