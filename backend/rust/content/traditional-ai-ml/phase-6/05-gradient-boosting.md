# Gradient Boosting

> Phase 6 — Ensemble Methods | Kata 6.5

---

## Concept & Intuition

### What problem are we solving?

AdaBoost re-weights samples to focus on mistakes. Gradient Boosting takes a more general approach: each new model is trained to predict the **residual errors** (or more precisely, the negative gradient of the loss function) of the current ensemble. This framework is incredibly flexible because it works with any differentiable loss function -- squared error for regression, log-loss for classification, absolute error for robust regression, and many more.

The key insight is that gradient boosting performs gradient descent in function space. Instead of adjusting numerical parameters as in traditional gradient descent, we add entire functions (typically small decision trees) to the ensemble. Each tree represents one step in the direction that most reduces the loss. The learning rate controls how much each tree contributes, providing a regularization mechanism that prevents overfitting.

Gradient Boosting with decision trees as base learners has become one of the most successful algorithms in applied machine learning. It dominates tabular data competitions and is a workhorse in industry. Understanding the residual-fitting perspective is essential for grasping why it works so well.

### Why naive approaches fail

Simply fitting a large tree to the data overfits. Fitting a small tree underfits. Gradient boosting threads the needle: each small tree captures a little bit of the remaining signal, and the accumulation of many small corrections builds a powerful predictor. Training a single model on the original targets misses the iterative error-correction that makes boosting effective. The learning rate is critical -- without it (learning_rate=1.0), each tree tries to fully correct all errors at once, leading to oscillation and overfitting.

### Mental models

- **Sculptor analogy**: the first tree carves the rough shape. Each subsequent tree adds finer details by correcting the remaining imperfections. The learning rate determines how bold each carving stroke is.
- **Gradient descent in function space**: at each step, we ask "in what direction (which function) should I move to most reduce my loss?" The answer is the negative gradient, which for squared error simplifies to the residuals.
- **Shrinkage as patience**: a small learning rate means each tree contributes only a tiny correction. This requires more trees but produces a better final model -- the ensemble explores the function space more carefully.

### Visual explanations

```
Gradient Boosting for Regression:

  Round 0: F_0(x) = mean(y)

  Round 1: residuals = y - F_0(x)
           Train tree h_1 on residuals
           F_1(x) = F_0(x) + lr * h_1(x)

  Round 2: residuals = y - F_1(x)
           Train tree h_2 on residuals
           F_2(x) = F_1(x) + lr * h_2(x)

  ...

  Round T: F_T(x) = F_0(x) + lr * sum(h_t(x))

Example (learning_rate = 0.3):
  True y = 10.0
  F_0 = 5.0      residual = 5.0
  h_1  = 4.8     F_1 = 5.0 + 0.3*4.8 = 6.44      residual = 3.56
  h_2  = 3.4     F_2 = 6.44 + 0.3*3.4 = 7.46     residual = 2.54
  h_3  = 2.5     F_3 = 7.46 + 0.3*2.5 = 8.21     residual = 1.79
  ...slowly converges to 10.0
```

---

## Hands-on Exploration

1. Generate a noisy nonlinear regression dataset. Initialize the prediction to the mean of y and compute the residuals.
2. Fit a shallow decision tree to the residuals. Add its predictions (scaled by learning rate) to the ensemble. Repeat for 10 rounds and track how residuals shrink.
3. Compare learning rates of 0.01, 0.1, and 1.0. With lr=1.0, observe oscillation; with lr=0.01, see smooth but slow convergence.
4. Plot the ensemble prediction after 1, 5, 10, 50, and 100 rounds to visualize the progressive refinement.

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

    // --- True function + noisy data ---
    let true_fn = |x: f64| -> f64 {
        (2.0 * x).sin() + 0.5 * x - 0.1 * x * x
    };

    let n_train = 100;
    let mut x_train: Vec<f64> = (0..n_train)
        .map(|_| rand_f64(&mut rng) * 6.0 - 1.0)
        .collect();
    x_train.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y_train: Vec<f64> = x_train.iter()
        .map(|&x| true_fn(x) + (rand_f64(&mut rng) - 0.5) * 0.6)
        .collect();

    let n_test = 40;
    let x_test: Vec<f64> = (0..n_test)
        .map(|i| -1.0 + i as f64 * 6.0 / (n_test - 1) as f64)
        .collect();
    let y_test: Vec<f64> = x_test.iter().map(|&x| true_fn(x)).collect();

    // --- Regression tree stump (1D, finds best split point) ---
    fn fit_tree(x: &[f64], y: &[f64], max_leaves: usize) -> Vec<(f64, f64, f64)> {
        // Returns regions: (low, high, value)
        if x.is_empty() { return vec![]; }

        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let mut regions: Vec<(usize, usize)> = vec![(0, indices.len())]; // start, end in sorted order

        // Greedily split regions
        for _ in 0..max_leaves.saturating_sub(1) {
            let mut best_gain = 0.0;
            let mut best_region = 0;
            let mut best_split = 0;

            for (r, &(start, end)) in regions.iter().enumerate() {
                if end - start < 4 { continue; }
                let mean: f64 = (start..end).map(|i| y[indices[i]]).sum::<f64>()
                    / (end - start) as f64;
                let total_mse: f64 = (start..end).map(|i| (y[indices[i]] - mean).powi(2)).sum();

                for s in (start + 2)..(end - 1) {
                    let left_mean: f64 = (start..s).map(|i| y[indices[i]]).sum::<f64>()
                        / (s - start) as f64;
                    let right_mean: f64 = (s..end).map(|i| y[indices[i]]).sum::<f64>()
                        / (end - s) as f64;
                    let left_mse: f64 = (start..s).map(|i| (y[indices[i]] - left_mean).powi(2)).sum();
                    let right_mse: f64 = (s..end).map(|i| (y[indices[i]] - right_mean).powi(2)).sum();
                    let gain = total_mse - left_mse - right_mse;

                    if gain > best_gain {
                        best_gain = gain;
                        best_region = r;
                        best_split = s;
                    }
                }
            }

            if best_gain <= 0.0 { break; }
            let (start, end) = regions[best_region];
            regions.remove(best_region);
            regions.push((start, best_split));
            regions.push((best_split, end));
        }

        // Convert to (low, high, value) format
        regions.iter().map(|&(start, end)| {
            let lo = if start == 0 { f64::NEG_INFINITY } else { x[indices[start]] };
            let hi = if end == indices.len() { f64::INFINITY } else { x[indices[end - 1]] };
            let mean: f64 = (start..end).map(|i| y[indices[i]]).sum::<f64>()
                / (end - start) as f64;
            (lo, hi, mean)
        }).collect()
    }

    fn predict_tree(tree: &[(f64, f64, f64)], x: f64) -> f64 {
        for &(lo, hi, val) in tree {
            if x >= lo && x <= hi { return val; }
        }
        // Fallback: nearest region
        tree.last().map(|t| t.2).unwrap_or(0.0)
    }

    let mse = |pred: &[f64], truth: &[f64]| -> f64 {
        pred.iter().zip(truth).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64
    };

    // --- Gradient Boosting ---
    let learning_rates = [0.05, 0.1, 0.5];
    let n_rounds = 100;
    let max_leaves = 4;

    println!("=== Gradient Boosting Regression ===\n");

    for &lr in &learning_rates {
        // Initialize: F_0 = mean(y)
        let y_mean: f64 = y_train.iter().sum::<f64>() / n_train as f64;
        let mut current_pred: Vec<f64> = vec![y_mean; n_train];
        let mut test_pred: Vec<f64> = vec![y_mean; n_test];
        let mut trees: Vec<Vec<(f64, f64, f64)>> = Vec::new();

        let mut milestones = Vec::new();

        for round in 0..n_rounds {
            // Compute residuals (negative gradient for squared loss)
            let residuals: Vec<f64> = y_train.iter().zip(&current_pred)
                .map(|(y, p)| y - p)
                .collect();

            // Fit tree to residuals
            let tree = fit_tree(&x_train, &residuals, max_leaves);

            // Update predictions
            for i in 0..n_train {
                current_pred[i] += lr * predict_tree(&tree, x_train[i]);
            }
            for i in 0..n_test {
                test_pred[i] += lr * predict_tree(&tree, x_test[i]);
            }

            trees.push(tree);

            // Track milestones
            if round + 1 == 1 || round + 1 == 5 || round + 1 == 10
                || round + 1 == 50 || round + 1 == 100 {
                let train_mse = mse(&current_pred, &y_train);
                let test_mse = mse(&test_pred, &y_test);
                milestones.push((round + 1, train_mse, test_mse));
            }
        }

        println!("Learning rate = {:.2}:", lr);
        println!("  {:>6} {:>12} {:>12}", "Rounds", "Train MSE", "Test MSE");
        println!("  {}", "-".repeat(32));
        for (r, tr, te) in &milestones {
            println!("  {:>6} {:>12.4} {:>12.4}", r, tr, te);
        }
        println!();
    }

    // --- Detailed run with lr=0.1 ---
    println!("=== Detailed Run (lr=0.1) ===\n");
    let lr = 0.1;
    let y_mean: f64 = y_train.iter().sum::<f64>() / n_train as f64;
    let mut current_pred: Vec<f64> = vec![y_mean; n_train];
    let mut test_pred: Vec<f64> = vec![y_mean; n_test];
    let mut trees: Vec<Vec<(f64, f64, f64)>> = Vec::new();

    // Show residual shrinkage
    println!("Residual statistics per round:");
    println!("{:>5} {:>12} {:>12} {:>12}", "Round", "Mean|Resid|", "Max|Resid|", "Test MSE");
    println!("{}", "-".repeat(45));

    for round in 0..n_rounds {
        let residuals: Vec<f64> = y_train.iter().zip(&current_pred)
            .map(|(y, p)| y - p).collect();

        let tree = fit_tree(&x_train, &residuals, max_leaves);

        for i in 0..n_train {
            current_pred[i] += lr * predict_tree(&tree, x_train[i]);
        }
        for i in 0..n_test {
            test_pred[i] += lr * predict_tree(&tree, x_test[i]);
        }

        trees.push(tree);

        if round < 5 || round == 9 || round == 19 || round == 49 || round == 99 {
            let new_residuals: Vec<f64> = y_train.iter().zip(&current_pred)
                .map(|(y, p)| y - p).collect();
            let mean_abs: f64 = new_residuals.iter().map(|r| r.abs()).sum::<f64>() / n_train as f64;
            let max_abs: f64 = new_residuals.iter().map(|r| r.abs()).fold(0.0_f64, f64::max);
            let test_mse = mse(&test_pred, &y_test);
            println!("{:>5} {:>12.4} {:>12.4} {:>12.4}", round + 1, mean_abs, max_abs, test_mse);
        }
    }

    // --- Sample predictions ---
    println!("\nSample predictions (x, true, predicted):");
    for i in (0..n_test).step_by(8) {
        println!("  x={:+.2}  true={:+.3}  pred={:+.3}  error={:+.3}",
            x_test[i], y_test[i], test_pred[i], test_pred[i] - y_test[i]);
    }

    let final_test_mse = mse(&test_pred, &y_test);
    let baseline_mse = mse(&vec![y_mean; n_test], &y_test);

    println!("\n=== Summary ===");
    println!("Baseline MSE (predict mean): {:.4}", baseline_mse);
    println!("Gradient Boosting MSE:       {:.4}", final_test_mse);
    println!("Improvement:                 {:.1}%", (1.0 - final_test_mse / baseline_mse) * 100.0);
    println!("Number of trees:             {}", trees.len());
    println!("Learning rate:               {}", lr);

    println!();
    println!("kata_metric(\"baseline_mse\", {:.4})", baseline_mse);
    println!("kata_metric(\"gradient_boosting_mse\", {:.4})", final_test_mse);
    println!("kata_metric(\"n_trees\", {})", trees.len());
}
```

---

## Key Takeaways

- **Gradient Boosting fits each new tree to the residuals (negative gradient) of the current ensemble.** This generalizes to any differentiable loss function.
- **The learning rate (shrinkage) is a critical regularizer.** Small learning rates require more trees but produce better generalization by exploring function space more carefully.
- **Each tree is small (few leaves), capturing one "correction" to the ensemble.** The power comes from the accumulation of many small, targeted corrections.
- **Gradient Boosting is gradient descent in function space.** Each tree is a step in the direction that most reduces the loss, making the framework both intuitive and theoretically grounded.
