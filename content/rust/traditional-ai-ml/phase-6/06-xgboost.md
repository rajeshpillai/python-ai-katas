# XGBoost Concepts

> Phase 6 — Ensemble Methods | Kata 6.6

---

## Concept & Intuition

### What problem are we solving?

XGBoost (eXtreme Gradient Boosting) takes the gradient boosting framework and adds several engineering and algorithmic innovations that make it faster, more accurate, and more robust. The core improvements are: (1) a regularized objective function that penalizes tree complexity, (2) a second-order Taylor approximation of the loss for better split finding, (3) built-in handling of missing values, and (4) column subsampling similar to random forests.

The regularized objective adds a penalty proportional to the number of leaves and the magnitude of leaf weights. This explicitly controls model complexity, preventing the ensemble from growing arbitrarily complex trees. The second-order approximation (using both the gradient and the Hessian of the loss) provides more information about the curvature of the loss surface, enabling better split decisions than first-order gradient boosting.

While the actual XGBoost library is a large C++ codebase, understanding these core concepts is essential. In this kata, we implement a simplified version that captures the key ideas: regularized split finding, second-order approximations, and leaf weight computation.

### Why naive approaches fail

Standard gradient boosting with no regularization can overfit, especially with deep trees. It also uses only first-order gradient information, which is like optimizing with basic gradient descent instead of Newton's method. XGBoost's use of the Hessian (second derivative) is analogous to using Newton's method -- it takes bigger steps in flat regions and smaller steps in steep regions, converging faster.

### Mental models

- **Newton's method vs gradient descent**: first-order methods know the slope; second-order methods also know the curvature. Knowing the curvature means better step sizes.
- **Regularized scoring**: instead of just asking "which split reduces error most?", XGBoost asks "which split reduces error most *after accounting for the cost of adding complexity*?"
- **The gain formula**: Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma. This single formula captures the entire split decision.

### Visual explanations

```
XGBoost Objective:
  L = sum(loss(y_i, pred_i)) + sum(Omega(tree_k))

  where Omega(tree) = gamma * num_leaves + 0.5 * lambda * sum(leaf_weights^2)

Split Gain (with regularization):
  Gain = 0.5 * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma

  G = sum of gradients,  H = sum of hessians
  lambda = L2 regularization on leaf weights
  gamma  = minimum gain to make a split (pruning threshold)

Optimal leaf weight:
  w* = -G / (H + lambda)

  Compare to plain gradient boosting leaf:
  w* = -G / H   (no regularization)
```

---

## Hands-on Exploration

1. Implement the gradient and hessian computation for squared-error loss (gradient = pred - y, hessian = 1.0).
2. Compute the XGBoost gain formula for a candidate split. Show how lambda and gamma affect whether a split is accepted.
3. Build a simplified XGBoost tree that uses the gain formula for split finding and the optimal leaf weight formula for predictions.
4. Compare the regularized model against an unregularized gradient boosting model on a noisy dataset.

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

    // --- Generate noisy regression data ---
    let n = 200;
    let true_fn = |x: f64| -> f64 { (3.0 * x).sin() + 0.3 * x };
    let mut x_data: Vec<f64> = (0..n).map(|_| rand_f64(&mut rng) * 4.0 - 1.0).collect();
    x_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y_data: Vec<f64> = x_data.iter()
        .map(|&x| true_fn(x) + (rand_f64(&mut rng) - 0.5) * 0.8)
        .collect();

    let split = 150;
    let (train_x, test_x) = (&x_data[..split], &x_data[split..]);
    let (train_y, test_y) = (&y_data[..split], &y_data[split..]);

    // --- XGBoost-style tree building ---
    // For squared loss: gradient = pred - y, hessian = 1.0
    struct XGBTree {
        regions: Vec<(f64, f64, f64)>, // (low, high, leaf_weight)
    }

    fn xgb_gain(g_l: f64, h_l: f64, g_r: f64, h_r: f64, lambda: f64, gamma: f64) -> f64 {
        0.5 * (g_l * g_l / (h_l + lambda)
             + g_r * g_r / (h_r + lambda)
             - (g_l + g_r) * (g_l + g_r) / (h_l + h_r + lambda))
            - gamma
    }

    fn optimal_weight(g: f64, h: f64, lambda: f64) -> f64 {
        -g / (h + lambda)
    }

    fn build_xgb_tree(
        x: &[f64], gradients: &[f64], hessians: &[f64],
        max_depth: usize, lambda: f64, gamma: f64,
    ) -> XGBTree {
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let mut regions = Vec::new();
        xgb_split(&indices, x, gradients, hessians, 0, max_depth, lambda, gamma, &mut regions);
        XGBTree { regions }
    }

    fn xgb_split(
        indices: &[usize], x: &[f64], grad: &[f64], hess: &[f64],
        depth: usize, max_depth: usize, lambda: f64, gamma: f64,
        regions: &mut Vec<(f64, f64, f64)>,
    ) {
        if indices.is_empty() { return; }

        let g_total: f64 = indices.iter().map(|&i| grad[i]).sum();
        let h_total: f64 = indices.iter().map(|&i| hess[i]).sum();

        if depth >= max_depth || indices.len() < 4 {
            let lo = if regions.is_empty() { f64::NEG_INFINITY } else { x[indices[0]] };
            let hi = x[*indices.last().unwrap()];
            let weight = optimal_weight(g_total, h_total, lambda);
            regions.push((lo, hi, weight));
            return;
        }

        // Find best split using gain formula
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split_idx = indices.len() / 2;
        let mut g_left = 0.0;
        let mut h_left = 0.0;

        for s in 0..indices.len() - 1 {
            g_left += grad[indices[s]];
            h_left += hess[indices[s]];
            let g_right = g_total - g_left;
            let h_right = h_total - h_left;

            if h_left < 1.0 || h_right < 1.0 { continue; }

            let gain = xgb_gain(g_left, h_left, g_right, h_right, lambda, gamma);
            if gain > best_gain {
                best_gain = gain;
                best_split_idx = s + 1;
            }
        }

        // If no valid split (gain <= 0 due to gamma), make a leaf
        if best_gain <= 0.0 {
            let lo = if regions.is_empty() { f64::NEG_INFINITY } else { x[indices[0]] };
            let hi = x[*indices.last().unwrap()];
            let weight = optimal_weight(g_total, h_total, lambda);
            regions.push((lo, hi, weight));
            return;
        }

        xgb_split(&indices[..best_split_idx], x, grad, hess, depth + 1, max_depth, lambda, gamma, regions);
        xgb_split(&indices[best_split_idx..], x, grad, hess, depth + 1, max_depth, lambda, gamma, regions);
    }

    fn predict_xgb(tree: &XGBTree, x: f64) -> f64 {
        for &(lo, hi, w) in &tree.regions {
            if x >= lo && x <= hi { return w; }
        }
        // Nearest region fallback
        let mut min_dist = f64::MAX;
        let mut val = 0.0;
        for &(lo, hi, w) in &tree.regions {
            let mid = (lo.max(-100.0) + hi.min(100.0)) / 2.0;
            let d = (x - mid).abs();
            if d < min_dist { min_dist = d; val = w; }
        }
        val
    }

    let mse = |pred: &[f64], truth: &[f64]| -> f64 {
        pred.iter().zip(truth).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / pred.len() as f64
    };

    // --- Compare different regularization settings ---
    println!("=== XGBoost-Style Gradient Boosting ===\n");

    let configs: Vec<(&str, f64, f64)> = vec![
        ("No regularization", 0.0, 0.0),
        ("Lambda only (L2)", 1.0, 0.0),
        ("Gamma only (pruning)", 0.0, 0.5),
        ("Full regularization", 1.0, 0.5),
    ];

    let n_rounds = 50;
    let lr = 0.3;
    let max_depth = 3;

    for (name, lambda, gamma_val) in &configs {
        let y_mean: f64 = train_y.iter().sum::<f64>() / split as f64;
        let mut train_pred: Vec<f64> = vec![y_mean; split];
        let mut test_pred: Vec<f64> = vec![y_mean; test_x.len()];

        for _ in 0..n_rounds {
            // Gradients and hessians for squared loss
            let gradients: Vec<f64> = train_pred.iter().zip(train_y)
                .map(|(p, y)| p - y).collect();
            let hessians: Vec<f64> = vec![1.0; split]; // constant for squared loss

            let tree = build_xgb_tree(
                train_x, &gradients, &hessians, max_depth, *lambda, *gamma_val,
            );

            for i in 0..split {
                train_pred[i] += lr * predict_xgb(&tree, train_x[i]);
            }
            for i in 0..test_x.len() {
                test_pred[i] += lr * predict_xgb(&tree, test_x[i]);
            }
        }

        let train_mse = mse(&train_pred, train_y);
        let test_mse = mse(&test_pred, test_y);

        println!("{:<25} train_mse={:.4}  test_mse={:.4}  gap={:.4}",
            name, train_mse, test_mse, test_mse - train_mse);
    }

    // --- Demonstrate the gain formula ---
    println!("\n=== Gain Formula Demonstration ===");
    println!("Showing how lambda and gamma affect split decisions:\n");

    // A hypothetical split with these gradient/hessian sums
    let g_l = -3.0;
    let h_l = 5.0;
    let g_r = 2.0;
    let h_r = 7.0;

    println!("  G_L={:.1}, H_L={:.1}, G_R={:.1}, H_R={:.1}", g_l, h_l, g_r, h_r);
    println!("  {:>10} {:>10} {:>10} {:>10}", "lambda", "gamma", "gain", "split?");
    println!("  {}", "-".repeat(44));

    for &lam in &[0.0, 0.5, 1.0, 5.0] {
        for &gam in &[0.0, 0.5, 1.0] {
            let gain = xgb_gain(g_l, h_l, g_r, h_r, lam, gam);
            let split_ok = if gain > 0.0 { "yes" } else { "no" };
            println!("  {:>10.1} {:>10.1} {:>10.4} {:>10}", lam, gam, gain, split_ok);
        }
    }

    // --- Optimal leaf weights ---
    println!("\n=== Optimal Leaf Weights ===");
    println!("w* = -G / (H + lambda)\n");
    let g = -4.0;
    let h = 8.0;
    for &lam in &[0.0, 0.5, 1.0, 5.0, 10.0] {
        let w = optimal_weight(g, h, lam);
        println!("  G={:.1}, H={:.1}, lambda={:>5.1} --> w*={:.4} (shrunk by {:.1}%)",
            g, h, lam, w, (1.0 - w / optimal_weight(g, h, 0.0)) * 100.0);
    }

    // --- Full run with best config ---
    println!("\n=== Detailed Run (lambda=1.0, gamma=0.5) ===");
    let lambda = 1.0;
    let gamma_val = 0.5;
    let y_mean: f64 = train_y.iter().sum::<f64>() / split as f64;
    let mut train_pred: Vec<f64> = vec![y_mean; split];
    let mut test_pred: Vec<f64> = vec![y_mean; test_x.len()];

    println!("{:>5} {:>12} {:>12}", "Round", "Train MSE", "Test MSE");
    println!("{}", "-".repeat(32));

    for round in 0..n_rounds {
        let gradients: Vec<f64> = train_pred.iter().zip(train_y).map(|(p, y)| p - y).collect();
        let hessians: Vec<f64> = vec![1.0; split];

        let tree = build_xgb_tree(train_x, &gradients, &hessians, max_depth, lambda, gamma_val);

        for i in 0..split {
            train_pred[i] += lr * predict_xgb(&tree, train_x[i]);
        }
        for i in 0..test_x.len() {
            test_pred[i] += lr * predict_xgb(&tree, test_x[i]);
        }

        if round < 3 || round == 9 || round == 24 || round == 49 {
            println!("{:>5} {:>12.4} {:>12.4}",
                round + 1, mse(&train_pred, train_y), mse(&test_pred, test_y));
        }
    }

    let final_mse = mse(&test_pred, test_y);
    println!("\nFinal test MSE: {:.4}", final_mse);
    println!();
    println!("kata_metric(\"xgb_test_mse\", {:.4})", final_mse);
    println!("kata_metric(\"lambda\", {:.1})", lambda);
    println!("kata_metric(\"gamma\", {:.1})", gamma_val);
}
```

---

## Key Takeaways

- **XGBoost adds regularization to the gradient boosting objective.** Lambda penalizes large leaf weights (L2), gamma penalizes having too many leaves (structural complexity).
- **Second-order information (Hessians) enables smarter splits.** The gain formula uses both gradient and Hessian sums, analogous to Newton's method vs gradient descent.
- **The gain formula is a single, elegant expression** that decides whether a split is worth making, automatically incorporating regularization and pruning.
- **Regularization shrinks leaf weights toward zero.** The optimal leaf weight w* = -G/(H+lambda) shows how lambda prevents any single leaf from having too much influence.
