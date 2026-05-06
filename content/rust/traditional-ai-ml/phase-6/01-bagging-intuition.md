# Bagging Intuition

> Phase 6 — Ensemble Methods | Kata 6.1

---

## Concept & Intuition

### What problem are we solving?

A single decision tree is unstable: change a few training points and the tree structure can look completely different. This instability means high variance -- individual trees overfit to the noise in whatever sample they see. Bagging (Bootstrap AGGregatING) solves this by training many trees on different random subsets of the data and averaging their predictions.

The core idea is borrowed from statistics: if you have many noisy, independent estimators, their average is far less noisy than any individual one. A single tree might have 40% error variance, but averaging 50 trees cuts that variance dramatically. The individual trees are still overfit, but they overfit to *different* noise, so the errors cancel out.

Bagging works best when the base learner is powerful but unstable. Decision trees are the perfect candidate because they can capture complex patterns (low bias) but fluctuate wildly with different data (high variance). By aggregating many such trees, you keep the expressiveness while taming the instability.

### Why naive approaches fail

You might think: "Why not just build one really good tree?" The problem is that pruning a tree to reduce overfitting also reduces its ability to capture real patterns. You are trading variance for bias. Bagging offers a way to reduce variance *without* increasing bias -- you keep the deep, expressive trees but average away their individual quirks. Training the same tree multiple times on the same data is useless; the bootstrap sampling step is what creates the diversity needed for the average to work.

### Mental models

- **Wisdom of crowds**: ask 100 people to guess the weight of an ox. Individual guesses are noisy, but the average is remarkably accurate -- as long as the errors are not all in the same direction.
- **Parallel committee**: each tree is an expert trained on a slightly different version of reality. The committee vote is more reliable than any single member.
- **Noise cancellation**: each tree's error is partly signal (real pattern) and partly noise (random fluctuation). Averaging preserves the signal and cancels the noise.

### Visual explanations

```
Original Data: [x1, x2, x3, x4, x5, x6, x7, x8]

Bootstrap Sample 1: [x2, x2, x5, x1, x7, x3, x3, x8]  --> Tree 1 --> pred_1
Bootstrap Sample 2: [x4, x1, x6, x6, x3, x8, x5, x2]  --> Tree 2 --> pred_2
Bootstrap Sample 3: [x7, x3, x1, x5, x5, x2, x8, x4]  --> Tree 3 --> pred_3
  ...                                                        ...
Bootstrap Sample B: [x3, x8, x2, x2, x6, x1, x4, x7]  --> Tree B --> pred_B

Final prediction = average(pred_1, pred_2, ..., pred_B)

Variance Reduction:
  Single tree:    Var(pred)
  Bagged (B):     Var(pred) / B   (if trees were independent)
  In practice:    somewhere in between (trees are correlated)
```

---

## Hands-on Exploration

1. Generate a noisy regression dataset with a known nonlinear function. Fit a single deep decision tree stump and observe its overfit, jagged predictions.
2. Manually create 10 bootstrap samples by sampling with replacement. Train a separate tree on each and observe how their individual predictions differ wildly.
3. Average all 10 tree predictions together and compare the smooth ensemble prediction against the single tree. Measure MSE for both.
4. Increase the number of trees from 1 to 50 and observe how the test MSE decreases and stabilizes, showing diminishing returns after a certain point.

---

## Live Code

```rust
fn main() {
    // --- Simple LCG random number generator ---
    let mut rng_state: u64 = 42;
    let mut rand_f64 = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / (2147483648.0)
    };

    let mut rand_range = |state: &mut u64, low: f64, high: f64| -> f64 {
        low + rand_f64(state) * (high - low)
    };

    let mut rand_int = |state: &mut u64, max: usize| -> usize {
        let v = rand_f64(state);
        (v * max as f64) as usize % max
    };

    // --- True function + noisy data ---
    let true_fn = |x: f64| -> f64 {
        (2.0 * x).sin() + 0.5 * (5.0 * x).cos()
    };

    let n_train = 80;
    let mut x_train: Vec<f64> = (0..n_train)
        .map(|_| rand_range(&mut rng_state, 0.0, 4.0))
        .collect();
    x_train.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let y_train: Vec<f64> = x_train
        .iter()
        .map(|&x| true_fn(x) + rand_range(&mut rng_state, -0.3, 0.3))
        .collect();

    // Test points
    let n_test = 50;
    let x_test: Vec<f64> = (0..n_test)
        .map(|i| i as f64 * 4.0 / (n_test - 1) as f64)
        .collect();
    let y_true: Vec<f64> = x_test.iter().map(|&x| true_fn(x)).collect();

    // --- Simple decision tree stump (finds best split on 1D data) ---
    struct DecisionTree {
        splits: Vec<(f64, f64, f64)>, // (threshold, left_val, right_val)
    }

    fn build_tree(x: &[f64], y: &[f64], max_depth: usize) -> DecisionTree {
        let mut splits = Vec::new();
        build_recursive(x, y, &mut splits, 0, max_depth);
        DecisionTree { splits }
    }

    fn build_recursive(
        x: &[f64], y: &[f64], splits: &mut Vec<(f64, f64, f64)>,
        depth: usize, max_depth: usize,
    ) {
        if x.len() < 4 || depth >= max_depth {
            return;
        }
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;

        // Find best split
        let mut best_mse = f64::MAX;
        let mut best_thresh = x[0];
        let mut best_left = mean_y;
        let mut best_right = mean_y;

        let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
        sorted_indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        for i in 1..sorted_indices.len() {
            let thresh = (x[sorted_indices[i - 1]] + x[sorted_indices[i]]) / 2.0;
            let (mut sl, mut nl, mut sr, mut nr) = (0.0, 0.0, 0.0, 0.0);
            for j in 0..x.len() {
                if x[j] <= thresh {
                    sl += y[j];
                    nl += 1.0;
                } else {
                    sr += y[j];
                    nr += 1.0;
                }
            }
            if nl < 1.0 || nr < 1.0 {
                continue;
            }
            let ml = sl / nl;
            let mr = sr / nr;
            let mut mse = 0.0;
            for j in 0..x.len() {
                let pred = if x[j] <= thresh { ml } else { mr };
                mse += (y[j] - pred).powi(2);
            }
            if mse < best_mse {
                best_mse = mse;
                best_thresh = thresh;
                best_left = ml;
                best_right = mr;
            }
        }

        splits.push((best_thresh, best_left, best_right));
    }

    fn predict_tree(tree: &DecisionTree, x: f64) -> f64 {
        if tree.splits.is_empty() {
            return 0.0;
        }
        // Use last split as the deepest relevant one
        // Simple: use a recursive partition approach
        let mut val = 0.0;
        let mut count = 0;
        for &(thresh, left, right) in &tree.splits {
            val += if x <= thresh { left } else { right };
            count += 1;
        }
        val / count as f64
    }

    // For a proper simple implementation, let us use a piecewise-constant predictor
    // based on finding the best single split (decision stump)
    fn build_stump(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let mut best_mse = f64::MAX;
        let mut best = (x[0], mean_y, mean_y);

        for i in 0..x.len() {
            let thresh = x[i];
            let (mut sl, mut nl, mut sr, mut nr) = (0.0, 0.0, 0.0, 0.0);
            for j in 0..x.len() {
                if x[j] <= thresh {
                    sl += y[j]; nl += 1.0;
                } else {
                    sr += y[j]; nr += 1.0;
                }
            }
            if nl < 1.0 || nr < 1.0 { continue; }
            let ml = sl / nl;
            let mr = sr / nr;
            let mse: f64 = (0..x.len()).map(|j| {
                let p = if x[j] <= thresh { ml } else { mr };
                (y[j] - p).powi(2)
            }).sum();
            if mse < best_mse {
                best_mse = mse;
                best = (thresh, ml, mr);
            }
        }
        best
    }

    // Build a deeper piecewise-constant tree via recursive splitting
    struct PiecewiseTree {
        regions: Vec<(f64, f64, f64)>, // (low, high, value)
    }

    fn build_piecewise(x: &[f64], y: &[f64], depth: usize) -> PiecewiseTree {
        let mut regions = Vec::new();
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
        let sorted_x: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
        let sorted_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        split_region(&sorted_x, &sorted_y, 0, sorted_x.len(), depth, &mut regions);
        PiecewiseTree { regions }
    }

    fn split_region(
        x: &[f64], y: &[f64], start: usize, end: usize,
        depth: usize, regions: &mut Vec<(f64, f64, f64)>,
    ) {
        if end <= start || depth == 0 || (end - start) < 4 {
            if end > start {
                let mean: f64 = y[start..end].iter().sum::<f64>() / (end - start) as f64;
                let lo = if start == 0 { f64::NEG_INFINITY } else { x[start] };
                let hi = if end == x.len() { f64::INFINITY } else { x[end - 1] };
                regions.push((lo, hi, mean));
            }
            return;
        }

        // Find best split in this region
        let mut best_mse = f64::MAX;
        let mut best_split = start + 1;

        for s in (start + 2)..(end - 1) {
            let left_mean = y[start..s].iter().sum::<f64>() / (s - start) as f64;
            let right_mean = y[s..end].iter().sum::<f64>() / (end - s) as f64;
            let mse: f64 = y[start..s].iter().map(|&v| (v - left_mean).powi(2)).sum::<f64>()
                + y[s..end].iter().map(|&v| (v - right_mean).powi(2)).sum::<f64>();
            if mse < best_mse {
                best_mse = mse;
                best_split = s;
            }
        }

        split_region(x, y, start, best_split, depth - 1, regions);
        split_region(x, y, best_split, end, depth - 1, regions);
    }

    fn predict_piecewise(tree: &PiecewiseTree, x: f64) -> f64 {
        for &(lo, hi, val) in &tree.regions {
            if x >= lo && x <= hi {
                return val;
            }
        }
        // Fallback: closest region
        if let Some(&(_, _, val)) = tree.regions.first() {
            return val;
        }
        0.0
    }

    // --- Single deep tree ---
    let single_tree = build_piecewise(&x_train, &y_train, 6);
    let pred_single: Vec<f64> = x_test.iter().map(|&x| predict_piecewise(&single_tree, x)).collect();

    // --- Bagged ensemble ---
    let n_bags = 50;
    let mut bag_preds: Vec<Vec<f64>> = Vec::new();

    for _ in 0..n_bags {
        // Bootstrap sample: sample with replacement
        let mut x_bag = Vec::with_capacity(n_train);
        let mut y_bag = Vec::with_capacity(n_train);
        for _ in 0..n_train {
            let idx = rand_int(&mut rng_state, n_train);
            x_bag.push(x_train[idx]);
            y_bag.push(y_train[idx]);
        }

        let tree = build_piecewise(&x_bag, &y_bag, 6);
        let preds: Vec<f64> = x_test.iter().map(|&x| predict_piecewise(&tree, x)).collect();
        bag_preds.push(preds);
    }

    // Average bagged predictions
    let pred_bagged: Vec<f64> = (0..n_test)
        .map(|i| {
            let sum: f64 = bag_preds.iter().map(|p| p[i]).sum();
            sum / n_bags as f64
        })
        .collect();

    // --- MSE comparison ---
    let mse = |pred: &[f64], truth: &[f64]| -> f64 {
        pred.iter()
            .zip(truth.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / pred.len() as f64
    };

    let mse_single = mse(&pred_single, &y_true);
    let mse_bagged = mse(&pred_bagged, &y_true);

    println!("Single tree MSE:        {:.4}", mse_single);
    println!("Bagged ({} trees) MSE:  {:.4}", n_bags, mse_bagged);
    let reduction = (1.0 - mse_bagged / mse_single) * 100.0;
    println!("Variance reduction:     {:.1}%", reduction);
    println!();

    // --- Effect of number of bags on MSE ---
    let bag_counts = [1, 3, 5, 10, 20, 30, 50];
    println!("Trees --> MSE");
    println!("-----    -----");
    for &b in &bag_counts {
        let pred_b: Vec<f64> = (0..n_test)
            .map(|i| {
                let sum: f64 = bag_preds.iter().take(b).map(|p| p[i]).sum();
                sum / b as f64
            })
            .collect();
        let m = mse(&pred_b, &y_true);
        println!("  {:>3} -->  {:.4}", b, m);
    }

    // --- Sample predictions comparison ---
    println!("\nSample predictions (x, true, single_tree, bagged):");
    for i in (0..n_test).step_by(10) {
        println!(
            "  x={:.2}  true={:+.3}  single={:+.3}  bagged={:+.3}",
            x_test[i], y_true[i], pred_single[i], pred_bagged[i]
        );
    }

    // Metrics
    println!("\nkata_metric(\"mse_single_tree\", {:.4})", mse_single);
    println!("kata_metric(\"mse_bagged_50\", {:.4})", mse_bagged);
    println!("kata_metric(\"variance_reduction_pct\", {:.1})", reduction);
}
```

---

## Key Takeaways

- **Bagging reduces variance by averaging many overfit models.** Each tree is intentionally deep and overfit, but their average is stable.
- **Bootstrap sampling creates diversity.** Sampling with replacement means each tree sees a different version of the data, so they make different errors.
- **More trees is always better (with diminishing returns).** MSE drops rapidly at first, then plateaus. There is no overfitting risk from adding more trees.
- **Bagging does not reduce bias.** If each individual tree is biased (e.g., too shallow), averaging will not fix that. The base learner must be expressive.
