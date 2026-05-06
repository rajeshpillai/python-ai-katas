# Random Forests

> Phase 6 — Ensemble Methods | Kata 6.2

---

## Concept & Intuition

### What problem are we solving?

Bagging helps, but there is a subtle problem: if one feature is very strong, every bootstrap tree will split on it first. This means all your trees look similar and their errors are correlated. Averaging correlated predictions gives less variance reduction than averaging independent ones. Random Forests fix this by adding a second layer of randomness: at each split, only a random subset of features is considered.

By forcing trees to sometimes ignore the best feature and find alternative splitting strategies, Random Forests produce a more diverse set of trees. This decorrelation is the key innovation over plain bagging. The result is that the ensemble benefits more from averaging, achieving lower error with the same number of trees.

Random Forests also provide a natural measure of feature importance (how much each feature contributes to reducing prediction error across all trees) and out-of-bag (OOB) error estimation (a free validation score computed from the ~37% of data each tree never sees during training).

### Why naive approaches fail

Plain bagging with unrestricted trees tends to produce similar-looking trees because the strongest feature dominates the first few splits everywhere. You end up with 100 trees that are 80% identical in structure. The effective ensemble size is much smaller than the actual tree count. Random feature selection at each split breaks this pattern and forces the ensemble to explore the full feature space, leading to genuinely different trees with more independent errors.

### Mental models

- **Diverse committee**: instead of letting every committee member read the same briefing document, you give each member a random subset of documents. They form different opinions and the vote becomes more informative.
- **Feature exploration**: by sometimes blocking the "obvious" feature, the forest discovers secondary patterns that a single tree would never find.
- **Free lunch (almost)**: OOB error gives you a validation score without setting aside any data -- every tree automatically has a held-out set.

### Visual explanations

```
Bagging (all features available):         Random Forest (random feature subsets):

Tree 1: split on F3 > 5                   Tree 1: [F1, F4, F7] --> split on F4
Tree 2: split on F3 > 5                   Tree 2: [F2, F3, F5] --> split on F3
Tree 3: split on F3 > 5                   Tree 3: [F1, F6, F3] --> split on F6
  (all trees look similar)                   (trees are genuinely different)

Correlation between trees: HIGH            Correlation between trees: LOW
Variance reduction: MODERATE               Variance reduction: HIGH

Feature Importance (from Random Forest):
  F3  ||||||||||||||||||||  0.42
  F4  ||||||||||||||        0.28
  F6  ||||||||              0.15
  F1  |||||                 0.09
  F2  |||                   0.06
```

---

## Hands-on Exploration

1. Train a Random Forest on a classification dataset and compare its accuracy to a single decision tree and to a plain bagging ensemble.
2. Extract and visualize feature importances. Identify which features the forest considers most predictive.
3. Compute the OOB error score and compare it to held-out test error -- they should be very close.
4. Vary the number of trees from 10 to 200 and observe how both OOB error and test error converge.

---

## Live Code

```rust
fn main() {
    // --- Simple RNG ---
    let mut rng: u64 = 42;
    let mut rand_f64 = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / 2147483648.0
    };
    let mut rand_int = |s: &mut u64, max: usize| -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as usize) % max
    };

    // --- Generate classification dataset with 6 features ---
    let n_samples = 300;
    let n_features = 6;
    let n_informative = 3; // first 3 features are informative

    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for _ in 0..n_samples {
        let mut row = Vec::new();
        for _ in 0..n_features {
            row.push(rand_f64(&mut rng) * 2.0 - 1.0);
        }
        // Label depends on first 3 features
        let score = 2.0 * row[0] - 1.5 * row[1] + row[2] + 0.3 * rand_f64(&mut rng);
        let label = if score > 0.0 { 1 } else { 0 };
        data.push(row);
        labels.push(label);
    }

    // Train/test split (80/20)
    let split = (n_samples as f64 * 0.8) as usize;
    let train_data = &data[..split];
    let train_labels = &labels[..split];
    let test_data = &data[split..];
    let test_labels = &labels[split..];

    // --- Decision stump on a single feature ---
    fn best_split_single_feature(
        data: &[Vec<f64>], labels: &[usize], feature: usize,
    ) -> (f64, f64) {
        // Returns (threshold, gini_reduction)
        let n = data.len() as f64;
        let pos: f64 = labels.iter().filter(|&&l| l == 1).count() as f64;
        let parent_gini = 1.0 - (pos / n).powi(2) - ((n - pos) / n).powi(2);

        let mut best_thresh = 0.0;
        let mut best_gain = 0.0;

        let mut vals: Vec<f64> = data.iter().map(|r| r[feature]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();

        for w in vals.windows(2) {
            let thresh = (w[0] + w[1]) / 2.0;
            let (mut lp, mut ln, mut rp, mut rn) = (0.0, 0.0, 0.0, 0.0);
            for (i, row) in data.iter().enumerate() {
                if row[feature] <= thresh {
                    if labels[i] == 1 { lp += 1.0; } else { ln += 1.0; }
                } else {
                    if labels[i] == 1 { rp += 1.0; } else { rn += 1.0; }
                }
            }
            let lt = lp + ln;
            let rt = rp + rn;
            if lt < 1.0 || rt < 1.0 { continue; }
            let lg = 1.0 - (lp / lt).powi(2) - (ln / lt).powi(2);
            let rg = 1.0 - (rp / rt).powi(2) - (rn / rt).powi(2);
            let gain = parent_gini - (lt / n) * lg - (rt / n) * rg;
            if gain > best_gain {
                best_gain = gain;
                best_thresh = thresh;
            }
        }
        (best_thresh, best_gain)
    }

    // --- Tree node for multi-feature decision tree ---
    #[derive(Clone)]
    enum TreeNode {
        Leaf(usize), // predicted class
        Split {
            feature: usize,
            threshold: f64,
            left: Box<TreeNode>,
            right: Box<TreeNode>,
        },
    }

    fn build_rf_tree(
        data: &[Vec<f64>], labels: &[usize], max_depth: usize,
        feature_subset_size: usize, rng: &mut u64, n_features: usize,
    ) -> TreeNode {
        build_rf_node(data, labels, 0, max_depth, feature_subset_size, rng, n_features)
    }

    fn build_rf_node(
        data: &[Vec<f64>], labels: &[usize], depth: usize, max_depth: usize,
        feature_subset_size: usize, rng: &mut u64, n_features: usize,
    ) -> TreeNode {
        if data.is_empty() {
            return TreeNode::Leaf(0);
        }
        let pos = labels.iter().filter(|&&l| l == 1).count();
        let neg = labels.len() - pos;

        // Pure node or max depth
        if pos == 0 || neg == 0 || depth >= max_depth || data.len() < 4 {
            return TreeNode::Leaf(if pos >= neg { 1 } else { 0 });
        }

        // Select random subset of features
        let mut candidates: Vec<usize> = Vec::new();
        let mut available: Vec<usize> = (0..n_features).collect();
        for _ in 0..feature_subset_size.min(n_features) {
            if available.is_empty() { break; }
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((*rng >> 33) as usize) % available.len();
            candidates.push(available.remove(idx));
        }

        // Find best split among candidates
        let mut best_feature = candidates[0];
        let mut best_threshold = 0.0;
        let mut best_gain = -1.0;

        for &f in &candidates {
            let (thresh, gain) = best_split_single_feature(data, labels, f);
            if gain > best_gain {
                best_gain = gain;
                best_feature = f;
                best_threshold = thresh;
            }
        }

        if best_gain <= 0.0 {
            return TreeNode::Leaf(if pos >= neg { 1 } else { 0 });
        }

        // Split data
        let mut left_data = Vec::new();
        let mut left_labels = Vec::new();
        let mut right_data = Vec::new();
        let mut right_labels = Vec::new();

        for (i, row) in data.iter().enumerate() {
            if row[best_feature] <= best_threshold {
                left_data.push(row.clone());
                left_labels.push(labels[i]);
            } else {
                right_data.push(row.clone());
                right_labels.push(labels[i]);
            }
        }

        TreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(build_rf_node(
                &left_data, &left_labels, depth + 1, max_depth,
                feature_subset_size, rng, n_features,
            )),
            right: Box::new(build_rf_node(
                &right_data, &right_labels, depth + 1, max_depth,
                feature_subset_size, rng, n_features,
            )),
        }
    }

    fn predict_node(node: &TreeNode, row: &[f64]) -> usize {
        match node {
            TreeNode::Leaf(c) => *c,
            TreeNode::Split { feature, threshold, left, right } => {
                if row[*feature] <= *threshold {
                    predict_node(left, row)
                } else {
                    predict_node(right, row)
                }
            }
        }
    }

    fn accuracy(predictions: &[usize], truth: &[usize]) -> f64 {
        let correct = predictions.iter().zip(truth).filter(|(p, t)| p == t).count();
        correct as f64 / truth.len() as f64
    }

    // --- Single tree (all features) ---
    let single_tree = build_rf_tree(
        train_data, train_labels, 8, n_features, &mut rng, n_features,
    );
    let single_preds: Vec<usize> = test_data.iter()
        .map(|r| predict_node(&single_tree, r))
        .collect();
    let single_acc = accuracy(&single_preds, test_labels);

    // --- Bagging (all features at each split) ---
    let n_trees = 50;
    let mut bagging_trees = Vec::new();
    let mut oob_predictions: Vec<Vec<usize>> = vec![Vec::new(); split];

    for t in 0..n_trees {
        let mut boot_data = Vec::new();
        let mut boot_labels = Vec::new();
        let mut in_bag = vec![false; split];

        for _ in 0..split {
            let idx = rand_int(&mut rng, split);
            boot_data.push(train_data[idx].clone());
            boot_labels.push(train_labels[idx]);
            in_bag[idx] = true;
        }

        let tree = build_rf_tree(
            &boot_data, &boot_labels, 8,
            n_features, // all features for bagging
            &mut rng, n_features,
        );

        // OOB predictions
        for i in 0..split {
            if !in_bag[i] {
                let pred = predict_node(&tree, &train_data[i]);
                oob_predictions[i].push(pred);
            }
        }

        bagging_trees.push(tree);
    }

    // Bagging test predictions (majority vote)
    let bag_preds: Vec<usize> = test_data.iter().map(|r| {
        let votes: Vec<usize> = bagging_trees.iter().map(|t| predict_node(t, r)).collect();
        let ones = votes.iter().filter(|&&v| v == 1).count();
        if ones > votes.len() / 2 { 1 } else { 0 }
    }).collect();
    let bag_acc = accuracy(&bag_preds, test_labels);

    // --- Random Forest (random feature subset at each split) ---
    let max_features = (n_features as f64).sqrt().ceil() as usize; // sqrt(n_features)
    let mut rf_trees = Vec::new();
    let mut rf_oob: Vec<Vec<usize>> = vec![Vec::new(); split];
    let mut feature_usage = vec![0.0_f64; n_features];

    for _ in 0..n_trees {
        let mut boot_data = Vec::new();
        let mut boot_labels = Vec::new();
        let mut in_bag = vec![false; split];

        for _ in 0..split {
            let idx = rand_int(&mut rng, split);
            boot_data.push(train_data[idx].clone());
            boot_labels.push(train_labels[idx]);
            in_bag[idx] = true;
        }

        let tree = build_rf_tree(
            &boot_data, &boot_labels, 8,
            max_features, // random subset for RF
            &mut rng, n_features,
        );

        // OOB predictions
        for i in 0..split {
            if !in_bag[i] {
                let pred = predict_node(&tree, &train_data[i]);
                rf_oob[i].push(pred);
            }
        }

        // Track feature importance via permutation (simplified)
        rf_trees.push(tree);
    }

    // RF test predictions
    let rf_preds: Vec<usize> = test_data.iter().map(|r| {
        let votes: Vec<usize> = rf_trees.iter().map(|t| predict_node(t, r)).collect();
        let ones = votes.iter().filter(|&&v| v == 1).count();
        if ones > votes.len() / 2 { 1 } else { 0 }
    }).collect();
    let rf_acc = accuracy(&rf_preds, test_labels);

    // OOB accuracy for RF
    let mut oob_correct = 0;
    let mut oob_total = 0;
    for i in 0..split {
        if !rf_oob[i].is_empty() {
            let ones = rf_oob[i].iter().filter(|&&v| v == 1).count();
            let majority = if ones > rf_oob[i].len() / 2 { 1 } else { 0 };
            if majority == train_labels[i] { oob_correct += 1; }
            oob_total += 1;
        }
    }
    let oob_acc = oob_correct as f64 / oob_total as f64;

    println!("=== Accuracy Comparison ===");
    println!("Single tree:       {:.4}", single_acc);
    println!("Bagging ({} trees): {:.4}", n_trees, bag_acc);
    println!("Random Forest:     {:.4}", rf_acc);
    println!("RF OOB accuracy:   {:.4}  (free, no test set needed)", oob_acc);
    println!();

    // --- Feature importance via permutation ---
    println!("=== Feature Importance (permutation-based) ===");
    let baseline_acc = rf_acc;
    let mut importances = vec![0.0_f64; n_features];

    for f in 0..n_features {
        // Permute feature f in test data
        let mut permuted: Vec<Vec<f64>> = test_data.to_vec();
        // Shuffle column f
        for i in 0..permuted.len() {
            let j = rand_int(&mut rng, permuted.len());
            let tmp = permuted[i][f];
            permuted[i][f] = permuted[j][f];
            permuted[j][f] = tmp;
        }

        let perm_preds: Vec<usize> = permuted.iter().map(|r| {
            let votes: Vec<usize> = rf_trees.iter().map(|t| predict_node(t, r)).collect();
            let ones = votes.iter().filter(|&&v| v == 1).count();
            if ones > votes.len() / 2 { 1 } else { 0 }
        }).collect();
        let perm_acc = accuracy(&perm_preds, test_labels);
        importances[f] = baseline_acc - perm_acc;
    }

    // Normalize
    let total_imp: f64 = importances.iter().map(|x| x.max(0.0)).sum();
    if total_imp > 0.0 {
        for v in importances.iter_mut() {
            *v = v.max(0.0) / total_imp;
        }
    }

    let mut sorted: Vec<(usize, f64)> = importances.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (f, imp) in &sorted {
        let bar: String = std::iter::repeat('|').take((imp * 40.0) as usize).collect();
        let informative = if *f < n_informative { " (informative)" } else { " (noise)" };
        println!("  F{} {:<40} {:.3}{}", f, bar, imp, informative);
    }

    // --- Convergence: accuracy vs number of trees ---
    println!("\n=== RF Accuracy vs Number of Trees ===");
    let tree_counts = [1, 5, 10, 20, 30, 50];
    for &tc in &tree_counts {
        let preds: Vec<usize> = test_data.iter().map(|r| {
            let votes: Vec<usize> = rf_trees.iter().take(tc)
                .map(|t| predict_node(t, r)).collect();
            let ones = votes.iter().filter(|&&v| v == 1).count();
            if ones > votes.len() / 2 { 1 } else { 0 }
        }).collect();
        let acc = accuracy(&preds, test_labels);
        println!("  {:>3} trees --> accuracy = {:.4}", tc, acc);
    }

    println!();
    println!("kata_metric(\"single_tree_accuracy\", {:.4})", single_acc);
    println!("kata_metric(\"bagging_accuracy\", {:.4})", bag_acc);
    println!("kata_metric(\"random_forest_accuracy\", {:.4})", rf_acc);
    println!("kata_metric(\"oob_accuracy\", {:.4})", oob_acc);
}
```

---

## Key Takeaways

- **Random Forests add feature randomness on top of bagging.** At each split, only a random subset of features is considered, decorrelating the trees.
- **Decorrelation is the key.** Less correlated trees means the average benefits more from cancellation of errors, yielding lower variance.
- **OOB error is a free validation metric.** Each tree's out-of-bag samples provide an unbiased error estimate without needing a separate validation set.
- **Feature importances reveal what the data cares about.** The forest naturally ranks features by how much they contribute to prediction quality via permutation importance.
