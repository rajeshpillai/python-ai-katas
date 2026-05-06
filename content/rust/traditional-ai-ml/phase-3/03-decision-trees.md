# Decision Trees

> Phase 3 — Supervised Learning: Classification | Kata 3.03

---

## Concept & Intuition

### What problem are we solving?

Decision trees classify data by learning a hierarchy of if-then rules from the training data. At each node, the tree asks a question about one feature ("Is income > $50K?"), splitting the data into branches. The process repeats recursively until reaching leaf nodes that assign a class label. Decision trees are arguably the most interpretable ML model — you can literally draw the decision process as a flowchart.

The key challenge is deciding which feature to split on at each node. Random splitting would create an inefficient tree. Instead, we measure how well each possible split separates the classes using metrics like Gini impurity or information gain (entropy). The best split is the one that creates the most homogeneous (pure) child nodes.

In this kata, we build a decision tree classifier from scratch using the CART (Classification and Regression Trees) algorithm with Gini impurity as the splitting criterion.

### Why naive approaches fail

A naive approach might split on the first feature at an arbitrary threshold. But this ignores which feature provides the most discriminative information. Worse, without a stopping criterion, the tree will keep splitting until every leaf contains a single training example — perfect training accuracy but catastrophic overfitting. Controlling tree depth, minimum samples per leaf, and minimum impurity decrease are essential for building useful trees.

### Mental models

- **20 questions game**: A decision tree is like playing 20 questions. Each question (split) should maximally reduce uncertainty about the answer (class label).
- **Gini impurity as uncertainty**: Gini = 1 - sum(p_i^2). A pure node (all one class) has Gini = 0. A perfectly mixed node has maximum Gini. Splits should reduce Gini.
- **Recursive partitioning**: The tree recursively divides the feature space into rectangles, each assigned to a class. The decision boundary is always axis-aligned.

### Visual explanations

```
  Decision Tree:                     Feature Space:

  [Is x1 > 5?]                      x2
       / \                           |  A A A|B B B
     Yes  No                         |  A A A|B B B
     /      \                        |  A A A|B B B
  [Class B] [Is x2 > 3?]            |-------+------
               / \                   |  A A A|A A A
             Yes  No                 |  A A A|A A A
             /      \                +-------------- x1
          [Class A] [Class A]             5     (x1 split)
                                          3     (x2 split)
```

---

## Hands-on Exploration

1. Implement Gini impurity and information gain calculations.
2. Build a recursive tree-building algorithm that selects the best split at each node.
3. Implement prediction by traversing the tree.
4. Control tree depth to observe the underfitting/overfitting tradeoff.

---

## Live Code

```rust
fn main() {
    println!("=== Decision Tree Classification ===\n");

    // Dataset: classify based on 2 features
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    // Class 0: lower-left quadrant
    for _ in 0..20 {
        rng = lcg(rng);
        let x1 = (rng as f64 / u64::MAX as f64) * 5.0;
        rng = lcg(rng);
        let x2 = (rng as f64 / u64::MAX as f64) * 5.0;
        features.push(vec![x1, x2]);
        labels.push(0);
    }

    // Class 1: upper-right quadrant
    for _ in 0..15 {
        rng = lcg(rng);
        let x1 = 4.0 + (rng as f64 / u64::MAX as f64) * 5.0;
        rng = lcg(rng);
        let x2 = 4.0 + (rng as f64 / u64::MAX as f64) * 5.0;
        features.push(vec![x1, x2]);
        labels.push(1);
    }

    // Class 2: lower-right
    for _ in 0..15 {
        rng = lcg(rng);
        let x1 = 5.0 + (rng as f64 / u64::MAX as f64) * 4.0;
        rng = lcg(rng);
        let x2 = (rng as f64 / u64::MAX as f64) * 4.0;
        features.push(vec![x1, x2]);
        labels.push(2);
    }

    let n = features.len();
    let n_classes = 3;

    // Split
    let train_n = 38;
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, 99);
    let x_train: Vec<Vec<f64>> = indices[..train_n].iter().map(|&i| features[i].clone()).collect();
    let y_train: Vec<usize> = indices[..train_n].iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = indices[train_n..].iter().map(|&i| features[i].clone()).collect();
    let y_test: Vec<usize> = indices[train_n..].iter().map(|&i| labels[i]).collect();

    println!("Dataset: {} samples, 3 classes", n);
    println!("Train: {}, Test: {}\n", train_n, n - train_n);

    // Build trees with different max depths
    println!("--- Depth Comparison ---");
    println!("{:<10} {:>12} {:>12} {:>8}", "MaxDepth", "Train Acc", "Test Acc", "Nodes");
    println!("{}", "-".repeat(44));

    for max_depth in &[1, 2, 3, 5, 10, 20] {
        let tree = build_tree(&x_train, &y_train, n_classes, *max_depth, 2);
        let train_preds: Vec<usize> = x_train.iter().map(|x| predict_tree(&tree, x)).collect();
        let test_preds: Vec<usize> = x_test.iter().map(|x| predict_tree(&tree, x)).collect();

        let train_acc = acc(&y_train, &train_preds);
        let test_acc = acc(&y_test, &test_preds);
        let node_count = count_nodes(&tree);

        println!("{:<10} {:>11.1}% {:>11.1}% {:>8}",
            max_depth, train_acc * 100.0, test_acc * 100.0, node_count);
    }

    // Best tree (depth 5)
    let best_tree = build_tree(&x_train, &y_train, n_classes, 5, 2);
    println!("\n--- Tree Structure (max_depth=5) ---");
    print_tree(&best_tree, 0);

    // Feature importance
    println!("\n--- Feature Importance ---");
    let importance = feature_importance(&best_tree, 2);
    let total: f64 = importance.iter().sum();
    for (i, imp) in importance.iter().enumerate() {
        let pct = if total > 0.0 { imp / total * 100.0 } else { 0.0 };
        let bar = "#".repeat((pct / 3.0) as usize);
        println!("  Feature {}: {:.1}% |{}", i, pct, bar);
    }

    // Gini impurity demonstration
    println!("\n--- Gini Impurity Examples ---");
    println!("  Pure node [10, 0, 0]:  Gini = {:.4}", gini(&[10, 0, 0]));
    println!("  Mixed [5, 5, 0]:       Gini = {:.4}", gini(&[5, 5, 0]));
    println!("  Equal [5, 5, 5]:       Gini = {:.4}", gini(&[5, 5, 5]));
    println!("  Mostly one [8, 1, 1]:  Gini = {:.4}", gini(&[8, 1, 1]));

    let test_preds: Vec<usize> = x_test.iter().map(|x| predict_tree(&best_tree, x)).collect();
    kata_metric("best_tree_test_accuracy", acc(&y_test, &test_preds));
    kata_metric("best_tree_nodes", count_nodes(&best_tree) as f64);
    kata_metric("gini_pure", gini(&[10, 0, 0]));
    kata_metric("gini_mixed", gini(&[5, 5, 5]));
}

#[derive(Debug)]
enum TreeNode {
    Leaf { class: usize },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        impurity_decrease: f64,
    },
}

fn gini(counts: &[usize]) -> f64 {
    let total: f64 = counts.iter().sum::<usize>() as f64;
    if total == 0.0 { return 0.0; }
    1.0 - counts.iter().map(|&c| (c as f64 / total).powi(2)).sum::<f64>()
}

fn build_tree(
    x: &[Vec<f64>], y: &[usize], n_classes: usize,
    max_depth: usize, min_samples: usize,
) -> TreeNode {
    let counts = class_counts(y, n_classes);
    let majority = counts.iter().enumerate().max_by_key(|(_, &c)| c).unwrap().0;

    if max_depth == 0 || y.len() < min_samples || gini(&counts) < 1e-10 {
        return TreeNode::Leaf { class: majority };
    }

    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_gini = f64::INFINITY;

    let n_features = x[0].len();
    let parent_gini = gini(&counts);

    for feat in 0..n_features {
        let mut vals: Vec<f64> = x.iter().map(|row| row[feat]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals.dedup();

        for w in vals.windows(2) {
            let threshold = (w[0] + w[1]) / 2.0;

            let mut left_counts = vec![0usize; n_classes];
            let mut right_counts = vec![0usize; n_classes];

            for (i, row) in x.iter().enumerate() {
                if row[feat] <= threshold {
                    left_counts[y[i]] += 1;
                } else {
                    right_counts[y[i]] += 1;
                }
            }

            let left_n: f64 = left_counts.iter().sum::<usize>() as f64;
            let right_n: f64 = right_counts.iter().sum::<usize>() as f64;
            let total = left_n + right_n;

            if left_n == 0.0 || right_n == 0.0 { continue; }

            let weighted_gini = (left_n / total) * gini(&left_counts)
                              + (right_n / total) * gini(&right_counts);

            if weighted_gini < best_gini {
                best_gini = weighted_gini;
                best_feature = feat;
                best_threshold = threshold;
            }
        }
    }

    if best_gini >= parent_gini - 1e-10 {
        return TreeNode::Leaf { class: majority };
    }

    let mut left_x = Vec::new();
    let mut left_y = Vec::new();
    let mut right_x = Vec::new();
    let mut right_y = Vec::new();

    for (i, row) in x.iter().enumerate() {
        if row[best_feature] <= best_threshold {
            left_x.push(row.clone());
            left_y.push(y[i]);
        } else {
            right_x.push(row.clone());
            right_y.push(y[i]);
        }
    }

    TreeNode::Split {
        feature: best_feature,
        threshold: best_threshold,
        left: Box::new(build_tree(&left_x, &left_y, n_classes, max_depth - 1, min_samples)),
        right: Box::new(build_tree(&right_x, &right_y, n_classes, max_depth - 1, min_samples)),
        impurity_decrease: parent_gini - best_gini,
    }
}

fn predict_tree(node: &TreeNode, x: &[f64]) -> usize {
    match node {
        TreeNode::Leaf { class } => *class,
        TreeNode::Split { feature, threshold, left, right, .. } => {
            if x[*feature] <= *threshold {
                predict_tree(left, x)
            } else {
                predict_tree(right, x)
            }
        }
    }
}

fn print_tree(node: &TreeNode, depth: usize) {
    let indent = "  ".repeat(depth);
    match node {
        TreeNode::Leaf { class } => println!("{}-> Class {}", indent, class),
        TreeNode::Split { feature, threshold, left, right, impurity_decrease } => {
            println!("{}[Feature {} <= {:.2}?] (gain={:.4})", indent, feature, threshold, impurity_decrease);
            println!("{}Yes:", indent);
            print_tree(left, depth + 1);
            println!("{}No:", indent);
            print_tree(right, depth + 1);
        }
    }
}

fn count_nodes(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf { .. } => 1,
        TreeNode::Split { left, right, .. } => 1 + count_nodes(left) + count_nodes(right),
    }
}

fn feature_importance(node: &TreeNode, n_features: usize) -> Vec<f64> {
    let mut importance = vec![0.0; n_features];
    collect_importance(node, &mut importance);
    importance
}

fn collect_importance(node: &TreeNode, importance: &mut Vec<f64>) {
    if let TreeNode::Split { feature, left, right, impurity_decrease, .. } = node {
        importance[*feature] += impurity_decrease;
        collect_importance(left, importance);
        collect_importance(right, importance);
    }
}

fn class_counts(y: &[usize], n_classes: usize) -> Vec<usize> {
    let mut counts = vec![0; n_classes];
    for &label in y { counts[label] += 1; }
    counts
}

fn acc(a: &[usize], p: &[usize]) -> f64 {
    a.iter().zip(p.iter()).filter(|(ai, pi)| ai == pi).count() as f64 / a.len() as f64
}

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    let mut rng = seed;
    for i in (1..arr.len()).rev() { rng = lcg(rng); arr.swap(i, (rng % (i as u64 + 1)) as usize); }
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Decision trees recursively split the feature space using the feature and threshold that best separate the classes, as measured by Gini impurity.
- Trees are highly interpretable — you can trace any prediction through a sequence of human-readable rules.
- Controlling tree depth and minimum samples per leaf prevents overfitting. Deeper trees memorize noise; shallower trees may underfit.
- Feature importance emerges naturally from how much each feature contributes to reducing impurity across all splits.
