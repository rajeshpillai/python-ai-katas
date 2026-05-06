# Train-Test Split

> Phase 1 — Data Wrangling | Kata 1.06

---

## Concept & Intuition

### What problem are we solving?

A model that memorizes the training data but fails on new data is useless. This fundamental problem — the gap between training performance and real-world performance — is why we split data into separate training and testing sets. The training set is used to build the model; the test set is held out to evaluate how well the model generalizes to unseen data. Without this discipline, you have no reliable way to estimate real-world performance.

The split ratio, the splitting method, and how you handle the split all matter. A 50/50 split wastes half your data for training. A 95/5 split gives you unreliable test estimates. The standard is 70-80% train, 20-30% test. But the split must also be random to avoid systematic bias — if your data is sorted by date, taking the first 80% trains on old data and tests on recent data, which is a different problem (time-series prediction).

In this kata, we implement random train-test splitting with a reproducible pseudo-random number generator, stratified splitting that preserves class distributions, and demonstrate why evaluating on training data gives misleadingly optimistic results.

### Why naive approaches fail

Taking the first N rows as training and the rest as testing assumes the data is randomly ordered — which it rarely is. Databases are often sorted by ID, date, or category. This systematic split introduces bias: the training set might contain only one type of customer while the test set contains another. Even with random splitting, small test sets produce unreliable estimates with high variance. A model might score 95% on one random split and 85% on another purely due to which samples ended up in the test set.

### Mental models

- **Train-test as simulation**: The test set simulates future unseen data. If you peek at the test set during training (data leakage), your simulation is compromised.
- **Stratification preserves proportions**: In a dataset with 90% class A and 10% class B, a random split might put all class B samples in the test set. Stratified splitting ensures both sets reflect the original distribution.
- **The overfitting trap**: High training accuracy + low test accuracy = overfitting. The model memorized the training data instead of learning generalizable patterns.

### Visual explanations

```
  Full Dataset (100 samples)
  +--------------------------------------------------+
  | A A B A A A B A A A B A A A B A A A A B A A A A  |  (80% A, 20% B)
  +--------------------------------------------------+
                    |
           Random Stratified Split
                    |
    Training (80%)           Test (20%)
  +--------------------+    +----------+
  | A A B A A A B A A  |    | A B A A  |   Both sets maintain
  | A A B A A A A B A  |    | B A A A  |   ~80%/20% ratio
  | A A A B A A A A A  |    |          |
  | A B A A            |    |          |
  +--------------------+    +----------+
```

---

## Hands-on Exploration

1. Create a labeled dataset and implement random train-test splitting.
2. Build a simple pseudo-random number generator for reproducible splits.
3. Implement stratified splitting that preserves class distributions.
4. Demonstrate overfitting by comparing train vs. test accuracy.

---

## Live Code

```rust
fn main() {
    println!("=== Train-Test Split ===\n");

    // Create a dataset with features and labels
    // Predicting if a student passes (1) or fails (0) based on hours_studied, sleep_hours
    let features: Vec<Vec<f64>> = vec![
        vec![1.0, 4.0], vec![2.0, 5.0], vec![2.5, 6.0], vec![3.0, 7.0],
        vec![3.5, 6.5], vec![4.0, 7.0], vec![4.5, 8.0], vec![5.0, 7.5],
        vec![5.5, 6.0], vec![6.0, 7.0], vec![6.5, 8.0], vec![7.0, 7.5],
        vec![7.5, 6.5], vec![8.0, 7.0], vec![8.5, 8.0], vec![9.0, 7.5],
        vec![1.5, 3.0], vec![2.0, 4.0], vec![3.0, 5.0], vec![9.5, 8.0],
    ];

    let labels: Vec<f64> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let n_samples = features.len();
    let n_pass = labels.iter().filter(|&&l| l == 1.0).count();
    let n_fail = labels.iter().filter(|&&l| l == 0.0).count();
    println!("Dataset: {} samples ({} pass, {} fail)", n_samples, n_pass, n_fail);
    println!("Pass ratio: {:.1}%\n", n_pass as f64 / n_samples as f64 * 100.0);

    // Method 1: Simple random split (70/30)
    println!("--- Method 1: Random Split (70/30, seed=42) ---");
    let (train_idx, test_idx) = random_split(n_samples, 0.7, 42);
    print_split_stats(&labels, &train_idx, &test_idx);

    // Method 2: Stratified split
    println!("\n--- Method 2: Stratified Split (70/30, seed=42) ---");
    let (strat_train, strat_test) = stratified_split(&labels, 0.7, 42);
    print_split_stats(&labels, &strat_train, &strat_test);

    // Demonstrate overfitting
    println!("\n--- Overfitting Demonstration ---");
    println!("Using a simple threshold classifier (hours_studied > threshold)\n");

    let train_x: Vec<Vec<f64>> = strat_train.iter().map(|&i| features[i].clone()).collect();
    let train_y: Vec<f64> = strat_train.iter().map(|&i| labels[i]).collect();
    let test_x: Vec<Vec<f64>> = strat_test.iter().map(|&i| features[i].clone()).collect();
    let test_y: Vec<f64> = strat_test.iter().map(|&i| labels[i]).collect();

    // Try different thresholds
    println!("{:<12} {:>12} {:>12}", "Threshold", "Train Acc", "Test Acc");
    println!("{}", "-".repeat(36));

    let mut best_threshold = 0.0;
    let mut best_test_acc = 0.0;

    for t in 0..20 {
        let threshold = 1.0 + t as f64 * 0.5;
        let train_acc = evaluate_threshold(&train_x, &train_y, threshold);
        let test_acc = evaluate_threshold(&test_x, &test_y, threshold);
        println!("{:<12.1} {:>11.1}% {:>11.1}%", threshold, train_acc * 100.0, test_acc * 100.0);

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_threshold = threshold;
        }
    }

    println!("\nBest threshold by test accuracy: {:.1} ({:.1}%)",
        best_threshold, best_test_acc * 100.0);

    // Multiple random splits to show variance
    println!("\n--- Split Variance (10 different seeds) ---");
    println!("{:<8} {:>12} {:>12}", "Seed", "Train Acc", "Test Acc");
    println!("{}", "-".repeat(32));

    let mut test_accs = Vec::new();
    for seed in 1..=10 {
        let (tr, te) = stratified_split(&labels, 0.7, seed as u64);
        let tr_x: Vec<Vec<f64>> = tr.iter().map(|&i| features[i].clone()).collect();
        let tr_y: Vec<f64> = tr.iter().map(|&i| labels[i]).collect();
        let te_x: Vec<Vec<f64>> = te.iter().map(|&i| features[i].clone()).collect();
        let te_y: Vec<f64> = te.iter().map(|&i| labels[i]).collect();

        let train_acc = evaluate_threshold(&tr_x, &tr_y, best_threshold);
        let test_acc = evaluate_threshold(&te_x, &te_y, best_threshold);
        test_accs.push(test_acc);
        println!("{:<8} {:>11.1}% {:>11.1}%", seed, train_acc * 100.0, test_acc * 100.0);
    }

    let mean_acc = test_accs.iter().sum::<f64>() / test_accs.len() as f64;
    let std_acc = (test_accs.iter().map(|a| (a - mean_acc).powi(2)).sum::<f64>()
        / test_accs.len() as f64).sqrt();
    println!("\nTest accuracy: {:.1}% +/- {:.1}%", mean_acc * 100.0, std_acc * 100.0);

    kata_metric("best_threshold", best_threshold);
    kata_metric("best_test_accuracy", best_test_acc);
    kata_metric("mean_test_accuracy", mean_acc);
    kata_metric("std_test_accuracy", std_acc);
}

fn random_split(n: usize, train_ratio: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, seed);

    let train_size = (n as f64 * train_ratio) as usize;
    let train = indices[..train_size].to_vec();
    let test = indices[train_size..].to_vec();
    (train, test)
}

fn stratified_split(labels: &[f64], train_ratio: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    // Group indices by label
    let unique_labels = unique_f64(labels);
    let mut train = Vec::new();
    let mut test = Vec::new();

    for &label in &unique_labels {
        let mut indices: Vec<usize> = labels.iter().enumerate()
            .filter(|(_, &l)| l == label)
            .map(|(i, _)| i)
            .collect();

        shuffle(&mut indices, seed + label as u64);

        let train_size = (indices.len() as f64 * train_ratio).round() as usize;
        train.extend_from_slice(&indices[..train_size]);
        test.extend_from_slice(&indices[train_size..]);
    }

    (train, test)
}

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    // Simple LCG-based Fisher-Yates shuffle
    let mut rng = seed;
    for i in (1..arr.len()).rev() {
        rng = lcg_next(rng);
        let j = (rng % (i as u64 + 1)) as usize;
        arr.swap(i, j);
    }
}

fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn unique_f64(data: &[f64]) -> Vec<f64> {
    let mut unique = Vec::new();
    for &val in data {
        if !unique.iter().any(|&u| u == val) {
            unique.push(val);
        }
    }
    unique
}

fn evaluate_threshold(features: &[Vec<f64>], labels: &[f64], threshold: f64) -> f64 {
    let mut correct = 0;
    for (x, &y) in features.iter().zip(labels.iter()) {
        let pred = if x[0] >= threshold { 1.0 } else { 0.0 };
        if pred == y {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64
}

fn print_split_stats(labels: &[f64], train_idx: &[usize], test_idx: &[usize]) {
    let train_pass = train_idx.iter().filter(|&&i| labels[i] == 1.0).count();
    let train_fail = train_idx.iter().filter(|&&i| labels[i] == 0.0).count();
    let test_pass = test_idx.iter().filter(|&&i| labels[i] == 1.0).count();
    let test_fail = test_idx.iter().filter(|&&i| labels[i] == 0.0).count();

    println!("  Train: {} samples ({} pass, {} fail) - pass ratio: {:.1}%",
        train_idx.len(), train_pass, train_fail,
        train_pass as f64 / train_idx.len() as f64 * 100.0);
    println!("  Test:  {} samples ({} pass, {} fail) - pass ratio: {:.1}%",
        test_idx.len(), test_pass, test_fail,
        test_pass as f64 / test_idx.len() as f64 * 100.0);
}

fn kata_metric(name: &str, value: f64) {
    println!("[METRIC] {} = {:.4}", name, value);
}
```

---

## Key Takeaways

- Always evaluate models on held-out test data, never on training data. Training accuracy is a misleadingly optimistic estimate of real-world performance.
- Stratified splitting ensures that the class distribution in both train and test sets matches the original data, avoiding biased evaluation.
- Small test sets produce unreliable estimates — running multiple splits reveals the variance in your performance metric.
- The train-test split should be the very first step in your ML pipeline, before any data exploration or feature engineering that uses label information.
