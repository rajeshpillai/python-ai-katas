# Cross-Validation

> Phase 4 — Model Evaluation & Selection | Kata 4.04

---

## Concept & Intuition

### What problem are we solving?

A single train-test split gives one estimate of model performance, but that estimate depends on which samples happened to land in each set. With limited data, this randomness creates high variance in the estimate. Cross-validation provides a more reliable performance estimate by systematically rotating which data is used for training and testing.

In K-fold cross-validation, the data is divided into K equal folds. The model is trained K times, each time using K-1 folds for training and the remaining fold for testing. The final performance metric is the average across all K folds. This gives every data point a chance to be in the test set exactly once, maximizing the use of available data while providing a robust performance estimate with confidence bounds.

In this kata, we implement K-fold and stratified K-fold cross-validation from scratch, and demonstrate how cross-validation provides more reliable performance estimates than a single split.

### Why naive approaches fail

A single 80/20 split uses only 80% of data for training and gives only one performance number. If you are unlucky, all the hard-to-classify examples end up in the test set. With a different random seed, you might get a very different result. Cross-validation eliminates this randomness by averaging over multiple splits. It also uses 100% of the data for both training and testing (across different folds), which is especially valuable when data is scarce.

### Mental models

- **K-fold as round-robin**: Every data point serves as test data exactly once. This eliminates the "lucky split" problem and gives a more reliable estimate.
- **Stratified folds preserve class ratios**: In each fold, the class distribution matches the overall dataset. This prevents folds where the minority class is underrepresented.
- **Cross-validation for model selection, not final evaluation**: Use CV to compare models and tune hyperparameters. Then train the final model on all training data and evaluate once on a truly held-out test set.

### Visual explanations

```
  5-Fold Cross-Validation:

  Fold 1: [TEST] [train] [train] [train] [train]  -> score_1
  Fold 2: [train] [TEST] [train] [train] [train]  -> score_2
  Fold 3: [train] [train] [TEST] [train] [train]  -> score_3
  Fold 4: [train] [train] [train] [TEST] [train]  -> score_4
  Fold 5: [train] [train] [train] [train] [TEST]  -> score_5

  Final = mean(scores) +/- std(scores)
```

---

## Hands-on Exploration

1. Implement K-fold cross-validation that partitions data into K folds.
2. Implement stratified K-fold that preserves class distributions.
3. Run CV with a simple classifier and compare variance across folds.
4. Show how CV performance compares to single train-test splits.

---

## Live Code

```rust
fn main() {
    println!("=== Cross-Validation ===\n");

    // Generate dataset
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    for _ in 0..30 {
        let x1 = 1.0 + randf(&mut rng) * 4.0;
        let x2 = 1.0 + randf(&mut rng) * 4.0;
        features.push(vec![x1, x2]);
        labels.push(0);
    }
    for _ in 0..30 {
        let x1 = 4.0 + randf(&mut rng) * 4.0;
        let x2 = 4.0 + randf(&mut rng) * 4.0;
        features.push(vec![x1, x2]);
        labels.push(1);
    }

    let n = features.len();
    println!("Dataset: {} samples (30 class 0, 30 class 1)\n", n);

    // Single train-test splits with different seeds
    println!("--- Single Train-Test Splits (showing variance) ---");
    let mut single_accs = Vec::new();
    for seed in 1..=10 {
        let mut idx: Vec<usize> = (0..n).collect();
        shuffle(&mut idx, seed);
        let train_n = 48;
        let x_train: Vec<Vec<f64>> = idx[..train_n].iter().map(|&i| features[i].clone()).collect();
        let y_train: Vec<usize> = idx[..train_n].iter().map(|&i| labels[i]).collect();
        let x_test: Vec<Vec<f64>> = idx[train_n..].iter().map(|&i| features[i].clone()).collect();
        let y_test: Vec<usize> = idx[train_n..].iter().map(|&i| labels[i]).collect();

        let acc = knn_evaluate(&x_train, &y_train, &x_test, &y_test, 5);
        single_accs.push(acc);
        println!("  Seed {:>2}: accuracy = {:.1}%", seed, acc * 100.0);
    }
    let single_mean = single_accs.iter().sum::<f64>() / single_accs.len() as f64;
    let single_std = (single_accs.iter().map(|a| (a - single_mean).powi(2)).sum::<f64>()
        / single_accs.len() as f64).sqrt();
    println!("  Mean: {:.1}% +/- {:.1}%\n", single_mean * 100.0, single_std * 100.0);

    // K-Fold Cross-Validation
    for k in &[3, 5, 10] {
        println!("--- {}-Fold Cross-Validation ---", k);
        let folds = k_fold_split(n, *k, 42);
        let mut fold_accs = Vec::new();

        for (fold_idx, test_indices) in folds.iter().enumerate() {
            let train_indices: Vec<usize> = (0..n).filter(|i| !test_indices.contains(i)).collect();
            let x_train: Vec<Vec<f64>> = train_indices.iter().map(|&i| features[i].clone()).collect();
            let y_train: Vec<usize> = train_indices.iter().map(|&i| labels[i]).collect();
            let x_test: Vec<Vec<f64>> = test_indices.iter().map(|&i| features[i].clone()).collect();
            let y_test: Vec<usize> = test_indices.iter().map(|&i| labels[i]).collect();

            let acc = knn_evaluate(&x_train, &y_train, &x_test, &y_test, 5);
            fold_accs.push(acc);
            println!("  Fold {}: train={}, test={}, accuracy={:.1}%",
                fold_idx + 1, x_train.len(), x_test.len(), acc * 100.0);
        }

        let cv_mean = fold_accs.iter().sum::<f64>() / fold_accs.len() as f64;
        let cv_std = (fold_accs.iter().map(|a| (a - cv_mean).powi(2)).sum::<f64>()
            / fold_accs.len() as f64).sqrt();
        println!("  CV Score: {:.1}% +/- {:.1}%\n", cv_mean * 100.0, cv_std * 100.0);
    }

    // Stratified K-Fold
    println!("--- Stratified 5-Fold (preserves class ratio) ---");
    let strat_folds = stratified_k_fold(&labels, 5, 42);
    let mut strat_accs = Vec::new();

    for (fold_idx, test_indices) in strat_folds.iter().enumerate() {
        let train_indices: Vec<usize> = (0..n).filter(|i| !test_indices.contains(i)).collect();
        let x_train: Vec<Vec<f64>> = train_indices.iter().map(|&i| features[i].clone()).collect();
        let y_train: Vec<usize> = train_indices.iter().map(|&i| labels[i]).collect();
        let x_test: Vec<Vec<f64>> = test_indices.iter().map(|&i| features[i].clone()).collect();
        let y_test: Vec<usize> = test_indices.iter().map(|&i| labels[i]).collect();

        let test_c0 = y_test.iter().filter(|&&l| l == 0).count();
        let test_c1 = y_test.iter().filter(|&&l| l == 1).count();
        let acc = knn_evaluate(&x_train, &y_train, &x_test, &y_test, 5);
        strat_accs.push(acc);
        println!("  Fold {}: test has {} c0, {} c1, accuracy={:.1}%",
            fold_idx + 1, test_c0, test_c1, acc * 100.0);
    }

    let strat_mean = strat_accs.iter().sum::<f64>() / strat_accs.len() as f64;
    let strat_std = (strat_accs.iter().map(|a| (a - strat_mean).powi(2)).sum::<f64>()
        / strat_accs.len() as f64).sqrt();
    println!("  Stratified CV Score: {:.1}% +/- {:.1}%\n", strat_mean * 100.0, strat_std * 100.0);

    // Model comparison using CV
    println!("--- Model Comparison via CV ---");
    println!("{:<12} {:>12} {:>12}", "K (KNN)", "CV Mean", "CV Std");
    println!("{}", "-".repeat(36));
    for &knn_k in &[1, 3, 5, 7, 11] {
        let mut accs = Vec::new();
        for test_indices in &strat_folds {
            let train_indices: Vec<usize> = (0..n).filter(|i| !test_indices.contains(i)).collect();
            let x_tr: Vec<Vec<f64>> = train_indices.iter().map(|&i| features[i].clone()).collect();
            let y_tr: Vec<usize> = train_indices.iter().map(|&i| labels[i]).collect();
            let x_te: Vec<Vec<f64>> = test_indices.iter().map(|&i| features[i].clone()).collect();
            let y_te: Vec<usize> = test_indices.iter().map(|&i| labels[i]).collect();
            accs.push(knn_evaluate(&x_tr, &y_tr, &x_te, &y_te, knn_k));
        }
        let mean = accs.iter().sum::<f64>() / accs.len() as f64;
        let std = (accs.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / accs.len() as f64).sqrt();
        println!("{:<12} {:>11.1}% {:>11.1}%", knn_k, mean * 100.0, std * 100.0);
    }

    kata_metric("single_split_std", single_std);
    kata_metric("cv5_mean", strat_mean);
    kata_metric("cv5_std", strat_std);
}

fn k_fold_split(n: usize, k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle(&mut indices, seed);
    let fold_size = n / k;
    let mut folds = Vec::new();
    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { start + fold_size };
        folds.push(indices[start..end].to_vec());
    }
    folds
}

fn stratified_k_fold(labels: &[usize], k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    let classes: Vec<usize> = { let mut c: Vec<usize> = labels.to_vec(); c.sort(); c.dedup(); c };

    for &class in &classes {
        let mut class_idx: Vec<usize> = labels.iter().enumerate()
            .filter(|(_, &l)| l == class).map(|(i, _)| i).collect();
        shuffle(&mut class_idx, seed + class as u64);
        for (i, idx) in class_idx.into_iter().enumerate() {
            folds[i % k].push(idx);
        }
    }
    folds
}

fn knn_evaluate(x_train: &[Vec<f64>], y_train: &[usize], x_test: &[Vec<f64>], y_test: &[usize], k: usize) -> f64 {
    let preds: Vec<usize> = x_test.iter().map(|x| knn_predict(x_train, y_train, x, k)).collect();
    preds.iter().zip(y_test.iter()).filter(|(p, a)| p == a).count() as f64 / y_test.len() as f64
}

fn knn_predict(x_train: &[Vec<f64>], y_train: &[usize], query: &[f64], k: usize) -> usize {
    let mut dists: Vec<(usize, f64)> = x_train.iter().enumerate().map(|(i, x)| {
        let d = x.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        (i, d)
    }).collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut votes = [0usize; 2];
    for i in 0..k.min(dists.len()) { votes[y_train[dists[i].0]] += 1; }
    if votes[1] > votes[0] { 1 } else { 0 }
}

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn shuffle(arr: &mut Vec<usize>, seed: u64) {
    let mut rng = seed;
    for i in (1..arr.len()).rev() { rng = lcg(rng); arr.swap(i, (rng % (i as u64 + 1)) as usize); }
}

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Cross-validation provides a more reliable performance estimate than a single train-test split by averaging over multiple partitions.
- Stratified K-fold ensures each fold preserves the original class distribution, critical for imbalanced or small datasets.
- CV is the standard tool for model selection and hyperparameter tuning. Always use it to compare models fairly.
- The standard deviation across folds indicates estimate reliability. High variance suggests the model is sensitive to which data it trains on.
