# Imbalanced Classes

> Phase 3 — Supervised Learning: Classification | Kata 3.07

---

## Concept & Intuition

### What problem are we solving?

In many real-world classification problems, the classes are not equally represented. Fraud detection might have 99.9% legitimate transactions and 0.1% fraud. Disease screening might have 95% healthy patients and 5% positive cases. A naive classifier that always predicts the majority class achieves 99.9% accuracy on fraud detection — but catches zero fraud. High accuracy is meaningless when classes are imbalanced.

The fundamental issue is that standard algorithms optimize for overall accuracy, which is dominated by the majority class. When the minority class is the one you care about (fraud, disease, equipment failure), you need strategies that explicitly account for the imbalance: resampling techniques, cost-sensitive learning, and evaluation metrics that look beyond accuracy.

In this kata, we implement three approaches: oversampling the minority class, undersampling the majority class, and class-weighted learning. We also demonstrate why precision, recall, and F1-score are the right metrics for imbalanced problems.

### Why naive approaches fail

Standard accuracy gives a dangerously misleading picture. A model with 99% accuracy on a 99/1 imbalanced dataset might be completely useless — it could simply predict the majority class for everything. Gradient-based training is also biased: with 99x more majority samples, gradients are dominated by the majority class. The model learns to be very good at classifying the majority class and terrible at the minority class, which is exactly backwards for most applications.

### Mental models

- **Cost asymmetry**: Missing a fraud case (false negative) costs far more than flagging a legitimate transaction (false positive). The model should reflect these asymmetric costs.
- **Resampling as balancing the diet**: Oversampling gives the model more examples of the minority class to learn from. Undersampling removes majority class examples to prevent them from overwhelming the minority signal.
- **Metrics that matter**: Precision asks "Of all predicted positives, how many are real?" Recall asks "Of all real positives, how many did we find?" F1 balances both.

### Visual explanations

```
  Imbalanced Dataset:              After Oversampling:
  [- - - - - - - - - - +]         [- - - - - - - - - - + + + + +]
  90% negative, 10% positive       Duplicate minority to balance

  After Undersampling:             Class-Weighted:
  [- - + ]                         Each + sample counts 9x more
  Reduce majority to match         in the loss function
```

---

## Hands-on Exploration

1. Create a severely imbalanced dataset (95/5 split).
2. Train a baseline classifier and observe its failure on the minority class.
3. Implement oversampling (random duplication) and undersampling.
4. Implement class-weighted logistic regression.
5. Compare all approaches using precision, recall, and F1-score.

---

## Live Code

```rust
fn main() {
    println!("=== Imbalanced Classes ===\n");

    // Generate imbalanced dataset: 95% class 0, 5% class 1
    let mut rng = 42u64;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    // Class 0 (majority): 190 samples
    for _ in 0..190 {
        let x1 = 3.0 + randf(&mut rng) * 4.0;
        let x2 = 3.0 + randf(&mut rng) * 4.0;
        features.push(vec![x1, x2]);
        labels.push(0);
    }

    // Class 1 (minority): 10 samples
    for _ in 0..10 {
        let x1 = 6.0 + randf(&mut rng) * 2.0;
        let x2 = 6.0 + randf(&mut rng) * 2.0;
        features.push(vec![x1, x2]);
        labels.push(1);
    }

    let n = features.len();
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = labels.iter().filter(|&&l| l == 0).count();
    println!("Dataset: {} samples ({} negative, {} positive)", n, n_neg, n_pos);
    println!("Imbalance ratio: {:.1}:1\n", n_neg as f64 / n_pos as f64);

    // Standardize
    let (scaled, _, _) = standardize(&features);

    // Split (stratified)
    let (train_idx, test_idx) = stratified_split(&labels, 0.8, 42);
    let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| scaled[i].clone()).collect();
    let y_train: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
    let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| scaled[i].clone()).collect();
    let y_test: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

    let train_pos = y_train.iter().filter(|&&l| l == 1).count();
    let test_pos = y_test.iter().filter(|&&l| l == 1).count();
    println!("Train: {} ({} pos), Test: {} ({} pos)\n",
        x_train.len(), train_pos, x_test.len(), test_pos);

    // Strategy 0: Always predict majority class
    println!("--- Baseline: Always predict majority ---");
    let baseline_preds = vec![0usize; x_test.len()];
    print_metrics(&y_test, &baseline_preds);

    // Strategy 1: Standard logistic regression (no balancing)
    println!("\n--- Strategy 1: Standard Logistic Regression ---");
    let y_train_f: Vec<f64> = y_train.iter().map(|&l| l as f64).collect();
    let (w1, b1) = train_logistic(&x_train, &y_train_f, vec![1.0, 1.0], 0.1, 500);
    let preds1 = predict_binary(&x_test, &w1, b1);
    print_metrics(&y_test, &preds1);

    // Strategy 2: Random oversampling
    println!("\n--- Strategy 2: Random Oversampling ---");
    let (x_over, y_over) = random_oversample(&x_train, &y_train, &mut rng);
    let over_pos = y_over.iter().filter(|&&l| l == 1).count();
    let over_neg = y_over.iter().filter(|&&l| l == 0).count();
    println!("  Oversampled: {} samples ({} neg, {} pos)", x_over.len(), over_neg, over_pos);
    let y_over_f: Vec<f64> = y_over.iter().map(|&l| l as f64).collect();
    let (w2, b2) = train_logistic(&x_over, &y_over_f, vec![1.0, 1.0], 0.1, 500);
    let preds2 = predict_binary(&x_test, &w2, b2);
    print_metrics(&y_test, &preds2);

    // Strategy 3: Random undersampling
    println!("\n--- Strategy 3: Random Undersampling ---");
    let (x_under, y_under) = random_undersample(&x_train, &y_train, &mut rng);
    let under_pos = y_under.iter().filter(|&&l| l == 1).count();
    let under_neg = y_under.iter().filter(|&&l| l == 0).count();
    println!("  Undersampled: {} samples ({} neg, {} pos)", x_under.len(), under_neg, under_pos);
    let y_under_f: Vec<f64> = y_under.iter().map(|&l| l as f64).collect();
    let (w3, b3) = train_logistic(&x_under, &y_under_f, vec![1.0, 1.0], 0.1, 500);
    let preds3 = predict_binary(&x_test, &w3, b3);
    print_metrics(&y_test, &preds3);

    // Strategy 4: Class-weighted logistic regression
    println!("\n--- Strategy 4: Class-Weighted Learning ---");
    let weight_neg = 1.0;
    let weight_pos = n_neg as f64 / n_pos as f64;
    println!("  Class weights: neg={:.1}, pos={:.1}", weight_neg, weight_pos);
    let (w4, b4) = train_logistic(&x_train, &y_train_f, vec![weight_neg, weight_pos], 0.1, 500);
    let preds4 = predict_binary(&x_test, &w4, b4);
    print_metrics(&y_test, &preds4);

    // Summary comparison
    println!("\n--- Summary ---");
    println!("{:<25} {:>10} {:>10} {:>10} {:>10}",
        "Strategy", "Accuracy", "Precision", "Recall", "F1");
    println!("{}", "-".repeat(65));

    let strategies = vec![
        ("Always majority", &baseline_preds),
        ("Standard LR", &preds1),
        ("Oversampling", &preds2),
        ("Undersampling", &preds3),
        ("Class-weighted", &preds4),
    ];

    for (name, preds) in &strategies {
        let (_, prec, rec, f1) = compute_metrics(&y_test, preds);
        let acc_val = y_test.iter().zip(preds.iter()).filter(|(a, p)| a == p).count() as f64
            / y_test.len() as f64;
        println!("{:<25} {:>9.1}% {:>10.3} {:>10.3} {:>10.3}",
            name, acc_val * 100.0, prec, rec, f1);
    }

    let (_, _, _, f1_weighted) = compute_metrics(&y_test, &preds4);
    kata_metric("baseline_f1", 0.0);
    kata_metric("standard_lr_f1", { let (_, _, _, f1) = compute_metrics(&y_test, &preds1); f1 });
    kata_metric("oversampling_f1", { let (_, _, _, f1) = compute_metrics(&y_test, &preds2); f1 });
    kata_metric("weighted_f1", f1_weighted);
    kata_metric("imbalance_ratio", n_neg as f64 / n_pos as f64);
}

fn compute_metrics(actual: &[usize], predicted: &[usize]) -> (f64, f64, f64, f64) {
    let mut tp = 0; let mut fp = 0; let mut fn_ = 0; let mut tn = 0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        match (*a, *p) {
            (1, 1) => tp += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_ += 1,
            (0, 0) => tn += 1,
            _ => {}
        }
    }
    let accuracy = (tp + tn) as f64 / actual.len() as f64;
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    (accuracy, precision, recall, f1)
}

fn print_metrics(actual: &[usize], predicted: &[usize]) {
    let (acc, prec, rec, f1) = compute_metrics(actual, predicted);
    println!("  Accuracy:  {:.1}%", acc * 100.0);
    println!("  Precision: {:.3}", prec);
    println!("  Recall:    {:.3}", rec);
    println!("  F1-Score:  {:.3}", f1);
}

fn random_oversample(x: &[Vec<f64>], y: &[usize], rng: &mut u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let majority_n = y.iter().filter(|&&l| l == 0).count();
    let minority_idx: Vec<usize> = y.iter().enumerate().filter(|(_, &l)| l == 1).map(|(i, _)| i).collect();

    let mut new_x = x.to_vec();
    let mut new_y = y.to_vec();

    let to_add = majority_n - minority_idx.len();
    for _ in 0..to_add {
        *rng = lcg(*rng);
        let idx = minority_idx[(*rng as usize) % minority_idx.len()];
        new_x.push(x[idx].clone());
        new_y.push(1);
    }
    (new_x, new_y)
}

fn random_undersample(x: &[Vec<f64>], y: &[usize], rng: &mut u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let minority_n = y.iter().filter(|&&l| l == 1).count();
    let mut majority_idx: Vec<usize> = y.iter().enumerate().filter(|(_, &l)| l == 0).map(|(i, _)| i).collect();

    // Shuffle majority and take only minority_n
    for i in (1..majority_idx.len()).rev() {
        *rng = lcg(*rng);
        let j = (*rng % (i as u64 + 1)) as usize;
        majority_idx.swap(i, j);
    }
    majority_idx.truncate(minority_n);

    let minority_idx: Vec<usize> = y.iter().enumerate().filter(|(_, &l)| l == 1).map(|(i, _)| i).collect();
    let all_idx: Vec<usize> = majority_idx.into_iter().chain(minority_idx).collect();

    let new_x: Vec<Vec<f64>> = all_idx.iter().map(|&i| x[i].clone()).collect();
    let new_y: Vec<usize> = all_idx.iter().map(|&i| y[i]).collect();
    (new_x, new_y)
}

fn train_logistic(
    x: &[Vec<f64>], y: &[f64], class_weights: Vec<f64>, lr: f64, epochs: usize,
) -> (Vec<f64>, f64) {
    let n = x.len() as f64;
    let p = x[0].len();
    let mut w = vec![0.0; p];
    let mut b = 0.0;

    for _ in 0..epochs {
        let mut gw = vec![0.0; p];
        let mut gb = 0.0;
        for i in 0..x.len() {
            let z: f64 = x[i].iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + b;
            let pred = 1.0 / (1.0 + (-z).exp());
            let weight = if y[i] > 0.5 { class_weights[1] } else { class_weights[0] };
            let err = (pred - y[i]) * weight;
            for j in 0..p { gw[j] += err * x[i][j]; }
            gb += err;
        }
        for j in 0..p { w[j] -= lr * gw[j] / n; }
        b -= lr * gb / n;
    }
    (w, b)
}

fn predict_binary(x: &[Vec<f64>], w: &[f64], b: f64) -> Vec<usize> {
    x.iter().map(|xi| {
        let z: f64 = xi.iter().zip(w.iter()).map(|(x, w)| x * w).sum::<f64>() + b;
        let p = 1.0 / (1.0 + (-z).exp());
        if p >= 0.5 { 1 } else { 0 }
    }).collect()
}

fn stratified_split(labels: &[usize], train_ratio: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for class in 0..=1 {
        let mut idx: Vec<usize> = labels.iter().enumerate()
            .filter(|(_, &l)| l == class).map(|(i, _)| i).collect();
        let mut rng = seed + class as u64;
        for i in (1..idx.len()).rev() { rng = lcg(rng); idx.swap(i, (rng % (i as u64 + 1)) as usize); }
        let split = (idx.len() as f64 * train_ratio).round() as usize;
        train.extend_from_slice(&idx[..split]);
        test.extend_from_slice(&idx[split..]);
    }
    (train, test)
}

fn standardize(data: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = data.len() as f64;
    let p = data[0].len();
    let mut means = vec![0.0; p];
    let mut stds = vec![0.0; p];
    for j in 0..p { let col: Vec<f64> = data.iter().map(|r| r[j]).collect();
        means[j] = col.iter().sum::<f64>() / n;
        stds[j] = (col.iter().map(|x| (x - means[j]).powi(2)).sum::<f64>() / n).sqrt(); }
    let scaled = data.iter().map(|row| row.iter().enumerate().map(|(j, &v)| {
        if stds[j].abs() < 1e-10 { 0.0 } else { (v - means[j]) / stds[j] }
    }).collect()).collect();
    (scaled, means, stds)
}

fn randf(rng: &mut u64) -> f64 { *rng = lcg(*rng); (*rng as f64 / u64::MAX as f64 - 0.5) * 2.0 }

fn lcg(s: u64) -> u64 { s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) }

fn kata_metric(name: &str, value: f64) { println!("[METRIC] {} = {:.4}", name, value); }
```

---

## Key Takeaways

- Accuracy is misleading for imbalanced datasets. A classifier that always predicts the majority class can achieve very high accuracy while being completely useless.
- Precision, recall, and F1-score provide a more honest evaluation, especially for the minority class that typically matters most.
- Oversampling, undersampling, and class-weighted learning are complementary strategies that address imbalance at different stages of the pipeline.
- The right strategy depends on the domain: in fraud detection, missing fraud (low recall) is costly; in medical screening, false alarms (low precision) waste resources.
